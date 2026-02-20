package k8s

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"time"

	"github.com/justin-oleary/straggler-shield/pkg/metrics"
	"github.com/justin-oleary/straggler-shield/pkg/pulse"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
)

const (
	zombieTaintKey  = "sunk.coreweave.com/zombie-quarantine"
	zombieCondition = corev1.NodeConditionType("GPUStraggler")
)

// readyTransitionWindow is how recently a Ready transition must have occurred
// for us to treat the node as "just joined or rebooted."
// Override with READY_WINDOW_SECONDS (integer seconds).
var readyTransitionWindow = func() time.Duration {
	if s := os.Getenv("READY_WINDOW_SECONDS"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			return time.Duration(v) * time.Second
		}
	}
	return 5 * time.Minute
}()

// pulseFunc is the GPU pulse runner signature.
// Defined as a type so tests can inject a mock without CGO or a real GPU.
type pulseFunc func() (time.Duration, error)

// Controller runs GPU pulse validation when nodes (re)join the cluster.
type Controller struct {
	client   kubernetes.Interface
	runPulse pulseFunc
	logger   *slog.Logger
}

// NewController returns a Controller wired to the real CUDA pulse.
func NewController(client kubernetes.Interface) *Controller {
	return &Controller{client: client, runPulse: pulse.RunPulse, logger: slog.Default()}
}

// newControllerWithPulse injects a custom pulse function.
// Only for use in unit tests — avoids CGO and GPU dependencies.
func newControllerWithPulse(client kubernetes.Interface, fn pulseFunc) *Controller {
	return &Controller{client: client, runPulse: fn, logger: slog.Default()}
}

// withLogger swaps the controller's logger. Used in tests to capture structured
// log output without touching the global default logger.
func (c *Controller) withLogger(l *slog.Logger) *Controller {
	c.logger = l
	return c
}

// ReconcileNode is the primary entry point. It should be called whenever a node
// transitions to Ready (watch event or informer sync). It:
//  1. Checks whether the node just joined or rebooted.
//  2. Runs pulse.RunPulse() against the local GPU.
//  3. Removes the zombie quarantine taint if the pulse passes.
//  4. Applies the taint and emits a structured MFU evidence log if it fails.
func (c *Controller) ReconcileNode(ctx context.Context, nodeName string) error {
	node, err := c.client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get node %s: %w", nodeName, err)
	}

	if !justBecameReady(node, readyTransitionWindow) {
		return nil // steady-state node — nothing to do
	}

	c.logger.Info("node ready after join/reboot — running GPU pulse", "node", nodeName)

	elapsed, err := c.runPulse()
	if err == nil {
		c.logger.Info("GPU pulse passed", "node", nodeName, "elapsed", elapsed)
		return c.removeTaint(ctx, nodeName, node)
	}

	if pulse.IsStragglerErr(err) {
		logReason := "latency threshold exceeded"
		promReason := "latency_threshold_exceeded"
		switch {
		case errors.Is(err, pulse.ErrHighVariance):
			logReason = "fail-slow variance pattern (high CV across runs)"
			promReason = "high_variance"
		case errors.Is(err, pulse.ErrInterconnectDegraded):
			logReason = "NVLink/P2P interconnect degraded"
			promReason = "interconnect_degraded"
		}

		// Build the structured MFU evidence log. If the error carries a
		// PulseFailure, include the exact measured and threshold values so
		// the log record is self-contained proof of why the node was caught.
		logArgs := []any{
			"node_name", nodeName,
			"failure_reason", logReason,
			"elapsed_ms", elapsed.Milliseconds(),
		}
		var detail *pulse.PulseFailure
		if errors.As(err, &detail) {
			logArgs = append(logArgs,
				"measured_value", detail.MeasuredValue,
				"threshold_value", detail.ThresholdValue,
				"unit", detail.Unit,
			)
		}
		c.logger.Warn("zombie node quarantined", logArgs...)

		metrics.StragglerTotal.WithLabelValues(promReason).Inc()
		return c.applyTaint(ctx, nodeName, node, elapsed)
	}

	// Hard failure (ECC errors, thermal, CUDA crash) — also quarantine.
	c.logger.Error("GPU pulse hard failure — quarantining node",
		"node_name", nodeName,
		"failure_reason", "pre_flight_failure",
		"err", err,
	)
	metrics.StragglerTotal.WithLabelValues("pre_flight_failure").Inc()
	return c.applyTaint(ctx, nodeName, node, elapsed)
}

// justBecameReady returns true when the node's Ready=True condition transitioned
// within the given window. Nodes that have been stable for hours return false.
func justBecameReady(node *corev1.Node, within time.Duration) bool {
	for _, c := range node.Status.Conditions {
		if c.Type == corev1.NodeReady && c.Status == corev1.ConditionTrue {
			return time.Since(c.LastTransitionTime.Time) < within
		}
	}
	return false
}

// IsNodeReady reports whether the node's Ready condition is True.
// Exported for use by the watch loop in cmd/agent.
func IsNodeReady(node *corev1.Node) bool {
	for _, c := range node.Status.Conditions {
		if c.Type == corev1.NodeReady {
			return c.Status == corev1.ConditionTrue
		}
	}
	return false
}

// applyTaint adds the zombie-quarantine NoSchedule taint to the node spec and
// records a GPUStraggler condition in the status subresource. Idempotent.
func (c *Controller) applyTaint(ctx context.Context, nodeName string, node *corev1.Node, elapsed time.Duration) error {
	// skip if already tainted
	for _, t := range node.Spec.Taints {
		if t.Key == zombieTaintKey {
			return nil
		}
	}

	type specPatch struct {
		Spec struct {
			Taints []corev1.Taint `json:"taints"`
		} `json:"spec"`
	}
	sp := specPatch{}
	sp.Spec.Taints = append(node.Spec.Taints, corev1.Taint{
		Key:    zombieTaintKey,
		Value:  elapsed.String(),
		Effect: corev1.TaintEffectNoSchedule,
	})
	specBytes, err := json.Marshal(sp)
	if err != nil {
		return fmt.Errorf("marshal taint patch: %w", err)
	}
	if _, err := c.client.CoreV1().Nodes().Patch(
		ctx, nodeName, types.MergePatchType, specBytes, metav1.PatchOptions{},
	); err != nil {
		return fmt.Errorf("patch node spec: %w", err)
	}

	// record why the node was quarantined
	type statusPatch struct {
		Status struct {
			Conditions []corev1.NodeCondition `json:"conditions"`
		} `json:"status"`
	}
	cond := corev1.NodeCondition{
		Type:               zombieCondition,
		Status:             corev1.ConditionTrue,
		Reason:             "StragglerDetected",
		Message:            fmt.Sprintf("GPU pulse took %s (threshold 500ms)", elapsed),
		LastTransitionTime: metav1.Now(),
	}
	st := statusPatch{}
	st.Status.Conditions = upsertCondition(node.Status.Conditions, cond)
	statusBytes, err := json.Marshal(st)
	if err != nil {
		return fmt.Errorf("marshal status patch: %w", err)
	}
	if _, err := c.client.CoreV1().Nodes().Patch(
		ctx, nodeName, types.MergePatchType, statusBytes,
		metav1.PatchOptions{}, "status",
	); err != nil {
		return fmt.Errorf("patch node status: %w", err)
	}

	return nil
}

// removeTaint strips the zombie-quarantine taint and clears the GPUStraggler
// condition. Called when a previously quarantined node passes the pulse. Idempotent.
func (c *Controller) removeTaint(ctx context.Context, nodeName string, node *corev1.Node) error {
	filtered := make([]corev1.Taint, 0, len(node.Spec.Taints))
	for _, t := range node.Spec.Taints {
		if t.Key != zombieTaintKey {
			filtered = append(filtered, t)
		}
	}
	if len(filtered) == len(node.Spec.Taints) {
		return nil // zombie taint was not present
	}

	type specPatch struct {
		Spec struct {
			Taints []corev1.Taint `json:"taints"`
		} `json:"spec"`
	}
	sp := specPatch{}
	sp.Spec.Taints = filtered
	specBytes, err := json.Marshal(sp)
	if err != nil {
		return fmt.Errorf("marshal taint removal patch: %w", err)
	}
	if _, err := c.client.CoreV1().Nodes().Patch(
		ctx, nodeName, types.MergePatchType, specBytes, metav1.PatchOptions{},
	); err != nil {
		return fmt.Errorf("patch node spec (remove taint): %w", err)
	}

	// clear the condition
	type statusPatch struct {
		Status struct {
			Conditions []corev1.NodeCondition `json:"conditions"`
		} `json:"status"`
	}
	cond := corev1.NodeCondition{
		Type:               zombieCondition,
		Status:             corev1.ConditionFalse,
		Reason:             "PulsePassed",
		Message:            "GPU pulse passed; node cleared for Slurm scheduling",
		LastTransitionTime: metav1.Now(),
	}
	st := statusPatch{}
	st.Status.Conditions = upsertCondition(node.Status.Conditions, cond)
	statusBytes, err := json.Marshal(st)
	if err != nil {
		return fmt.Errorf("marshal status patch (clear condition): %w", err)
	}
	if _, err := c.client.CoreV1().Nodes().Patch(
		ctx, nodeName, types.MergePatchType, statusBytes,
		metav1.PatchOptions{}, "status",
	); err != nil {
		return fmt.Errorf("patch node status (clear condition): %w", err)
	}

	c.logger.Info("zombie taint removed — node cleared for Slurm", "node_name", nodeName)
	return nil
}

func upsertCondition(conditions []corev1.NodeCondition, c corev1.NodeCondition) []corev1.NodeCondition {
	for i, existing := range conditions {
		if existing.Type == c.Type {
			conditions[i] = c
			return conditions
		}
	}
	return append(conditions, c)
}
