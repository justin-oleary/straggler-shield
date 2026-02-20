package k8s

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/justin-oleary/straggler-shield/pkg/pulse"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestReconcileNode(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string

		// node pre-condition in the fake API server
		node *corev1.Node

		// pulse mock
		pulseDuration time.Duration
		pulseErr      error

		// expected observable state after ReconcileNode
		wantTaint      bool
		wantEffect     corev1.TaintEffect // only checked when wantTaint == true
		wantPulseCalls int
		wantLogReason  string // substring expected in structured log output; empty = skip check
	}{
		{
			// Node rebooted after NVSentinel intervention; it had a zombie taint
			// from the previous failure. Pulse now returns healthy latency — the
			// taint must be removed so Slurm can resume the job.
			name:           "healthy node clears pre-existing quarantine",
			node:           quarantinedNode("gpu-node-0", 2*time.Minute),
			pulseDuration:  150 * time.Millisecond,
			pulseErr:       nil,
			wantTaint:      false,
			wantPulseCalls: 1,
		},
		{
			// Node rejoined the cluster but the GPU is fail-slow. The pulse
			// detects straggler latency and the controller must quarantine
			// the node before Slurm marks it available.
			name:           "zombie node — fail-slow straggler quarantined",
			node:           freshNode("gpu-node-1", 1*time.Minute),
			pulseDuration:  600 * time.Millisecond,
			pulseErr:       pulse.ErrStragglerDetected,
			wantTaint:      true,
			wantEffect:     corev1.TaintEffectNoSchedule,
			wantPulseCalls: 1,
		},
		{
			// Node has been Ready for hours. ReconcileNode must return
			// immediately — running a pulse on a training-active node
			// would spike memory bandwidth and perturb active jobs.
			name:           "steady-state node — zero overhead, pulse never called",
			node:           freshNode("gpu-node-2", 2*time.Hour),
			pulseDuration:  0,
			pulseErr:       nil,
			wantTaint:      false,
			wantPulseCalls: 0,
		},
		{
			// Node just rejoined; mean pulse latency is within the 500ms threshold
			// but run-to-run variance is high — the Falcon fail-slow pattern.
			// Intermittent thermal throttling causes some passes to spike while
			// the mean stays artificially low. The controller must quarantine the
			// node and log the specific variance-routing reason so operators can
			// distinguish erratic GPUs from uniformly degraded ones.
			name:           "erratic node — high variance quarantined with correct log reason",
			node:           freshNode("gpu-node-3", 3*time.Minute),
			pulseDuration:  300 * time.Millisecond,
			pulseErr:       pulse.ErrHighVariance,
			wantTaint:      true,
			wantEffect:     corev1.TaintEffectNoSchedule,
			wantPulseCalls: 1,
			wantLogReason:  "fail-slow variance pattern",
		},
	}

	for _, tc := range cases {
		tc := tc // capture for parallel sub-tests
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			clientset := fake.NewSimpleClientset(tc.node)

			calls := 0
			ctrl := newControllerWithPulse(clientset, func() (time.Duration, error) {
				calls++
				return tc.pulseDuration, tc.pulseErr
			})

			// Inject a per-test logger backed by a buffer when we need to assert
			// the structured log reason. Using a scoped logger avoids the data
			// race that would result from calling slog.SetDefault in parallel tests.
			var logBuf bytes.Buffer
			if tc.wantLogReason != "" {
				ctrl = ctrl.withLogger(slog.New(slog.NewTextHandler(&logBuf, nil)))
			}

			if err := ctrl.ReconcileNode(context.Background(), tc.node.Name); err != nil {
				t.Fatalf("ReconcileNode returned unexpected error: %v", err)
			}

			if calls != tc.wantPulseCalls {
				t.Errorf("pulse called %d time(s), want %d", calls, tc.wantPulseCalls)
			}

			got, err := clientset.CoreV1().Nodes().Get(
				context.Background(), tc.node.Name, metav1.GetOptions{},
			)
			if err != nil {
				t.Fatalf("Get node after reconcile: %v", err)
			}

			taint := findTaint(got, zombieTaintKey)
			hasTaint := taint != nil

			if hasTaint != tc.wantTaint {
				t.Errorf("hasTaint=%v, want %v (taints: %v)", hasTaint, tc.wantTaint, got.Spec.Taints)
			}

			if tc.wantTaint && taint != nil && taint.Effect != tc.wantEffect {
				t.Errorf("taint effect=%v, want %v", taint.Effect, tc.wantEffect)
			}

			if tc.wantLogReason != "" {
				logged := logBuf.String()
				if !strings.Contains(logged, tc.wantLogReason) {
					t.Errorf("log output missing reason %q\ngot: %s", tc.wantLogReason, logged)
				}
			}
		})
	}
}

// freshNode returns a node whose Ready condition just transitioned at -age.
func freshNode(name string, age time.Duration) *corev1.Node {
	return &corev1.Node{
		TypeMeta:   metav1.TypeMeta{Kind: "Node", APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: corev1.NodeStatus{
			Conditions: []corev1.NodeCondition{
				{
					Type:               corev1.NodeReady,
					Status:             corev1.ConditionTrue,
					LastTransitionTime: metav1.NewTime(time.Now().Add(-age)),
				},
			},
		},
	}
}

// quarantinedNode returns a freshly-Ready node that already carries the zombie
// taint — simulating a node that was quarantined in a previous failure cycle
// and has just rebooted.
func quarantinedNode(name string, age time.Duration) *corev1.Node {
	n := freshNode(name, age)
	n.Spec.Taints = []corev1.Taint{
		{Key: zombieTaintKey, Effect: corev1.TaintEffectNoSchedule, Value: "820ms"},
	}
	return n
}

// findTaint returns the first taint matching key, or nil if absent.
func findTaint(node *corev1.Node, key string) *corev1.Taint {
	for i := range node.Spec.Taints {
		if node.Spec.Taints[i].Key == key {
			return &node.Spec.Taints[i]
		}
	}
	return nil
}
