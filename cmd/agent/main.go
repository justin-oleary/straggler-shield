package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/justin-oleary/straggler-shield/pkg/k8s"
	_ "github.com/justin-oleary/straggler-shield/pkg/metrics" // register collectors

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// nodeLocks ensures ReconcileNode never runs concurrently for the same node.
// Values are *sync.Mutex; TryLock discards duplicate Ready events that fire
// while a pulse is already in flight.
var nodeLocks sync.Map

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	nodeName := os.Getenv("NODE_NAME")
	if nodeName == "" {
		slog.Error("NODE_NAME not set — mount the node name via the downward API")
		os.Exit(1)
	}

	cfg, err := rest.InClusterConfig()
	if err != nil {
		slog.Error("failed to load in-cluster config", "err", err)
		os.Exit(1)
	}
	clientset, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		slog.Error("failed to create clientset", "err", err)
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	ctrl := k8s.NewController(clientset)

	go serveMetrics(ctx)

	slog.Info("straggler-shield starting", "node", nodeName)
	run(ctx, ctrl, clientset, nodeName)
}

// serveMetrics runs the Prometheus /metrics endpoint on :9090 until ctx is
// cancelled. Exits cleanly on SIGINT/SIGTERM via srv.Shutdown.
func serveMetrics(ctx context.Context) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())

	srv := &http.Server{Addr: ":9090", Handler: mux}

	go func() {
		<-ctx.Done()
		if err := srv.Shutdown(context.Background()); err != nil {
			slog.Error("metrics server shutdown error", "err", err)
		}
	}()

	slog.Info("metrics server listening", "addr", ":9090")
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("metrics server failed", "err", err)
	}
}

// run watches the node's Ready condition indefinitely, reconnecting with
// exponential backoff whenever the API server closes the watch channel.
// The API server closes watch streams server-side every 5–10 minutes by design;
// this is normal and must never be treated as a fatal error.
func run(ctx context.Context, ctrl *k8s.Controller, clientset kubernetes.Interface, nodeName string) {
	const maxBackoff = 30 * time.Second
	backoff := time.Second

	for {
		if err := watchOnce(ctx, ctrl, clientset, nodeName); err != nil {
			if ctx.Err() != nil {
				return // context cancelled — clean shutdown
			}
			slog.Warn("watch ended, reconnecting", "node", nodeName, "err", err, "backoff", backoff)
		}
		if ctx.Err() != nil {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
			backoff = min(backoff*2, maxBackoff)
		}
	}
}

// watchOnce opens a single watch stream and processes node events until the
// stream closes or the context is cancelled. A closed channel is returned as
// nil so run() reconnects without logging a spurious error.
func watchOnce(ctx context.Context, ctrl *k8s.Controller, clientset kubernetes.Interface, nodeName string) error {
	w, err := clientset.CoreV1().Nodes().Watch(ctx, metav1.ListOptions{
		FieldSelector: "metadata.name=" + nodeName,
	})
	if err != nil {
		return fmt.Errorf("watch node %s: %w", nodeName, err)
	}
	defer w.Stop()

	var wasReady bool

	for {
		select {
		case <-ctx.Done():
			return nil
		case ev, ok := <-w.ResultChan():
			if !ok {
				return nil // server closed — caller reconnects
			}
			if ev.Type != watch.Modified && ev.Type != watch.Added {
				continue
			}
			node, ok := ev.Object.(*corev1.Node)
			if !ok {
				continue
			}

			ready := k8s.IsNodeReady(node)
			if ready && !wasReady {
				go tryReconcile(ctx, ctrl, nodeName)
			}
			wasReady = ready
		}
	}
}

// tryReconcile acquires a per-node TryLock before calling ReconcileNode.
// If a reconciliation is already in progress for this node, the event is
// discarded — the in-flight pulse will apply or clear the taint based on its
// result, and a duplicate run would observe the same GPU state anyway.
func tryReconcile(ctx context.Context, ctrl *k8s.Controller, nodeName string) {
	v, _ := nodeLocks.LoadOrStore(nodeName, &sync.Mutex{})
	mu := v.(*sync.Mutex)
	if !mu.TryLock() {
		slog.Info("reconcile already in progress — discarding duplicate ready event", "node", nodeName)
		return
	}
	defer mu.Unlock()

	if err := ctrl.ReconcileNode(ctx, nodeName); err != nil {
		slog.Error("reconcile failed", "node", nodeName, "err", err)
	}
}
