// Package metrics registers the Prometheus collectors for the GPU validator.
// Import this package anywhere in the binary to ensure collectors are
// registered with the default registry before promhttp.Handler is called.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// PulseDuration is a per-device histogram of mean GEMM latency across the
	// five timed runs. The "device" label is the 0-based GPU index. Buckets
	// span 1ms → ~131s to cover both healthy A100 (~25ms) and worst-case
	// thermal stalls without underflow or overflow.
	PulseDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gpu_validator_pulse_duration_seconds",
			Help:    "Mean wall-clock duration of GPU GEMM pulse runs per device.",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 18),
		},
		[]string{"device"},
	)

	// PulseCV is a per-device gauge of the coefficient of variation (σ/μ)
	// across the last set of pulse runs. A healthy deterministic GEMM workload
	// produces CV well below 5%. Values above 20% trigger ErrHighVariance.
	PulseCV = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_validator_pulse_cv",
			Help: "Coefficient of variation (σ/μ) across GEMM pulse runs per device. >0.20 triggers quarantine.",
		},
		[]string{"device"},
	)

	// StragglerTotal counts quarantine events labelled by failure reason.
	//
	// Observed reason values:
	//   latency_threshold_exceeded   — mean GEMM latency > 500ms
	//   high_variance                — CV > 20% (fail-slow pattern)
	//   interconnect_degraded        — NVLink/P2P bandwidth below threshold
	//   pre_flight_failure           — ECC errors or thermal recovery incomplete
	StragglerTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gpu_validator_straggler_detected_total",
			Help: "Total number of nodes quarantined by the GPU validator, by failure reason.",
		},
		[]string{"reason"},
	)
)
