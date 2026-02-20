//go:build cuda

package pulse

/*
#cgo CFLAGS:  -I${SRCDIR}/../../cuda
#cgo LDFLAGS: -L${SRCDIR}/../../cuda -lgpupulse -lcudart -lstdc++ -Wl,-rpath,/usr/local/lib
#include "gpu_pulse.h"
*/
import "C"
import (
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/justin-oleary/straggler-shield/pkg/metrics"
)

// pulseRuns is the number of timed GEMM passes per device per validation cycle.
const pulseRuns = 5

// RunPulse executes the full multi-GPU validation pipeline:
//  1. Pre-flight: ECC + idle temperature check on all devices
//  2. Per-device: N timed GEMM passes; records duration and CV to Prometheus
//  3. P2P ring: bandwidth check along the ring 0→1→…→N-1→0
//  4. Post-pulse: clock frequency validation on all devices
//
// Returns the worst-case mean duration and the first error encountered.
// Any device failure causes the entire node to be quarantined.
func RunPulse() (time.Duration, error) {
	if err := preflight(); err != nil {
		return 0, err
	}

	count := deviceCount()

	var worstMean time.Duration
	for dev := 0; dev < count; dev++ {
		mean, cv, err := runDevicePulse(dev)

		devLabel := strconv.Itoa(dev)
		metrics.PulseDuration.WithLabelValues(devLabel).Observe(mean.Seconds())
		metrics.PulseCV.WithLabelValues(devLabel).Set(cv)

		if err != nil {
			return mean, err
		}
		if mean > worstMean {
			worstMean = mean
		}
	}

	// Ring topology: 0→1, 1→2, …, N-1→0.
	// Catches any single broken NVLink segment, including links that do not
	// involve GPU 0, which a star check from GPU 0 would miss entirely.
	// Skip on single-GPU nodes where no inter-device links exist.
	if count > 1 {
		for i := 0; i < count; i++ {
			if err := checkP2P(i, (i+1)%count); err != nil {
				return worstMean, err
			}
		}
	}

	if err := validateClocks(); err != nil {
		return worstMean, &PulseFailure{
			Cause:          fmt.Errorf("%w: %v", ErrStragglerDetected, err),
			MeasuredValue:  float64(worstMean.Milliseconds()),
			ThresholdValue: float64(stragglerThreshold.Milliseconds()),
			Unit:           "ms",
		}
	}

	return worstMean, nil
}

// runDevicePulse runs pulseRuns timed GEMM passes on deviceID and returns the
// mean duration, coefficient of variation, and any error encountered.
func runDevicePulse(deviceID int) (mean time.Duration, cv float64, err error) {
	durations := make([]time.Duration, pulseRuns)

	for i := range durations {
		start := time.Now()
		rc := C.run_gpu_pulse(C.int(deviceID))
		elapsed := time.Since(start)

		switch int(rc) {
		case int(C.GPU_PULSE_OK):
			// ok
		case int(C.GPU_PULSE_ERR_CUDA):
			return elapsed, 0, fmt.Errorf("cuda error on GPU %d run %d (rc=%d)", deviceID, i+1, int(rc))
		case int(C.GPU_PULSE_ERR_OOM):
			return elapsed, 0, fmt.Errorf("out of device memory on GPU %d run %d (rc=%d)", deviceID, i+1, int(rc))
		default:
			return elapsed, 0, fmt.Errorf("gpu_pulse returned code %d on GPU %d run %d", int(rc), deviceID, i+1)
		}
		durations[i] = elapsed
	}

	mean, cv = computeStats(durations)

	if mean > stragglerThreshold {
		return mean, cv, &PulseFailure{
			Cause:          fmt.Errorf("GPU %d: %w (mean=%v)", deviceID, ErrStragglerDetected, mean),
			MeasuredValue:  float64(mean.Milliseconds()),
			ThresholdValue: float64(stragglerThreshold.Milliseconds()),
			Unit:           "ms",
		}
	}
	if cv > maxCoefficientOfVar {
		return mean, cv, &PulseFailure{
			Cause:          fmt.Errorf("GPU %d: %w (cv=%.3f)", deviceID, ErrHighVariance, cv),
			MeasuredValue:  cv,
			ThresholdValue: maxCoefficientOfVar,
			Unit:           "cv",
		}
	}
	return mean, cv, nil
}

// checkP2P times a 100 MiB cudaMemcpyPeer from src to dst and returns
// ErrInterconnectDegraded if the link is unavailable or bandwidth is too low.
// Called in ring order by RunPulse.
func checkP2P(src, dst int) error {
	var bwGBs C.double
	rc := C.run_p2p_check(C.int(src), C.int(dst), &bwGBs)

	switch int(rc) {
	case int(C.GPU_PULSE_OK):
		// ok — fall through to bandwidth check
	case int(C.GPU_PULSE_ERR_P2P):
		return &PulseFailure{
			Cause:          fmt.Errorf("GPU %d→%d: %w (peer access unavailable)", src, dst, ErrInterconnectDegraded),
			MeasuredValue:  0,
			ThresholdValue: minP2PBandwidthGBs,
			Unit:           "gbs",
		}
	default:
		return &PulseFailure{
			Cause:          fmt.Errorf("GPU %d→%d: %w (p2p check rc=%d)", src, dst, ErrInterconnectDegraded, int(rc)),
			MeasuredValue:  0,
			ThresholdValue: minP2PBandwidthGBs,
			Unit:           "gbs",
		}
	}

	bw := float64(bwGBs)
	if bw < minP2PBandwidthGBs {
		return &PulseFailure{
			Cause:          fmt.Errorf("GPU %d→%d: %w (%.2f GB/s < %.1f GB/s minimum)", src, dst, ErrInterconnectDegraded, bw, minP2PBandwidthGBs),
			MeasuredValue:  bw,
			ThresholdValue: minP2PBandwidthGBs,
			Unit:           "gbs",
		}
	}
	return nil
}

// deviceCount returns the number of CUDA-visible GPUs. Returns 1 on error so
// single-device validation always proceeds.
func deviceCount() int {
	n := int(C.gpu_device_count())
	if n < 1 {
		return 1
	}
	return n
}

// computeStats returns the mean duration and coefficient of variation (σ/μ).
func computeStats(durations []time.Duration) (mean time.Duration, cv float64) {
	var sum int64
	for _, d := range durations {
		sum += d.Nanoseconds()
	}
	meanNs := sum / int64(len(durations))

	var variance float64
	for _, d := range durations {
		delta := float64(d.Nanoseconds() - meanNs)
		variance += delta * delta
	}
	variance /= float64(len(durations))

	mean = time.Duration(meanNs)
	if meanNs > 0 {
		cv = math.Sqrt(variance) / float64(meanNs)
	}
	return
}
