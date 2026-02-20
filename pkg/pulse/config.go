package pulse

import (
	"os"
	"strconv"
	"time"
)

// stragglerThreshold is the mean-latency ceiling per device.
// Resolution order:
//  1. PULSE_THRESHOLD_MS env var (operator override, always wins)
//  2. detectGPUThreshold() — architecture-calibrated value from nvidia-smi
//  3. 500ms fallback if nvidia-smi is unavailable or GPU is unrecognized
var stragglerThreshold = func() time.Duration {
	if s := os.Getenv("PULSE_THRESHOLD_MS"); s != "" {
		if v, err := strconv.ParseInt(s, 10, 64); err == nil && v > 0 {
			return time.Duration(v) * time.Millisecond
		}
	}
	return detectGPUThreshold()
}()

// maxCoefficientOfVar is the CV ceiling across runs on a single device.
// Override with PULSE_CV_MAX (float, e.g. "0.20").
var maxCoefficientOfVar = envFloat64("PULSE_CV_MAX", 0.20)

// minP2PBandwidthGBs is the minimum acceptable NVLink/PCIe P2P bandwidth.
// Override with P2P_MIN_GBS (float, e.g. "5.0").
var minP2PBandwidthGBs = envFloat64("P2P_MIN_GBS", 5.0)

// maxIdleTempC is the GPU temperature ceiling at pre-flight.
// Override with IDLE_TEMP_MAX (integer Celsius).
var maxIdleTempC = envInt("IDLE_TEMP_MAX", 70)

// minClockFraction is the post-pulse SM clock floor as a fraction of device
// maximum. Not env-configurable — changing requires recompile.
const minClockFraction = 0.5

// ThresholdMS returns the active GEMM latency threshold in milliseconds —
// either the env-var override or the architecture-calibrated value.
// Exported for the benchmark harness and structured log context.
func ThresholdMS() int64 {
	return stragglerThreshold.Milliseconds()
}

func envFloat64(key string, def float64) float64 {
	if s := os.Getenv(key); s != "" {
		if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
			return v
		}
	}
	return def
}

func envInt(key string, def int) int {
	if s := os.Getenv(key); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			return v
		}
	}
	return def
}
