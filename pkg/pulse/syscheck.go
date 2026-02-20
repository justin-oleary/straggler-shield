package pulse

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type gpuStats struct {
	SMClockMHz    int
	MaxSMClockMHz int
	TempC         int
	ECCErrors     int
}

// DetectGPUName returns the name of GPU 0 as reported by nvidia-smi, or
// "unknown" if nvidia-smi is unavailable. Exported for the benchmark harness.
func DetectGPUName() string {
	out, err := exec.Command(
		"nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0",
	).Output()
	if err != nil {
		return "unknown"
	}
	// Output may contain multiple lines on multi-GPU nodes when --id is omitted;
	// with --id=0 there is exactly one line. TrimSpace handles trailing newline.
	name := strings.TrimSpace(string(out))
	if name == "" {
		return "unknown"
	}
	return name
}

// detectGPUThreshold maps the detected GPU architecture to a calibrated GEMM
// latency threshold. Thresholds are derived from nominal FP32 GEMM performance
// on each architecture at P0 clocks with 20× headroom removed for tighter
// detection, then rounded to the nearest 5ms for operational margin.
//
// Architecture reference points (2048×2048 FP32 GEMM at P0):
//
//	A100 SXM4:  ~25ms  → threshold 100ms  (4× headroom)
//	H100 SXM5:  ~8ms   → threshold  35ms  (4× headroom)
//	H200:       ~7ms   → threshold  35ms  (shared with H100)
//	B200/GB200: ~3ms   → threshold  15ms  (5× headroom; Blackwell SM counts)
//
// Falls back to 500ms for unrecognized or unavailable hardware.
func detectGPUThreshold() time.Duration {
	name := strings.ToUpper(DetectGPUName())
	switch {
	case strings.Contains(name, "B200") || strings.Contains(name, "GB200"):
		return 15 * time.Millisecond
	case strings.Contains(name, "H100") || strings.Contains(name, "H200"):
		return 35 * time.Millisecond
	case strings.Contains(name, "A100"):
		return 100 * time.Millisecond
	default:
		return 500 * time.Millisecond
	}
}

// preflight checks every visible GPU for hard disqualifiers before the pulse
// workload runs. Returns a non-nil error on the first device that has:
//   - Uncorrectable ECC errors since last boot (bad HBM — no pulse needed)
//   - Idle temperature above maxIdleTempC (thermal recovery not complete)
//
// Proceeds silently if nvidia-smi is unavailable.
func preflight() error {
	stats, err := queryAllSMI()
	if err != nil {
		return nil // nvidia-smi absent or GPU not yet visible — proceed to pulse
	}

	for i, s := range stats {
		// Uncorrectable ECC errors indicate HBM instability. Per NVIDIA docs,
		// >8 per bank triggers row remapping; any nonzero count post-reboot
		// means the device had memory faults during the failure event.
		if s.ECCErrors > 0 {
			return fmt.Errorf("pre-flight GPU %d: %d uncorrectable ECC error(s) since last boot — quarantining without pulse", i, s.ECCErrors)
		}
		if s.TempC > maxIdleTempC {
			return fmt.Errorf("pre-flight GPU %d: idle temperature %d°C exceeds %d°C threshold (thermal recovery incomplete)", i, s.TempC, maxIdleTempC)
		}
	}
	return nil
}

// validateClocks queries all GPUs after the pulse workload to confirm each
// reached P0 under load. Catches the "clock speed stickiness" failure mode
// where clocks remain derated after a thermal event.
func validateClocks() error {
	stats, err := queryAllSMI()
	if err != nil {
		return nil // degrade gracefully
	}

	for i, s := range stats {
		if s.MaxSMClockMHz == 0 {
			continue // driver did not report max clock
		}
		threshold := int(float64(s.MaxSMClockMHz) * minClockFraction)
		if s.SMClockMHz < threshold {
			return fmt.Errorf(
				"post-pulse GPU %d: SM clock %dMHz below %.0f%% of max %dMHz — stuck in power-derated state under load",
				i, s.SMClockMHz, minClockFraction*100, s.MaxSMClockMHz,
			)
		}
	}
	return nil
}

// queryAllSMI returns stats for every visible GPU. The nvidia-smi output
// without --id returns one CSV row per device in ascending device order.
// In a DaemonSet the container sees only its assigned GPUs via the device
// plugin, so this always reflects the actual local device topology.
func queryAllSMI() ([]gpuStats, error) {
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,ecc.errors.uncorrected.aggregate.total",
		"--format=csv,noheader,nounits",
		// no --id: query all visible devices
	).Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi: %w", err)
	}

	parse := func(s string) int {
		s = strings.TrimSpace(s)
		if s == "N/A" || s == "[N/A]" {
			return 0
		}
		v, _ := strconv.Atoi(s)
		return v
	}

	var result []gpuStats
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		fields := strings.Split(line, ", ")
		if len(fields) != 4 {
			return nil, fmt.Errorf("nvidia-smi: unexpected field count in %q", line)
		}
		result = append(result, gpuStats{
			SMClockMHz:    parse(fields[0]),
			MaxSMClockMHz: parse(fields[1]),
			TempC:         parse(fields[2]),
			ECCErrors:     parse(fields[3]),
		})
	}
	return result, nil
}
