// benchmark is a standalone CLI for validating and demonstrating the
// straggler-shield pulse pipeline without a running Kubernetes cluster.
//
// It supports four simulated scenarios (no GPU required) and one real mode
// that invokes the full CUDA pulse pipeline (requires -tags cuda and a GPU).
//
// Usage:
//
//	benchmark [--scenario=<name>] [--count=<n>]
//
// Scenarios:
//
//	real            Run the actual CUDA pulse against the local GPU(s).
//	                Requires a GPU and the -tags cuda build.
//	healthy         Simulate a GPU passing all checks cleanly.
//	straggler       Simulate a GPU exceeding the mean-latency threshold.
//	high-variance   Simulate a fail-slow GPU: acceptable mean, high CV.
//	p2p-degraded    Simulate a broken NVLink ring segment.
//
// Output is a structured JSON report written to stdout. Each run's
// measured_value and threshold_value fields are the literal numbers used
// to make the quarantine decision — suitable for direct use as MFU evidence.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/justin-oleary/straggler-shield/pkg/pulse"
)

// runResult captures the outcome of a single benchmark run.
type runResult struct {
	Run            int     `json:"run"`
	ElapsedMS      int64   `json:"elapsed_ms"`
	Verdict        string  `json:"verdict"` // "pass" | "fail"
	FailureReason  string  `json:"failure_reason,omitempty"`
	MeasuredValue  float64 `json:"measured_value,omitempty"`
	ThresholdValue float64 `json:"threshold_value,omitempty"`
	Unit           string  `json:"unit,omitempty"` // "ms" | "cv" | "gbs"
}

type reportSummary struct {
	Total          int    `json:"total"`
	Passed         int    `json:"passed"`
	Failed         int    `json:"failed"`
	WorstElapsedMS int64  `json:"worst_elapsed_ms"`
	Verdict        string `json:"verdict"` // "HEALTHY" | "STRAGGLER"
}

type report struct {
	Timestamp          string        `json:"timestamp"`
	Hostname           string        `json:"hostname"`
	GPUArch            string        `json:"gpu_arch"`
	CalibratedThreshMS int64         `json:"calibrated_threshold_ms"`
	Scenario           string        `json:"scenario"`
	Runs               []runResult   `json:"runs"`
	Summary            reportSummary `json:"summary"`
}

// scenario is a function that mimics the pulse.RunPulse signature.
type scenario func() (time.Duration, error)

// scenarios maps CLI names to pulse functions. Simulated scenarios are
// threshold-aware — elapsed values scale with the calibrated device threshold
// so the numbers in the report are plausible for the detected hardware.
var scenarios = map[string]scenario{
	// real: invokes the actual CUDA pipeline. Works with -tags cuda + GPU;
	// returns a "built without cuda support" error in stub builds.
	"real": pulse.RunPulse,

	// healthy: mean latency at 25% of threshold — clearly passing on any arch.
	"healthy": func() (time.Duration, error) {
		elapsed := time.Duration(pulse.ThresholdMS()/4) * time.Millisecond
		if elapsed < time.Millisecond {
			elapsed = time.Millisecond
		}
		return elapsed, nil
	},

	// straggler: mean latency at 5× threshold — unambiguous latency failure.
	"straggler": func() (time.Duration, error) {
		threshMS := pulse.ThresholdMS()
		elapsed := time.Duration(threshMS*5) * time.Millisecond
		return elapsed, &pulse.PulseFailure{
			Cause:          fmt.Errorf("GPU 0: %w (mean=%dms)", pulse.ErrStragglerDetected, threshMS*5),
			MeasuredValue:  float64(threshMS * 5),
			ThresholdValue: float64(threshMS),
			Unit:           "ms",
		}
	},

	// high-variance: mean at 33% of threshold (passes latency check) but
	// CV = 0.35 — a textbook fail-slow Falcon-paper pattern.
	"high-variance": func() (time.Duration, error) {
		elapsed := time.Duration(pulse.ThresholdMS()/3) * time.Millisecond
		if elapsed < time.Millisecond {
			elapsed = time.Millisecond
		}
		return elapsed, &pulse.PulseFailure{
			Cause:          fmt.Errorf("GPU 0: %w (cv=0.350)", pulse.ErrHighVariance),
			MeasuredValue:  0.350,
			ThresholdValue: 0.20,
			Unit:           "cv",
		}
	},

	// p2p-degraded: NVLink ring segment 2→3 measuring 1.2 GB/s against the
	// 5 GB/s minimum — simulates a partially failed NVSwitch fabric port.
	"p2p-degraded": func() (time.Duration, error) {
		return 0, &pulse.PulseFailure{
			Cause:          fmt.Errorf("GPU 2→3: %w (1.20 GB/s < 5.0 GB/s minimum)", pulse.ErrInterconnectDegraded),
			MeasuredValue:  1.20,
			ThresholdValue: 5.0,
			Unit:           "gbs",
		}
	},
}

func main() {
	scenarioName := flag.String("scenario", "real",
		"pulse scenario: real, healthy, straggler, high-variance, p2p-degraded")
	count := flag.Int("count", 3, "number of benchmark runs")
	flag.Parse()

	fn, ok := scenarios[*scenarioName]
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown scenario %q\nvalid: real, healthy, straggler, high-variance, p2p-degraded\n", *scenarioName)
		os.Exit(1)
	}
	if *count < 1 {
		fmt.Fprintf(os.Stderr, "--count must be >= 1\n")
		os.Exit(1)
	}

	hostname, _ := os.Hostname()

	runs := execute(fn, *count)
	r := report{
		Timestamp:          time.Now().UTC().Format(time.RFC3339),
		Hostname:           hostname,
		GPUArch:            pulse.DetectGPUName(),
		CalibratedThreshMS: pulse.ThresholdMS(),
		Scenario:           *scenarioName,
		Runs:               runs,
		Summary:            summarize(runs),
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		fmt.Fprintf(os.Stderr, "json encode: %v\n", err)
		os.Exit(1)
	}
}

// execute runs fn count times and records each result.
func execute(fn scenario, count int) []runResult {
	results := make([]runResult, 0, count)
	for i := 1; i <= count; i++ {
		elapsed, err := fn()
		r := runResult{
			Run:       i,
			ElapsedMS: elapsed.Milliseconds(),
		}
		if err == nil {
			r.Verdict = "pass"
		} else {
			r.Verdict = "fail"
			r.FailureReason = err.Error()
			var detail *pulse.PulseFailure
			if errors.As(err, &detail) {
				r.MeasuredValue = detail.MeasuredValue
				r.ThresholdValue = detail.ThresholdValue
				r.Unit = detail.Unit
			}
		}
		results = append(results, r)
	}
	return results
}

// summarize aggregates run results into a top-level verdict.
func summarize(runs []runResult) reportSummary {
	s := reportSummary{Total: len(runs)}
	for _, r := range runs {
		if r.Verdict == "pass" {
			s.Passed++
		} else {
			s.Failed++
		}
		if r.ElapsedMS > s.WorstElapsedMS {
			s.WorstElapsedMS = r.ElapsedMS
		}
	}
	if s.Failed > 0 {
		s.Verdict = "STRAGGLER"
	} else {
		s.Verdict = "HEALTHY"
	}
	return s
}
