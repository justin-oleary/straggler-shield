package pulse

import "errors"

var (
	// ErrStragglerDetected is returned when mean GEMM latency across all runs
	// on any device exceeds the threshold, or when the post-pulse clock check
	// confirms a power-derated state under load.
	ErrStragglerDetected = errors.New("straggler detected: GPU pulse latency exceeded threshold")

	// ErrHighVariance is returned when mean latency is acceptable but the
	// coefficient of variation across runs exceeds the ceiling on any device.
	// Per the Falcon paper (arXiv:2410.12588), high CV is the primary signature
	// of fail-slow GPUs — they fail erratically, dragging AllReduce barriers.
	ErrHighVariance = errors.New("straggler detected: high run-to-run variance (fail-slow pattern)")

	// ErrInterconnectDegraded is returned when P2P bandwidth on any ring segment
	// falls below the minimum threshold, or when cudaDeviceCanAccessPeer reports
	// the link as unavailable. An NVLink failure that allows GEMM to pass but
	// causes AllReduce to stall is the canonical SUNK straggler scenario.
	ErrInterconnectDegraded = errors.New("straggler detected: NVLink/P2P bandwidth below threshold")
)

// IsStragglerErr reports whether err indicates the node should be quarantined.
// Covers all three failure modes so callers use a single predicate.
func IsStragglerErr(err error) bool {
	return errors.Is(err, ErrStragglerDetected) ||
		errors.Is(err, ErrHighVariance) ||
		errors.Is(err, ErrInterconnectDegraded)
}

// PulseFailure wraps a sentinel error with the measured value and threshold
// that triggered quarantine. Controllers use errors.As to extract these for
// structured MFU evidence logging. errors.Is still traverses Unwrap, so all
// existing predicate checks (IsStragglerErr, errors.Is) continue to work.
type PulseFailure struct {
	Cause          error
	MeasuredValue  float64 // CV ratio, bandwidth GB/s, or latency ms
	ThresholdValue float64
	Unit           string // "ms", "cv", "gbs"
}

func (f *PulseFailure) Error() string { return f.Cause.Error() }
func (f *PulseFailure) Unwrap() error { return f.Cause }
