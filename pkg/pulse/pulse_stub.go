//go:build !cuda

package pulse

import (
	"errors"
	"time"
)

// RunPulse is a stub used when building without the cuda tag.
// Compile with -tags cuda on a GPU host to get the real implementation.
func RunPulse() (time.Duration, error) {
	return 0, errors.New("built without cuda support: recompile with -tags cuda")
}
