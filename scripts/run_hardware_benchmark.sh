#!/usr/bin/env bash
# run_hardware_benchmark.sh
#
# Builds and runs the real CUDA GPU pulse benchmark on a bare-metal GPU
# instance (RunPod, Lambda Labs, etc.) and writes the structured JSON
# evidence report to evidence.json in the repo root.
#
# Assumptions:
#   - nvcc (CUDA toolkit) is present
#   - Ubuntu / Debian host
#   - sudo available (for Go install and placing libgpupulse.so)
#
# Usage:
#   bash scripts/run_hardware_benchmark.sh [run-count]
#   bash scripts/run_hardware_benchmark.sh 10   # override default of 5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO_VERSION="1.23.4"
EVIDENCE_FILE="${REPO_ROOT}/evidence.json"
RUN_COUNT="${1:-5}"

# ── Helpers ────────────────────────────────────────────────────────────────────
info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[0;33m[WARN]\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;34m── %s\033[0m\n' "$*"; }

# ── Preflight ──────────────────────────────────────────────────────────────────
check_nvcc() {
    step "Checking CUDA toolchain"
    command -v nvcc &>/dev/null \
        || die "nvcc not found — install the CUDA toolkit before running this script."
    info "nvcc: $(nvcc --version | grep 'release' | sed 's/.*release //' | cut -d, -f1)"
}

show_gpus() {
    step "Detecting GPUs"
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found — GPU detection will report 'unknown' in evidence"
        return
    fi
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null \
        | while IFS= read -r line; do info "  GPU ${line}"; done || true
}

# ── Go ─────────────────────────────────────────────────────────────────────────
ensure_go() {
    step "Checking Go"
    if command -v go &>/dev/null; then
        info "Go already installed: $(go version)"
        export PATH="$(go env GOROOT)/bin:${PATH}"
        return
    fi
    info "Installing Go ${GO_VERSION}..."
    local tmp
    tmp="$(mktemp -d)"
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
        -o "${tmp}/go.tar.gz"
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "${tmp}/go.tar.gz"
    rm -rf "${tmp}"
    export PATH="/usr/local/go/bin:${PATH}"
    info "Installed: $(go version)"
}

# ── Build ──────────────────────────────────────────────────────────────────────
build_so() {
    step "Building libgpupulse.so (nvcc)"
    cd "${REPO_ROOT}"
    make cuda
    local size
    size="$(du -sh cuda/libgpupulse.so | cut -f1)"
    info "cuda/libgpupulse.so  (${size})"

    # The binary's rpath is /usr/local/lib (set in the #cgo LDFLAGS).
    # Place the .so there so it is found at runtime without LD_LIBRARY_PATH.
    sudo cp cuda/libgpupulse.so /usr/local/lib/libgpupulse.so
    sudo ldconfig
    info "Installed to /usr/local/lib/libgpupulse.so"
}

build_benchmark() {
    step "Compiling benchmark binary (-tags cuda)"
    cd "${REPO_ROOT}"
    CGO_CFLAGS="-I${REPO_ROOT}/cuda" \
    CGO_LDFLAGS="-L${REPO_ROOT}/cuda -lgpupulse -lcudart -lstdc++ -Wl,-rpath,/usr/local/lib" \
    go build \
        -tags cuda \
        -ldflags="-s -w" \
        -o "${REPO_ROOT}/benchmark" \
        ./cmd/benchmark
    info "benchmark binary ready"
}

# ── Run ────────────────────────────────────────────────────────────────────────
run_benchmark() {
    step "Running GPU pulse benchmark  (${RUN_COUNT} real hardware runs)"
    info "Each run executes ${RUN_COUNT} GEMM passes + P2P ring check + clock validation"
    cd "${REPO_ROOT}"
    ./benchmark --scenario=real --count="${RUN_COUNT}" | tee "${EVIDENCE_FILE}"
    echo ""
}

print_summary() {
    step "Evidence"
    info "Report: ${EVIDENCE_FILE}"
    if command -v jq &>/dev/null; then
        local gpu thresh verdict passed failed worst
        gpu="$(     jq -r '.gpu_arch'                  "${EVIDENCE_FILE}")"
        thresh="$(  jq -r '.calibrated_threshold_ms'   "${EVIDENCE_FILE}")"
        verdict="$( jq -r '.summary.verdict'           "${EVIDENCE_FILE}")"
        passed="$(  jq -r '.summary.passed'            "${EVIDENCE_FILE}")"
        failed="$(  jq -r '.summary.failed'            "${EVIDENCE_FILE}")"
        worst="$(   jq -r '.summary.worst_elapsed_ms'  "${EVIDENCE_FILE}")"
        info "GPU:               ${gpu}"
        info "Threshold:         ${thresh}ms (auto-calibrated)"
        info "Passed / Failed:   ${passed} / ${failed}"
        info "Worst elapsed:     ${worst}ms"
        info "Verdict:           ${verdict}"
    fi
    echo ""
    info "Full structured evidence:"
    info "  cat ${EVIDENCE_FILE}"
    if command -v jq &>/dev/null; then
        info "  jq '.runs[] | select(.verdict == \"fail\")' ${EVIDENCE_FILE}"
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
    info "straggler-shield hardware benchmark"
    info "Repo: ${REPO_ROOT} | Runs: ${RUN_COUNT}"

    check_nvcc
    show_gpus
    ensure_go
    build_so
    build_benchmark
    run_benchmark
    print_summary
}

main
