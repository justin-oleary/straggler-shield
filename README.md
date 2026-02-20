### ⚖️ Intellectual Property Notice
**Patent Pending:** This system and its underlying methods for GPU cluster validation, variance-based "fail-slow" detection, and ring-topology interconnect profiling are the subject of a pending U.S. Patent Application (Application No. 63/987,090).

All rights reserved. Unauthorized commercial use, reproduction, or distribution of these methods is strictly prohibited under 35 U.S.C. § 111.

---

# straggler-shield

![Patent Pending](https://img.shields.io/badge/patent%20pending-63%2F987%2C090-CFB53B?style=flat-square)
![Go](https://img.shields.io/badge/go-1.23-00ADD8?style=flat-square&logo=go&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Kubernetes](https://img.shields.io/badge/kubernetes-daemonset-326CE5?style=flat-square&logo=kubernetes&logoColor=white)
![License](https://img.shields.io/badge/license-proprietary-red?style=flat-square)

A Kubernetes DaemonSet agent that intercepts GPU nodes before Slurm can resume them. When a node transitions to Ready after a reboot or join event, the agent runs a synthetic GPU workload and quarantines the node if it exhibits straggler behavior — before any job ever lands on it.

The taint applied is `sunk.coreweave.com/zombie-quarantine:NoSchedule`. Nodes that pass are cleared immediately.

## The problem

GPU nodes regularly return to a Ready state without being fit for distributed training. Common failure modes:

- **Fail-slow**: mean GEMM latency looks acceptable but the GPU throttles intermittently, creating high variance across runs. Every AllReduce barrier on a shared job stalls waiting for the slow node.
- **Clock deration**: SM clock remains stuck well below P0 after a thermal event, producing consistent slowdowns that compound across thousands of training steps.
- **NVLink failure**: the GEMM pulse passes but P2P bandwidth between devices is severely degraded, making multi-GPU AllReduce across that node unusable.
- **ECC errors / thermal recovery incomplete**: detected pre-flight, node quarantined without running the pulse at all.

Single-pass latency checks miss all of these. This agent runs five timed passes per device and evaluates mean, coefficient of variation, NVLink bandwidth, and post-pulse clock state before clearing a node.

## What it runs

For each GPU on the node:

1. **Pre-flight** — queries `nvidia-smi` for uncorrectable ECC errors and idle temperature. Any ECC error or temp above 70°C quarantines immediately.
2. **GEMM pulse** — five timed 2048×2048 FP32 matrix multiplications via a CUDA shared library. Computes mean latency and coefficient of variation across runs.
3. **P2P ring check** — 100 MiB `cudaMemcpyPeer` across each adjacent GPU pair in ring order (0→1, 1→2, …, N-1→0). Catches any single broken NVLink segment.
4. **Clock validation** — queries `nvidia-smi` post-pulse. SM clock must be ≥ 50% of device max, confirming the device boosted to P0 under load.

Thresholds are auto-calibrated to the detected GPU architecture:

| Check | H100 / H200 | A100 | B200 / GB200 | Default |
|---|---|---|---|---|
| Mean GEMM latency | 35 ms | 100 ms | 15 ms | 500 ms |
| Coefficient of variation | 20% | 20% | 20% | 20% |
| P2P bandwidth | 5 GB/s | 5 GB/s | 5 GB/s | 5 GB/s |
| Idle temperature | 70°C | 70°C | 70°C | 70°C |
| Post-pulse SM clock | ≥ 50% max | ≥ 50% max | ≥ 50% max | ≥ 50% max |

All thresholds are overridable via environment variables (`PULSE_THRESHOLD_MS`, `PULSE_CV_MAX`, `P2P_MIN_GBS`, `IDLE_TEMP_MAX`).

## Architecture

```
K8s node watch loop (watch.Modified/Added)
  └─ edge-detect Ready transition (within 5-minute window)
       └─ ReconcileNode()
            ├─ preflight()            nvidia-smi ECC + temp
            ├─ runDevicePulse() × N   per-device GEMM timing
            ├─ checkP2P() × ring      cudaMemcpyPeer ring topology
            └─ validateClocks()       post-pulse SM clock check
                 ├─ pass → removeTaint (clear zombie-quarantine)
                 └─ fail → applyTaint + GPUStraggler condition
```

Metrics are served on `:9090/metrics`. The agent runs as a DaemonSet with one replica per GPU node; it watches only its own node via the downward API `NODE_NAME` env var. Per-node reconciliation is guarded by a `sync.Mutex` — duplicate Ready events are discarded while a pulse is in flight.

## Building

Requires CUDA toolkit and `nvcc` on the build host.

```bash
# build the shared library and agent binary
make

# CI / no-GPU builds (stub pulse, always passes)
make go-stub

# GPU architecture targets — set in Makefile NVCC_FLAGS:
#   sm_70  Volta   (V100)
#   sm_80  Ampere  (A100)
#   sm_90  Hopper  (H100) — default
#   sm_100 Blackwell (B200)
```

The CUDA kernel compiles to `cuda/libgpupulse.so`. The Go binary links against it via CGO (`-tags cuda`).

## Deploying

```bash
kubectl apply -f deploy/rbac.yaml
kubectl apply -f deploy/daemonset.yaml
```

The agent needs:

- `NODE_NAME` set via the downward API
- GPU device plugin (`nvidia.com/gpu` resource) and `runtimeClassName: nvidia`
- RBAC: `get`, `watch`, `patch` on `nodes` and `nodes/status`

## Benchmarking on real hardware

A self-contained script is included for generating structured evidence on a bare-metal GPU instance (RunPod, Lambda Labs, etc.):

```bash
git clone https://github.com/justin-oleary/straggler-shield
cd straggler-shield
bash scripts/run_hardware_benchmark.sh 10   # 10 runs, writes evidence.json
```

Requires `nvcc`, `sudo`, and a Debian/Ubuntu host. Installs Go if absent.

## Quarantine taint

```
sunk.coreweave.com/zombie-quarantine:NoSchedule
```

A `GPUStraggler` node condition is also written to the status subresource with the failure reason and measured values. Both are cleared atomically when a node subsequently passes the pulse.

## Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `gpu_validator_pulse_duration_seconds` | Histogram | `device` | Mean GEMM latency per device per validation cycle |
| `gpu_validator_pulse_cv` | Gauge | `device` | Coefficient of variation across GEMM runs |
| `gpu_validator_straggler_detected_total` | Counter | `reason` | Quarantine events by failure reason |

Reason values: `latency_threshold_exceeded`, `high_variance`, `interconnect_degraded`, `pre_flight_failure`.

## References

- Falcon: [arXiv:2410.12588](https://arxiv.org/abs/2410.12588) — fail-slow GPU detection via iteration-time variance
- CoreWeave SUNK — zombie node quarantine taint convention
- Meta infrastructure post-mortem on H100 HBM3 ECC failures driving job stalls
