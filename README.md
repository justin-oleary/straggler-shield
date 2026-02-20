### ⚖️ Intellectual Property Notice
**Patent Pending:** This system and its underlying methods for GPU cluster validation, variance-based "fail-slow" detection, and ring-topology interconnect profiling are the subject of a pending U.S. Patent Application (Application No. 63/987,090).

All rights reserved. Unauthorized commercial use, reproduction, or distribution of these methods is strictly prohibited under 35 U.S.C. § 111.

---

# straggler-shield

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
3. **P2P check** — 100 MiB `cudaMemcpyPeer` from GPU 0 to each peer (NVLink and PCIe). Threshold: 5 GB/s.
4. **Clock validation** — queries `nvidia-smi` again post-pulse. SM clock must be ≥ 50% of max, confirming the device reached P0 under load.

Thresholds:

| Check | Threshold |
|---|---|
| Mean GEMM latency | 500ms (20× headroom over A100 nominal ~25ms) |
| Coefficient of variation | 20% (Falcon paper CV signal for fail-slow) |
| P2P bandwidth | 5 GB/s minimum |
| Idle temperature | 70°C maximum |
| Post-pulse SM clock | ≥ 50% of device max |

## Architecture

```
K8s node watch loop (watch.Modified/Added)
  └─ edge-detect Ready transition (within 5-minute window)
       └─ ReconcileNode()
            ├─ preflight()          nvidia-smi ECC + temp
            ├─ runDevicePulse() × N  per-device GEMM timing
            ├─ checkP2P() × peers    cudaMemcpyPeer bandwidth
            └─ validateClocks()      post-pulse SM clock check
                 ├─ pass → removeTaint (clear zombie-quarantine)
                 └─ fail → applyTaint + GPUStraggler condition
```

Metrics are served on `:9090/metrics`. The agent runs as a DaemonSet with one replica per GPU node; it watches only its own node via the downward API `NODE_NAME` env var.

## Building

Requires CUDA toolkit and `nvcc` on the build host.

```
# build the shared library and agent binary
make

# CI / no-GPU builds (stub pulse, always passes)
make go-stub

# adjust NVCC_FLAGS in the Makefile for your GPU architecture:
#   sm_70  Volta   (V100)
#   sm_80  Ampere  (A100) — default
#   sm_86  Ampere  (RTX 3090)
#   sm_90  Hopper  (H100)
```

The CUDA kernel compiles to `cuda/libgpupulse.so`. The Go binary links against it via CGO (`-tags cuda`).

## Deploying

The agent needs:

- `NODE_NAME` set via the downward API
- Hostpath or device plugin access to the GPU(s) (`nvidia.com/gpu` resource)
- RBAC: `get`, `watch`, `patch` on `nodes` and `nodes/status`
- The `libgpupulse.so` available at runtime — either baked into the image or mounted

Metrics port 9090 should be scraped by your Prometheus instance. Example ServiceMonitor label selector left to the cluster operator.

## Quarantine taint

```
sunk.coreweave.com/zombie-quarantine:NoSchedule
```

A `GPUStraggler` node condition is also written to the status subresource with the failure reason and measured latency. Both are cleared atomically when a node subsequently passes the pulse.

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
