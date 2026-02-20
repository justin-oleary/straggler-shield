#ifndef GPU_PULSE_H
#define GPU_PULSE_H

#ifdef __cplusplus
extern "C" {
#endif

// Return codes — 0 on success, positive on failure so CGO can test rc > 0.
#define GPU_PULSE_OK            0
#define GPU_PULSE_ERR_CUDA      1
#define GPU_PULSE_ERR_OOM       2
#define GPU_PULSE_ERR_P2P       3   // peer access unsupported or severely degraded

// gpu_device_count returns the number of CUDA-visible GPU devices, or -1 on error.
int gpu_device_count(void);

// run_gpu_pulse launches a 2048×2048 tiled GEMM on the specified device.
// One warm-up pass fires first to force P0 and JIT-compile PTX; the timed
// pass follows. Blocks on cudaDeviceSynchronize before returning.
//
// device_id: 0-based GPU index (must be < gpu_device_count())
// returns:   GPU_PULSE_OK (0) on success, GPU_PULSE_ERR_* (>0) on failure
int run_gpu_pulse(int device_id);

// run_p2p_check times a 100 MiB cudaMemcpyPeer transfer from src_device to
// dst_device after a warm-up pass. Requires NVLink or PCIe peer access.
//
// bandwidth_gbs: output — measured unidirectional bandwidth in GB/s
// returns: GPU_PULSE_OK, GPU_PULSE_ERR_P2P if peer access is unavailable,
//          or GPU_PULSE_ERR_OOM if device allocation fails
int run_p2p_check(int src_device, int dst_device, double *bandwidth_gbs);

#ifdef __cplusplus
}
#endif

#endif // GPU_PULSE_H
