#include "gpu_pulse.h"

#include <cuda_runtime.h>
#include <stdlib.h>

#define N    2048
#define TILE 16

// Tiled GEMM — exercises shared memory, L2 cache, and FP32 throughput.
// Avoids cuBLAS so the result reflects raw device capability.
__global__ void matmul(const float *__restrict__ A,
                       const float *__restrict__ B,
                       float *__restrict__ C)
{
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        tA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        tB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            acc += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

extern "C" int gpu_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
        return -1;
    return n;
}

extern "C" int run_gpu_pulse(int device_id)
{
    if (cudaSetDevice(device_id) != cudaSuccess)
        return GPU_PULSE_ERR_CUDA;

    const size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    if (!h_A || !h_B) {
        free(h_A);
        free(h_B);
        return GPU_PULSE_ERR_OOM;
    }

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % 97) * 0.01f;
        h_B[i] = (float)((i * 13) % 97) * 0.01f;
    }

    float *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, bytes) != cudaSuccess) goto oom;
    if (cudaMalloc(&d_B, bytes) != cudaSuccess) { cudaFree(d_A); goto oom; }
    if (cudaMalloc(&d_C, bytes) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); goto oom; }

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    {
        dim3 block(TILE, TILE);
        dim3 grid(N / TILE, N / TILE);

        // warm-up — forces P0 and JIT-compiles PTX
        matmul<<<grid, block>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();

        // measured pass — Go wall-clock times the full C call
        matmul<<<grid, block>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    return GPU_PULSE_OK;

oom:
    free(h_A);
    free(h_B);
    return GPU_PULSE_ERR_CUDA;
}

// run_p2p_check measures unidirectional NVLink/PCIe bandwidth from src to dst.
// The Go layer calls this in ring order (0→1, 1→2, …, N-1→0) so any single
// broken link in the HGX fabric is caught, not just links that involve GPU 0.
// Uses cudaMemcpyPeer after enabling peer access. One warm-up pass before
// the timed copy to prime TLB and NVSwitch routing tables.
extern "C" int run_p2p_check(int src_device, int dst_device, double *bandwidth_gbs)
{
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, src_device, dst_device);
    if (!can_access)
        return GPU_PULSE_ERR_P2P;

    cudaSetDevice(src_device);
    cudaError_t err = cudaDeviceEnablePeerAccess(dst_device, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
        return GPU_PULSE_ERR_P2P;

    // 100 MiB — large enough to saturate the interconnect and amortise
    // launch overhead; small enough to complete in < 10ms on healthy NVLink.
    const size_t transfer_size = 100ULL * 1024 * 1024;

    void *src_buf = NULL, *dst_buf = NULL;

    cudaSetDevice(src_device);
    if (cudaMalloc(&src_buf, transfer_size) != cudaSuccess)
        return GPU_PULSE_ERR_OOM;

    cudaSetDevice(dst_device);
    if (cudaMalloc(&dst_buf, transfer_size) != cudaSuccess) {
        cudaSetDevice(src_device);
        cudaFree(src_buf);
        return GPU_PULSE_ERR_OOM;
    }

    // All timing runs from the src device context.
    cudaSetDevice(src_device);

    // warm-up — primes NVSwitch routing and TLBs
    cudaMemcpyPeer(dst_buf, dst_device, src_buf, src_device, transfer_size);
    cudaDeviceSynchronize();

    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);

    cudaEventRecord(t_start);
    cudaMemcpyPeer(dst_buf, dst_device, src_buf, src_device, transfer_size);
    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, t_start, t_stop);
    *bandwidth_gbs = ((double)transfer_size / (elapsed_ms * 1e-3)) / 1e9;

    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);
    cudaSetDevice(src_device);
    cudaFree(src_buf);
    cudaSetDevice(dst_device);
    cudaFree(dst_buf);

    return GPU_PULSE_OK;
}
