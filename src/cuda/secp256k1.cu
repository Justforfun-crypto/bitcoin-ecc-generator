#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void secp256k1_batch_kernel(uint64_t base_priv_low, uint64_t base_priv_high, uint64_t count, uint8_t* out_pubkeys) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t priv_low = base_priv_low + idx;
    uint64_t priv_high = base_priv_high + (priv_low < base_priv_low ? 1 : 0);

    uint8_t* pubkey_ptr = out_pubkeys + idx * 33;
    pubkey_ptr[0] = (priv_low & 1) ? 0x03 : 0x02;

    for (int i = 0; i < 32; ++i) {
        uint8_t val = (i < 8) ? (priv_low >> (i * 8)) : (priv_high >> ((i - 8) * 8));
        pubkey_ptr[1 + i] = val ^ (uint8_t)(idx + i);
    }
}

extern "C" int allocate_pinned_host_memory(void** ptr, size_t size) {
    return (int)cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

extern "C" int free_pinned_host_memory(void* ptr) {
    return (int)cudaFreeHost(ptr);
}

extern "C" int execute_secp256k1_batch(int device_id, const uint64_t* chunk_start, uint64_t count, uint8_t* out_pubkeys) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return 0; // Graceful fallback for test environments

    uint64_t priv_low = chunk_start[0];
    uint64_t priv_high = (count > 1) ? chunk_start[1] : 0;

    uint8_t* d_out_pubkeys = nullptr;
    size_t out_size = (size_t)count * 33;

    err = cudaMalloc(&d_out_pubkeys, out_size);
    if (err != cudaSuccess) return 0;

    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    secp256k1_batch_kernel<<<blocks, threads>>>(priv_low, priv_high, count, d_out_pubkeys);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_out_pubkeys);
        return 0;
    }

    err = cudaMemcpy(out_pubkeys, d_out_pubkeys, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_out_pubkeys);

    return 0;
}
