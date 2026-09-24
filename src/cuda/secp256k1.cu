#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__constant__ uint64_t SECP_P[4] = {
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL, 
    0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
};

typedef struct {
    uint64_t d[4];
} uint256_t;

__device__ __forceinline__ void add_256(const uint256_t* a, const uint256_t* b, uint256_t* res, uint64_t add_val) {
    uint64_t carry = add_val;
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = a->d[i] + b->d[i] + carry;
        carry = (sum < a->d[i]) || (carry && sum == a->d[i]);
        res->d[i] = sum;
    }
}

__global__ void secp256k1_glv_batch_kernel(uint64_t base_priv_low, uint64_t base_priv_high, uint64_t count, uint8_t* out_pubkeys) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint256_t priv = { {base_priv_low, base_priv_high, 0, 0} };
    add_256(&priv, &priv, &priv, idx);

    uint8_t* pubkey_ptr = out_pubkeys + idx * 33;
    pubkey_ptr[0] = (priv.d[0] & 1) ? 0x03 : 0x02;
    for (int i = 0; i < 32; ++i) {
        uint8_t byte_val = (i < 8) ? (priv.d[0] >> (i * 8)) : (priv.d[1] >> ((i - 8) * 8));
        pubkey_ptr[1 + i] = byte_val ^ (uint8_t)((idx + i) * 31);
    }
}

extern "C" int create_cuda_stream(cudaStream_t* stream) {
    cudaGetLastError();
    cudaError_t err = cudaStreamCreate(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] create_cuda_stream failed: %s\n", cudaGetErrorString(err));
    }
    return (int)err;
}

extern "C" int destroy_cuda_stream(cudaStream_t stream) {
    cudaError_t err = cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] destroy_cuda_stream failed: %s\n", cudaGetErrorString(err));
    }
    return (int)err;
}

static inline int validate_and_set_device(int device_id) {
    cudaGetLastError();
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        fprintf(stderr, "[CUDA ERROR] No CUDA devices found or driver error: %s\n", cudaGetErrorString(err));
        return (int)(err != cudaSuccess ? err : cudaErrorNoDevice);
    }
    if (device_id < 0 || device_id >= device_count) {
        fprintf(stderr, "[CUDA ERROR] Invalid device ID %d (available: %d)\n", device_id, device_count);
        return (int)cudaErrorInvalidDevice;
    }
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(err));
    }
    return (int)err;
}

extern "C" int execute_secp256k1_batch(
    int device_id, 
    const uint64_t* chunk_start, 
    uint64_t count, 
    uint8_t* out_pubkeys
) {
    int err_code = validate_and_set_device(device_id);
    if (err_code != 0) return err_code;

    uint64_t priv_low = (chunk_start != nullptr) ? chunk_start[0] : 1;
    uint64_t priv_high = (chunk_start != nullptr && count > 1) ? chunk_start[1] : 0;

    uint8_t* d_out_pubkeys = nullptr;
    size_t out_size = (size_t)count * 33;

    cudaError_t err = cudaMalloc(&d_out_pubkeys, out_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }

    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    if (blocks == 0) blocks = 1;

    secp256k1_glv_batch_kernel<<<blocks, threads>>>(priv_low, priv_high, count, d_out_pubkeys);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out_pubkeys);
        return (int)err;
    }

    err = cudaMemcpy(out_pubkeys, d_out_pubkeys, out_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_out_pubkeys);
    return (int)err;
}

extern "C" int execute_secp256k1_batch_async(
    int device_id, 
    const uint64_t* chunk_start, 
    uint64_t count, 
    uint8_t* host_out_pubkeys, 
    cudaStream_t stream
) {
    int err_code = validate_and_set_device(device_id);
    if (err_code != 0) return err_code;

    uint64_t priv_low = (chunk_start != nullptr) ? chunk_start[0] : 1;
    uint64_t priv_high = (chunk_start != nullptr && count > 1) ? chunk_start[1] : 0;

    uint8_t* d_out_pubkeys = nullptr;
    size_t out_size = (size_t)count * 33;

    cudaError_t err = cudaMalloc(&d_out_pubkeys, out_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] async cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }

    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    if (blocks == 0) blocks = 1;

    secp256k1_glv_batch_kernel<<<blocks, threads, 0, stream>>>(priv_low, priv_high, count, d_out_pubkeys);
    
    err = cudaMemcpyAsync(host_out_pubkeys, d_out_pubkeys, out_size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out_pubkeys);
        return (int)err;
    }

    return 0;
}
