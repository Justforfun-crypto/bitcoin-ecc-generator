#include <cuda_runtime.h>
#include <cstdio>

extern "C" {
    cudaStream_t create_cuda_stream() {
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
        }
        return stream;
    }

    void destroy_cuda_stream(cudaStream_t stream) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
}
