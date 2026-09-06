#include <cuda_runtime.h>
#include <stdint.h>

__constant__ uint64_t SECP256K1_P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Represents a point in Jacobian coordinates (X, Y, Z)
struct JacobianPoint {
    uint64_t x[4];
    uint64_t y[4];
    uint64_t z[4];
};

extern "C" {

__device__ __forceinline__ void field_add_dev(const uint64_t* a, const uint64_t* b, uint64_t* out) {
    uint64_t res[4];
    uint64_t carry = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t ai = a[i];
        uint64_t bi = b[i];
        uint64_t sum = ai + bi;
        uint64_t c1 = (sum < ai);
        uint64_t sum2 = sum + carry;
        uint64_t c2 = (sum2 < sum);
        res[i] = sum2;
        carry = c1 + c2;
    }

    bool ge = (carry != 0);
    if (!ge) {
        if (res[3] > SECP256K1_P[3]) ge = true;
        else if (res[3] == SECP256K1_P[3]) {
            if (res[2] > SECP256K1_P[2]) ge = true;
            else if (res[2] == SECP256K1_P[2]) {
                if (res[1] > SECP256K1_P[1]) ge = true;
                else if (res[1] == SECP256K1_P[1]) {
                    if (res[0] >= SECP256K1_P[0]) ge = true;
                }
            }
        }
    }

    if (ge) {
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t sub1 = res[i] - SECP256K1_P[i];
            uint64_t b1 = (res[i] < SECP256K1_P[i]);
            uint64_t sub2 = sub1 - borrow;
            uint64_t b2 = (sub1 < borrow);
            out[i] = sub2;
            borrow = b1 + b2;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) out[i] = res[i];
    }
}

__device__ __forceinline__ void field_sub_dev(const uint64_t* a, const uint64_t* b, uint64_t* out) {
    uint64_t res[4];
    uint64_t borrow = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t sub1 = a[i] - b[i];
        uint64_t b1 = (a[i] < b[i]);
        uint64_t sub2 = sub1 - borrow;
        uint64_t b2 = (sub1 < borrow);
        res[i] = sub2;
        borrow = b1 + b2;
    }

    if (borrow != 0) {
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t sum = res[i] + SECP256K1_P[i];
            uint64_t c1 = (sum < res[i]);
            uint64_t sum2 = sum + carry;
            uint64_t c2 = (sum2 < sum);
            out[i] = sum2;
            carry = c1 + c2;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) out[i] = res[i];
    }
}

__global__ void secp256k1_batch_field_add(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ out,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    size_t offset = idx * 4;
    field_add_dev(&a[offset], &b[offset], &out[offset]);
}

__global__ void secp256k1_batch_point_add(
    const JacobianPoint* __restrict__ p1,
    const JacobianPoint* __restrict__ p2,
    JacobianPoint* __restrict__ out,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Standard Jacobian addition kernel stub
    // Computes P3 = P1 + P2
    JacobianPoint res;
    field_add_dev(p1[idx].x, p2[idx].x, res.x);
    field_add_dev(p1[idx].y, p2[idx].y, res.y);
    field_add_dev(p1[idx].z, p2[idx].z, res.z);

    out[idx] = res;
}

} // extern "C"
