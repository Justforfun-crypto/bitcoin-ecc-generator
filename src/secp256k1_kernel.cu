#include <cuda_runtime.h>
#include <stdint.h>

typedef uint64_t u64;

struct PointJacobian {
    u64 X[4];
    u64 Y[4];
    u64 Z[4];
};

__device__ __forceinline__ void field_sub(u64 *r, const u64 *a, const u64 *b) {
    u64 borrow = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i]) || (borrow && a[i] == b[i]) ? 1 : 0;
        r[i] = diff;
    }
}

__device__ __forceinline__ void mont_mul_dev(u64 *r, const u64 *a, const u64 *b) {
    const u64 P[4] = {
        0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    const u64 INV = 0xD838091DD2253531ULL;

    u64 t[5] = {0, 0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            u64 prod_lo = a[i] * b[j];
            u64 prod_hi = __umul64hi(a[i], b[j]);
            
            u64 sum = t[j] + prod_lo;
            u64 c1 = (sum < t[j]) ? 1 : 0;
            
            sum += carry;
            u64 c2 = (sum < carry) ? 1 : 0;
            
            t[j] = sum;
            carry = prod_hi + c1 + c2;
        }
        t[4] += carry;

        u64 m = t[0] * INV;
        carry = 0;
        
        u64 m_p0_lo = m * P[0];
        u64 m_p0_hi = __umul64hi(m, P[0]);
        u64 sum0 = t[0] + m_p0_lo;
        carry = m_p0_hi + ((sum0 < t[0]) ? 1 : 0);

        #pragma unroll
        for (int j = 1; j < 4; j++) {
            u64 prod_lo = m * P[j];
            u64 prod_hi = __umul64hi(m, P[j]);
            
            u64 sum = t[j] + prod_lo;
            u64 c1 = (sum < t[j]) ? 1 : 0;
            
            sum += carry;
            u64 c2 = (sum < carry) ? 1 : 0;
            
            t[j-1] = sum;
            carry = prod_hi + c1 + c2;
        }
        t[3] = t[4] + carry;
        t[4] = 0;
    }

    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
}

__device__ void point_add_jacobian(PointJacobian *r, const PointJacobian *p1, const PointJacobian *p2) {
    u64 z1_sq[4], z2_sq[4];
    u64 u1[4], u2[4], s1[4], s2[4];
    u64 h[4], r_val[4], h_sq[4], h_cube[4];
    u64 u1_h_sq[4], term[4];

    mont_mul_dev(z1_sq, p1->Z, p1->Z);
    mont_mul_dev(z2_sq, p2->Z, p2->Z);

    mont_mul_dev(u1, p1->X, z2_sq);
    mont_mul_dev(u2, p2->X, z1_sq);

    mont_mul_dev(s1, p1->Y, z2_sq);
    mont_mul_dev(s1, s1, p2->Z);

    mont_mul_dev(s2, p2->Y, z1_sq);
    mont_mul_dev(s2, s2, p1->Z);

    field_sub(h, u2, u1);
    field_sub(r_val, s2, s1);

    mont_mul_dev(h_sq, h, h);
    mont_mul_dev(h_cube, h_sq, h);

    mont_mul_dev(u1_h_sq, u1, h_sq);

    mont_mul_dev(r->X, r_val, r_val);
    field_sub(r->X, r->X, h_cube);
    field_sub(r->X, r->X, u1_h_sq);
    field_sub(r->X, r->X, u1_h_sq);

    field_sub(term, u1_h_sq, r->X);
    mont_mul_dev(r->Y, r_val, term);
    mont_mul_dev(term, s1, h_cube);
    field_sub(r->Y, r->Y, term);

    mont_mul_dev(r->Z, h, p1->Z);
    mont_mul_dev(r->Z, r->Z, p2->Z);
}

extern "C" __global__ void secp256k1_montgomery_mul_kernel(const u64 *a, const u64 *b, u64 *c, u64 n) {
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        mont_mul_dev(&c[idx * 4], &a[idx * 4], &b[idx * 4]);
    }
}

extern "C" __global__ void secp256k1_point_add_kernel(PointJacobian *out, const PointJacobian *p1, const PointJacobian *p2, u64 n) {
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        point_add_jacobian(&out[idx], &p1[idx], &p2[idx]);
    }
}
