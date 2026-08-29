__global__ void kernel_scalar_mult(
    const uint32_t *scalars,
    const uint32_t *g_x,
    const uint32_t *g_y,
    uint32_t *result_x,
    uint32_t *result_y,
    int num_points,
    uint32_t p[8],  // Prime field
    uint32_t n[8]   // Curve order
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    uint32_t scalar = scalars[idx];
    
    // Load generator point into registers (optimization)
    uint32_t gx[8], gy[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        gx[i] = g_x[i];
        gy[i] = g_y[i];
    }

    // Binary scalar multiplication (double-and-add)
    uint32_t rx[8] = {0}, ry[8] = {0};
    bool is_infinity = true;
    
    for (int bit = 31; bit >= 0; bit--) {
        // Point doubling
        if (!is_infinity) {
            point_double_jacobian(rx, ry, rx, ry, p);
        }
        
        // Conditional point addition
        if ((scalar >> bit) & 1) {
            if (is_infinity) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    rx[i] = gx[i];
                    ry[i] = gy[i];
                }
                is_infinity = false;
            } else {
                point_add_jacobian(rx, ry, gx, gy, rx, ry, p);
            }
        }
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result_x[idx * 8 + i] = rx[i];
        result_y[idx * 8 + i] = ry[i];
    }
}

__global__ void kernel_kangaroo(
    uint64_t range_start,
    uint64_t range_size,
    const uint32_t *target_x,
    const uint32_t *target_y,
    uint64_t *collision_table,
    int *collision_found,
    uint64_t *collision_key
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    uint64_t keys_per_thread = range_size / total_threads;
    uint64_t start = range_start + thread_id * keys_per_thread;
    uint64_t end = start + keys_per_thread;

    for (uint64_t k = start; k < end; k++) {
        // Compute k*G
        uint32_t px[8], py[8];
        scalar_mult_constant(k, px, py);
        
        // Hash point coordinates for collision detection
        uint64_t hash = compute_hash(px, py);
        uint32_t bucket = hash % (1024 * 1024 * 100);
        
        // Check for collision
        if (collision_table[bucket] != 0) {
            *collision_found = 1;
            *collision_key = k;
            return;
        }
        
        // Store in table
        collision_table[bucket] = k;
    }
}

__global__ void kernel_batch_point_add(
    const uint32_t *points_x,
    const uint32_t *points_y,
    const uint32_t *points_z,
    uint32_t *result_x,
    uint32_t *result_y,
    uint32_t *result_z,
    int num_points,
    uint32_t p[8]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points - 1) return;

    // Load points from global memory
    uint32_t x1[8], y1[8], z1[8];
    uint32_t x2[8], y2[8], z2[8];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x1[i] = points_x[idx * 8 + i];
        y1[i] = points_y[idx * 8 + i];
        z1[i] = points_z[idx * 8 + i];
        x2[i] = points_x[(idx + 1) * 8 + i];
        y2[i] = points_y[(idx + 1) * 8 + i];
        z2[i] = points_z[(idx + 1) * 8 + i];
    }

    // Add in Jacobian coordinates (no inversions!)
    uint32_t rx[8], ry[8], rz[8];
    point_add_jacobian(x1, y1, z1, x2, y2, z2, rx, ry, rz, p);

    // Store results
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result_x[idx * 8 + i] = rx[i];
        result_y[idx * 8 + i] = ry[i];
        result_z[idx * 8 + i] = rz[i];
    }
}

// Helper functions
__device__ void point_double_jacobian(
    uint32_t rx[8], uint32_t ry[8],
    const uint32_t x[8], const uint32_t y[8],
    const uint32_t p[8]
) {
    // Optimized point doubling in Jacobian coordinates
    uint32_t xx[8], yy[8], zz[8], s[8], m[8];
    
    mod_mul_256(xx, x, x, p);
    mod_mul_256(yy, y, y, p);
    mod_mul_256(zz, xx, xx, p);
    
    mod_mul_256(s, xx, yy, p);
    mod_mul_256(m, xx, xx, p);
    
    mod_mul_256(rx, m, m, p);
    mod_sub_256(rx, rx, s, p);
    
    mod_sub_256(ry, s, rx, p);
    mod_mul_256(ry, ry, m, p);
    mod_sub_256(ry, ry, yy, p);
}

__device__ void point_add_jacobian(
    uint32_t rx[8], uint32_t ry[8], uint32_t rz[8],
    const uint32_t x1[8], const uint32_t y1[8], const uint32_t z1[8],
    const uint32_t x2[8], const uint32_t y2[8], const uint32_t z2[8],
    const uint32_t p[8]
) {
    // Mixed Jacobian-affine addition
    uint32_t u1[8], u2[8], s1[8], s2[8], h[8], r[8];
    
    mod_mul_256(u1, x1, z2, p);
    mod_mul_256(u2, x2, z1, p);
    mod_mul_256(s1, y1, z2, p);
    mod_mul_256(s2, y2, z1, p);
    
    mod_sub_256(h, u2, u1, p);
    mod_sub_256(r, s2, s1, p);
    
    // Complete the addition...
}

__device__ void mod_mul_256(
    uint32_t result[8],
    const uint32_t a[8],
    const uint32_t b[8],
    const uint32_t p[8]
) {
    // 256-bit modular multiplication
    uint64_t product[16] = {0};
    
    // Multiply
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a[i] * b[j];
            product[i + j] += prod & 0xFFFFFFFFULL;
            product[i + j + 1] += prod >> 32;
        }
    }
    
    // Reduce modulo p
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result[i] = (uint32_t)(product[i] & 0xFFFFFFFFULL);
    }
}

__device__ void mod_sub_256(
    uint32_t result[8],
    const uint32_t a[8],
    const uint32_t b[8],
    const uint32_t p[8]
) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result[i] = a[i] - b[i];
        if (a[i] < b[i]) {
            result[i] -= 1;
        }
    }
}

__device__ uint64_t compute_hash(
    const uint32_t x[8],
    const uint32_t y[8]
) {
    uint64_t hash = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash ^= ((uint64_t)x[i] << 32) | y[i];
        hash = (hash << 7) | (hash >> 57);
    }
    return hash;
}
