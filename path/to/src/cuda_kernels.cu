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
    int