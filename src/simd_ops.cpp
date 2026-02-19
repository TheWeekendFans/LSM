#include "simd_ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace simd {

bool cpu_supports_avx2() {
#ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds < 7) return false;
    __cpuid(cpuInfo, 7);
    return (cpuInfo[1] & (1 << 5)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(7, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & bit_AVX2) != 0;
#endif
}

float dot_product(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    float res[8];
    _mm256_storeu_ps(res, sum);
    float total = 0.0f;
    for (int k = 0; k < 8; ++k) total += res[k];

    // Remainder
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

void batch_bilinear_interp(const float* src, int srcW, int srcH, int srcStride,
                           const float* xs, const float* ys, float* output, int n) {
    int i = 0;
    // Process 8 pixels at a time
    for (; i + 8 <= n; i += 8) {
        __m256 v_x = _mm256_loadu_ps(xs + i);
        __m256 v_y = _mm256_loadu_ps(ys + i);

        // Floor
        __m256 v_x0_f = _mm256_floor_ps(v_x);
        __m256 v_y0_f = _mm256_floor_ps(v_y);
        __m256i v_x0 = _mm256_cvtps_epi32(v_x0_f);
        __m256i v_y0 = _mm256_cvtps_epi32(v_y0_f);

        // Clamp indices
        __m256i v_w_max = _mm256_set1_epi32(srcW - 1);
        __m256i v_h_max = _mm256_set1_epi32(srcH - 1);
        __m256i v_zero = _mm256_setzero_si256();

        v_x0 = _mm256_max_epi32(v_zero, _mm256_min_epi32(v_x0, v_w_max));
        v_y0 = _mm256_max_epi32(v_zero, _mm256_min_epi32(v_y0, v_h_max));
        __m256i v_x1 = _mm256_max_epi32(v_zero, _mm256_min_epi32(_mm256_add_epi32(v_x0, _mm256_set1_epi32(1)), v_w_max));
        __m256i v_y1 = _mm256_max_epi32(v_zero, _mm256_min_epi32(_mm256_add_epi32(v_y0, _mm256_set1_epi32(1)), v_h_max));

        // Weights
        __m256 v_fx = _mm256_sub_ps(v_x, v_x0_f);
        __m256 v_fy = _mm256_sub_ps(v_y, v_y0_f);
        __m256 v_1_fx = _mm256_sub_ps(_mm256_set1_ps(1.0f), v_fx);
        __m256 v_1_fy = _mm256_sub_ps(_mm256_set1_ps(1.0f), v_fy);

        __m256 w00 = _mm256_mul_ps(v_1_fx, v_1_fy);
        __m256 w10 = _mm256_mul_ps(v_fx, v_1_fy);
        __m256 w01 = _mm256_mul_ps(v_1_fx, v_fy);
        __m256 w11 = _mm256_mul_ps(v_fx, v_fy);

        // Gather involves non-trivial addressing, using scalar fallback for gathering 4 points per pixel is often faster than AVX2 gather if data is not coherent.
        // However, for standard image grids, we can compute offsets.
        // Here we use a simpler approach: extract indices and load values (since scatter/gather can be slow or complex).
        // Optimization: For truly high performance, we'd pre-calculate offsets. Here we just implement the math in SIMD but load scalars to keep code readable/safe.
        
        float vals[8];
        float res[8];
        int x0_arr[8], y0_arr[8], x1_arr[8], y1_arr[8];
        _mm256_storeu_si256((__m256i*)x0_arr, v_x0);
        _mm256_storeu_si256((__m256i*)y0_arr, v_y0);
        _mm256_storeu_si256((__m256i*)x1_arr, v_x1);
        _mm256_storeu_si256((__m256i*)y1_arr, v_y1);
        
        float v00_load[8], v10_load[8], v01_load[8], v11_load[8];

        for(int k=0; k<8; ++k) {
            v00_load[k] = src[y0_arr[k] * srcStride + x0_arr[k]];
            v10_load[k] = src[y0_arr[k] * srcStride + x1_arr[k]];
            v01_load[k] = src[y1_arr[k] * srcStride + x0_arr[k]];
            v11_load[k] = src[y1_arr[k] * srcStride + x1_arr[k]];
        }

        __m256 v_v00 = _mm256_loadu_ps(v00_load);
        __m256 v_v10 = _mm256_loadu_ps(v10_load);
        __m256 v_v01 = _mm256_loadu_ps(v01_load);
        __m256 v_v11 = _mm256_loadu_ps(v11_load);

        __m256 val = _mm256_mul_ps(w00, v_v00);
        val = _mm256_fmadd_ps(w10, v_v10, val);
        val = _mm256_fmadd_ps(w01, v_v01, val);
        val = _mm256_fmadd_ps(w11, v_v11, val);

        _mm256_storeu_ps(output + i, val);
    }

    // Remainder
    for (; i < n; ++i) {
        float x = xs[i];
        float y = ys[i];
        int x0 = static_cast<int>(std::floor(x));
        int y0 = static_cast<int>(std::floor(y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        x0 = std::max(0, std::min(x0, srcW - 1));
        x1 = std::max(0, std::min(x1, srcW - 1));
        y0 = std::max(0, std::min(y0, srcH - 1));
        y1 = std::max(0, std::min(y1, srcH - 1));

        float fx = x - std::floor(x);
        float fy = y - std::floor(y);

        float v00 = src[y0 * srcStride + x0];
        float v10 = src[y0 * srcStride + x1];
        float v01 = src[y1 * srcStride + x0];
        float v11 = src[y1 * srcStride + x1];

        output[i] = (1 - fx) * (1 - fy) * v00 +
                    fx * (1 - fy) * v10 +
                    (1 - fx) * fy * v01 +
                    fx * fy * v11;
    }
}

void compute_gradient(const float* src, float* gx, float* gy,
                      int width, int height, int srcStride, int dstStride) {
    // 忽略边界，内部处理
    for (int r = 1; r < height - 1; ++r) {
        int c = 1;
        // Vectorize along row
        for (; c + 8 < width - 1; c += 8) {
            // GX: (src[r][c+1] - src[r][c-1]) * 0.5
            const float* rowPtr = src + r * srcStride;
            __m256 v_plus = _mm256_loadu_ps(rowPtr + c + 1);
            __m256 v_minus = _mm256_loadu_ps(rowPtr + c - 1);
            __m256 v_gx = _mm256_mul_ps(_mm256_sub_ps(v_plus, v_minus), _mm256_set1_ps(0.5f));
            _mm256_storeu_ps(gx + r * dstStride + c, v_gx);

            // GY: (src[r+1][c] - src[r-1][c]) * 0.5
            const float* upPtr = src + (r - 1) * srcStride;
            const float* downPtr = src + (r + 1) * srcStride;
            __m256 v_up = _mm256_loadu_ps(upPtr + c);
            __m256 v_down = _mm256_loadu_ps(downPtr + c);
            __m256 v_gy = _mm256_mul_ps(_mm256_sub_ps(v_down, v_up), _mm256_set1_ps(0.5f));
            _mm256_storeu_ps(gy + r * dstStride + c, v_gy);
        }
        // Remainder
        for (; c < width - 1; ++c) {
            gx[r * dstStride + c] = (src[r * srcStride + c + 1] - src[r * srcStride + c - 1]) * 0.5f;
            gy[r * dstStride + c] = (src[(r + 1) * srcStride + c] - src[(r - 1) * srcStride + c]) * 0.5f;
        }
    }
}

void assemble_normal_equations(const float* gx, const float* gy,
                                const float* diff,
                                const float* xs, const float* ys,
                                float* AtA, float* Atl, int n) {
    // 6 参数仿射模型偏导数
    // du/da0=1, du/da1=x, du/da2=y, du/db0=0, du/db1=0, du/db2=0 (对应 x')
    // dv/db0=1, dv/db1=x, dv/db2=y, dv/da0=0, dv/da1=0, dv/da2=0 (对应 y')
    // I(x',y') ≈ T(x,y)
    // Linearized: I(x,y) + Ix*(da0 + da1*x + da2*y) + Iy*(db0 + db1*x + db2*y) = T(x,y)
    // Residual v = T(x,y) - I(x,y)
    // Obs eq: Ix*da0 + Ix*x*da1 + Ix*y*da2 + Iy*db0 + Iy*x*db1 + Iy*y*db2 = diff
    
    // A matrix row k: [Ix, Ix*x, Ix*y, Iy, Iy*x, Iy*y]
    // We supply diff = T - I_warped
    
    // Accumulators for AtA (6x6) and Atl (6x1)
    // AtA is symmetric, calculate upper triangle (21 elements)
    
    // Scalar fallback for complexity
    // But let's vectorize the accumulation
    
    // Create temporary accumulators
    __m256 v_AtA[21];
    for(int k=0; k<21; ++k) v_AtA[k] = _mm256_setzero_ps();
    __m256 v_Atl[6];
    for(int k=0; k<6; ++k) v_Atl[k] = _mm256_setzero_ps();

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v_Ix = _mm256_loadu_ps(gx + i);
        __m256 v_Iy = _mm256_loadu_ps(gy + i);
        __m256 v_l = _mm256_loadu_ps(diff + i);
        __m256 v_x = _mm256_loadu_ps(xs + i);
        __m256 v_y = _mm256_loadu_ps(ys + i);

        // Design matrix row elements
        __m256 A0 = v_Ix;
        __m256 A1 = _mm256_mul_ps(v_Ix, v_x);
        __m256 A2 = _mm256_mul_ps(v_Ix, v_y);
        __m256 A3 = v_Iy;
        __m256 A4 = _mm256_mul_ps(v_Iy, v_x);
        __m256 A5 = _mm256_mul_ps(v_Iy, v_y);

        __m256 A_col[6] = {A0, A1, A2, A3, A4, A5};

        // Update AtA upper triangle
        int idx = 0;
        for (int r = 0; r < 6; ++r) {
            for (int c = r; c < 6; ++c) {
                // AtA[r][c] += A[r] * A[c]
                v_AtA[idx] = _mm256_fmadd_ps(A_col[r], A_col[c], v_AtA[idx]);
                idx++;
            }
        }

        // Update Atl
        for (int r = 0; r < 6; ++r) {
            // Atl[r] += A[r] * l
            v_Atl[r] = _mm256_fmadd_ps(A_col[r], v_l, v_Atl[r]);
        }
    }

    // Reduce AVX accumulators
    float tmp[8];
    float AtA_sum[21] = {0};
    for(int k=0; k<21; ++k) {
        _mm256_storeu_ps(tmp, v_AtA[k]);
        for(int j=0; j<8; ++j) AtA_sum[k] += tmp[j]; 
    }
    float Atl_sum[6] = {0};
    for(int k=0; k<6; ++k) {
        _mm256_storeu_ps(tmp, v_Atl[k]);
        for(int j=0; j<8; ++j) Atl_sum[k] += tmp[j];
    }

    // Process remainder
    for (; i < n; ++i) {
        float Ix = gx[i];
        float Iy = gy[i];
        float l = diff[i];
        float x = xs[i];
        float y = ys[i];

        float row[6];
        row[0] = Ix;
        row[1] = Ix * x;
        row[2] = Ix * y;
        row[3] = Iy;
        row[4] = Iy * x;
        row[5] = Iy * y;

        int idx = 0;
        for (int r = 0; r < 6; ++r) {
            for (int c = r; c < 6; ++c) {
                AtA_sum[idx++] += row[r] * row[c];
            }
            Atl_sum[r] += row[r] * l;
        }
    }

    // Copy back to output (fill full matrix from upper triangle)
    int idx = 0;
    for (int r = 0; r < 6; ++r) {
        for (int c = r; c < 6; ++c) {
            float val = AtA_sum[idx++];
            AtA[r * 6 + c] = val;
            if (r != c) AtA[c * 6 + r] = val;
        }
        Atl[r] = Atl_sum[r];
    }
}

} // namespace simd
