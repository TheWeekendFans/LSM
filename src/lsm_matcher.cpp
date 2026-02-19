#include "lsm_matcher.h"
#include "simd_ops.h"
#include <iostream>
#include <vector>
#include <algorithm>

LSMMatcher::LSMMatcher(int maxIter, float convergenceThreshold, bool useSIMD)
    : maxIter_(maxIter), convergenceThreshold_(convergenceThreshold), useSIMD_(useSIMD) {}

void LSMMatcher::setROI(int x, int y, int width, int height) {
    roiX_ = x;
    roiY_ = y;
    roiW_ = width;
    roiH_ = height;
    hasROI_ = true;
}

void LSMMatcher::clearROI() {
    hasROI_ = false;
}

float LSMMatcher::computeCorrelation(const Matrix2D& img1, const Matrix2D& img2,
                                     int startX, int startY, int w, int h) {
    // 简单的归一化互相关 (NCC)
    double sum1 = 0.0, sum2 = 0.0, sum12 = 0.0;
    double sqSum1 = 0.0, sqSum2 = 0.0;
    int count = 0;

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            float v1 = img1.at(startY + r, startX + c);
            float v2 = img2.at(startY + r, startX + c); // 注意：这里假设两图已对齐或在同一坐标系
            sum1 += v1;
            sum2 += v2;
            sum12 += v1 * v2;
            sqSum1 += v1 * v1;
            sqSum2 += v2 * v2;
            count++;
        }
    }

    double mean1 = sum1 / count;
    double mean2 = sum2 / count;
    double var1 = sqSum1 / count - mean1 * mean1;
    double var2 = sqSum2 / count - mean2 * mean2;

    if (var1 < 1e-9 || var2 < 1e-9) return 0.0f;

    // Covariance / (std1 * std2) is not exactly this but close enough for simple correlation metric
    // More precise: pearson correlation
    
    double num = 0.0;
    double den1 = 0.0;
    double den2 = 0.0;
    
    // Re-loop for numerical stability or use the formula above
    // Pearson = (E[XY] - E[X]E[Y]) / (sigmaX * sigmaY)
    double cov = sum12 / count - mean1 * mean2;
    return static_cast<float>(cov / (std::sqrt(var1) * std::sqrt(var2)));
}

LSMResult LSMMatcher::match(const Matrix2D& templateImg, const Matrix2D& searchImg,
                            float initDx, float initDy) {
    if (useSIMD_ && simd::cpu_supports_avx2()) {
        return matchCoreSIMD(templateImg, searchImg, initDx, initDy);
    } else {
        return matchCore(templateImg, searchImg, initDx, initDy);
    }
}

LSMResult LSMMatcher::matchCore(const Matrix2D& templateImg, const Matrix2D& searchImg,
                                float initDx, float initDy) {
    LSMResult result;
    result.dx = initDx;
    result.dy = initDy;
    result.a0 = initDx; result.a1 = 1.0f; result.a2 = 0.0f;
    result.b0 = initDy; result.b1 = 0.0f; result.b2 = 1.0f;

    // 确定模板区域
    int tx = hasROI_ ? roiX_ : 0;
    int ty = hasROI_ ? roiY_ : 0;
    int tw = hasROI_ ? roiW_ : templateImg.cols();
    int th = hasROI_ ? roiH_ : templateImg.rows();

    // 计算搜索图梯度 (用于 Jacobian)
    Matrix2D searchGx, searchGy;
    searchImg.computeGradient(searchGx, searchGy);

    const int maxPoints = tw * th;
    std::vector<float> AtA(36, 0.0f);
    std::vector<float> Atl(6, 0.0f);
    
    // 参数更新量
    float da[6] = {0}; // da0, da1, da2, db0, db1, db2

    for (int iter = 0; iter < maxIter_; ++iter) {
        std::fill(AtA.begin(), AtA.end(), 0.0f);
        std::fill(Atl.begin(), Atl.end(), 0.0f);
        
        double sqResSum = 0.0;
        int validPoints = 0;

        for (int r = 0; r < th; ++r) {
            for (int c = 0; c < tw; ++c) {
                // 模板中心化坐标 (相对于 ROI 中心)
                // 也可以直接用相对于 (0,0) 的坐标，这里简单起见用 ROI 局部坐标
                // 但为了仿射变换参数有物理意义 (平移对应中心移动)，建议将坐标原点设为 ROI 中心
                // 这里暂时用像素坐标，注意数值稳定性
                float x = static_cast<float>(c);
                float y = static_cast<float>(r);

                // 仿射变换: (x, y) -> (srcX, srcY) 在搜索图中的位置
                // 坐标系对齐：模板的 (c,r) 对应绝对坐标 (tx+c, ty+r)
                // 我们假设仿射参数是关于 (tx+c, ty+r) 的，或者关于局部坐标的？
                // 通常 LSM 估计的是 `search_coord = transform(template_coord)`
                // 初始状态: search_coord = template_coord + (dx, dy)
                // 所以 result.a0 是 total shift X
                
                float tmplX = static_cast<float>(tx + c);
                float tmplY = static_cast<float>(ty + r);

                float srcX = result.a0 + result.a1 * tmplX + result.a2 * tmplY;
                float srcY = result.b0 + result.b1 * tmplX + result.b2 * tmplY;

                // 边界检查
                if (srcX < 0 || srcX > searchImg.cols() - 1 || srcY < 0 || srcY > searchImg.rows() - 1)
                    continue;

                // 双线性插值获取搜索图灰度和梯度
                float sVal = searchImg.bilinearInterp(srcX, srcY);
                float sGx = searchGx.bilinearInterp(srcX, srcY);
                float sGy = searchGy.bilinearInterp(srcX, srcY);

                float tVal = templateImg.at(ty + r, tx + c);

                // 残差 v = T(x,y) - S(x',y')
                // 线性化: S(x') approx S(x'_current) + dS/dx * dx + dS/dy * dy
                // dx = da0 + x*da1 + y*da2
                // dy = db0 + x*db1 + y*db2
                // Obs eq: S(curr) + Sx*(...) + Sy*(...) = T
                // => Sx * (da...) + Sy * (db...) = T - S(curr) = diff
                float diff = tVal - sVal;
                
                sqResSum += diff * diff;
                validPoints++;

                // 构建 A 矩阵的一行 (6个元素)
                // [Sx, Sx*x, Sx*y, Sy, Sy*x, Sy*y]
                float row[6];
                row[0] = sGx;
                row[1] = sGx * tmplX;
                row[2] = sGx * tmplY;
                row[3] = sGy;
                row[4] = sGy * tmplX;
                row[5] = sGy * tmplY;

                // 累加 AtA (6x6) 和 Atl (6x1)
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 6; ++j) {
                        AtA[i * 6 + j] += row[i] * row[j];
                    }
                    Atl[i] += row[i] * diff;
                }
            }
        }

        if (validPoints < 10) break; // 点太少，无法计算

        result.residualRMS = static_cast<float>(std::sqrt(sqResSum / validPoints));

        // 解法方程
        if (!solveNormalEquation(AtA.data(), Atl.data(), da)) {
            // 求解失败 (矩阵奇异)
            break; 
        }

        // 更新参数
        result.a0 += da[0];
        result.a1 += da[1];
        result.a2 += da[2];
        result.b0 += da[3];
        result.b1 += da[4];
        result.b2 += da[5];

        result.dx = result.a0; // 更新平移分量，注意这里a0包含了原点信息吗？
        // 如果我们用的是绝对坐标 (tmplX, tmplY) 作为输入，那么 a0 就是 X 方向的截距
        // 如果 initDx 是相对于 identity 的偏移，我们需要小心。
        // 上面的公式 srcX = a0 + a1*x + a2*y
        // 当 init 时，srcX = x + dx => a0=dx, a1=1, a2=0. Correct.
        // 所以 result.dx 实际上并不单独存在，它融合在 a0 中，
        // 但为了方便用户理解 "shift"，我们可以近似认为 a0 附近是 shift, 
        // 但如果 a1!=1, a2!=0, 真正的 shift 取决于中心点位置。
        // 我们通常输出中心点的 shift。
        float cx = tx + tw / 2.0f;
        float cy = ty + th / 2.0f;
        float finalCx = result.a0 + result.a1 * cx + result.a2 * cy;
        float finalCy = result.b0 + result.b1 * cx + result.b2 * cy;
        result.dx = finalCx - cx;
        result.dy = finalCy - cy;

        result.iterations = iter + 1;

        // 检查收敛 (以参数更新量为准)
        // 简单判断平移量的变化
        if (std::fabs(da[0]) < convergenceThreshold_ && std::fabs(da[3]) < convergenceThreshold_) {
            result.converged = true;
            break;
        }
    }
    
    // 计算最终相关系数 (基于最终位置)
    // 需要重新 warp search region 到 template region 大小
    // 这里简化，只在整数像素对齐位置计算，或者跳过
    // 为了性能，我们可以只计算 RMS
    return result;
}

LSMResult LSMMatcher::matchCoreSIMD(const Matrix2D& templateImg, const Matrix2D& searchImg,
                                    float initDx, float initDy) {
    LSMResult result;
    result.dx = initDx;
    result.dy = initDy;
    result.a0 = initDx; result.a1 = 1.0f; result.a2 = 0.0f;
    result.b0 = initDy; result.b1 = 0.0f; result.b2 = 1.0f;

    int tx = hasROI_ ? roiX_ : 0;
    int ty = hasROI_ ? roiY_ : 0;
    int tw = hasROI_ ? roiW_ : templateImg.cols();
    int th = hasROI_ ? roiH_ : templateImg.rows();
    int numPixels = tw * th;

    // 预分配缓冲区 (对齐)
    // 我们需要存储每一轮所有像素的坐标, 梯度, diff
    float* bufXs = (float*)ALIGNED_ALLOC(32, numPixels * sizeof(float));
    float* bufYs = (float*)ALIGNED_ALLOC(32, numPixels * sizeof(float));
    float* bufTmplXs = (float*)ALIGNED_ALLOC(32, numPixels * sizeof(float)); // 模板原始坐标X
    float* bufTmplYs = (float*)ALIGNED_ALLOC(32, numPixels * sizeof(float)); // 模板原始坐标Y
    float* bufSVal = (float*)ALIGNED_ALLOC(32, numPixels * sizeof(float));
    float* bufGx = (float*)ALIGNED_ALLOC(32, numPixels * sizeof(float));
    float* bufGy = (float*)ALIGNED_ALLOC(32, numPixels * sizeof(float));
    float* bufDiff = (float*)ALIGNED_ALLOC(32, numPixels * sizeof(float));

    // 预计算模板坐标 (固定不变)
    for (int r = 0; r < th; ++r) {
        for (int c = 0; c < tw; ++c) {
            bufTmplXs[r * tw + c] = static_cast<float>(tx + c);
            bufTmplYs[r * tw + c] = static_cast<float>(ty + r);
        }
    }

    // 计算搜索图梯度 (整体)
    Matrix2D searchGx, searchGy;
    searchImg.computeGradient(searchGx, searchGy);

    float da[6] = {0};

    // 迭代
    for (int iter = 0; iter < maxIter_; ++iter) {
        // 1. 计算变换后的坐标
        // x' = a0 + a1*x + a2*y
        // y' = b0 + b1*x + b2*y
        // 这里可以用 SIMD 优化坐标计算，简单起见先循环
        // 实际上，这完全可以是向量化的 ax + by + c
        // simd::weighted_sum 可以用吗？不太适用，手写循环即可，编译器会自动优化简单算术
        #pragma omp parallel for
        for (int i = 0; i < numPixels; ++i) {
            float x = bufTmplXs[i];
            float y = bufTmplYs[i];
            bufXs[i] = result.a0 + result.a1 * x + result.a2 * y;
            bufYs[i] = result.b0 + result.b1 * x + result.b2 * y;
        }

        // 2. 批量双线性插值获取 Value, Gx, Gy
        simd::batch_bilinear_interp(searchImg.data(), searchImg.cols(), searchImg.rows(), searchImg.stride(),
                                    bufXs, bufYs, bufSVal, numPixels);
        simd::batch_bilinear_interp(searchGx.data(), searchGx.cols(), searchGx.rows(), searchGx.stride(),
                                    bufXs, bufYs, bufGx, numPixels);
        simd::batch_bilinear_interp(searchGy.data(), searchGy.cols(), searchGy.rows(), searchGy.stride(),
                                    bufXs, bufYs, bufGy, numPixels);

        // 3. 计算 Diff = T - S
        // T 也是不变量，可以预取，这里直接从 image 读取
        // 优化：可以将 T 预读到 buffer
        #pragma omp parallel for
        for (int r = 0; r < th; ++r) {
            for (int c = 0; c < tw; ++c) {
                int i = r * tw + c;
                bufDiff[i] = templateImg.at(ty + r, tx + c) - bufSVal[i];
            }
        }

        // 4. 组装法方程
        // A^T * A 和 A^T * l
        std::vector<float> AtA(36, 0.0f); // 内部其实不需要 aligned，因为 output small
        std::vector<float> Atl(6, 0.0f);
        
        // 使用 SIMD 累加
        // 注意：bufTmplXs 是 x, bufTmplYs 是 y
        simd::assemble_normal_equations(bufGx, bufGy, bufDiff, bufTmplXs, bufTmplYs,
                                        AtA.data(), Atl.data(), numPixels);

        // 5. 求解
        if (!solveNormalEquation(AtA.data(), Atl.data(), da)) {
            break;
        }

        // 6. 更新参数
        result.a0 += da[0];
        result.a1 += da[1];
        result.a2 += da[2];
        result.b0 += da[3];
        result.b1 += da[4];
        result.b2 += da[5];

        result.iterations = iter + 1;

        // 计算 RMS (可选，耗时)
        // ...

        // 检查收敛
        if (std::fabs(da[0]) < convergenceThreshold_ && std::fabs(da[3]) < convergenceThreshold_) {
            result.converged = true;
            break;
        }
    }

    // 计算结果 DX DY
    float cx = tx + tw / 2.0f;
    float cy = ty + th / 2.0f;
    float finalCx = result.a0 + result.a1 * cx + result.a2 * cy;
    float finalCy = result.b0 + result.b1 * cx + result.b2 * cy;
    result.dx = finalCx - cx;
    result.dy = finalCy - cy;

    ALIGNED_FREE(bufXs);
    ALIGNED_FREE(bufYs);
    ALIGNED_FREE(bufTmplXs);
    ALIGNED_FREE(bufTmplYs);
    ALIGNED_FREE(bufSVal);
    ALIGNED_FREE(bufGx);
    ALIGNED_FREE(bufGy);
    ALIGNED_FREE(bufDiff);

    return result;
}

bool LSMMatcher::solveNormalEquation(const float* AtA, const float* Atl, float* x) {
    // 6x6 矩阵求逆求解
    // 使用 Matrix2D 的 Gauss-Jordan
    Matrix2D A(6, 6);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            A.at(i, j) = AtA[i * 6 + j];
        }
    }

    try {
        Matrix2D A_inv = A.inverse();
        // x = A_inv * Atl
        for (int i = 0; i < 6; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 6; ++j) {
                sum += A_inv.at(i, j) * Atl[j];
            }
            x[i] = sum;
        }
        return true;
    } catch (...) {
        return false;
    }
}
