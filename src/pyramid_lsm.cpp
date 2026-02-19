#include "pyramid_lsm.h"
#include <iostream>
#include <chrono>

PyramidLSM::PyramidLSM(int pyramidLevels, int maxIterPerLevel, bool useSIMD)
    : pyramidLevels_(pyramidLevels), maxIterPerLevel_(maxIterPerLevel), useSIMD_(useSIMD) {}

PyramidLSMResult PyramidLSM::match(const Matrix2D& templateImg, const Matrix2D& searchImg) {
    auto startTime = std::chrono::high_resolution_clock::now();
    PyramidLSMResult result;

    // 1. 构建金字塔
    GaussianPyramid tmplPyramid(pyramidLevels_);
    GaussianPyramid searchPyramid(pyramidLevels_);
    
    tmplPyramid.build(templateImg);
    searchPyramid.build(searchImg);

    // 实际层数可能小于请求层数（如果图像太小）
    int actualLevels = std::min(tmplPyramid.numLevels(), searchPyramid.numLevels());

    float currentDx = 0.0f;
    float currentDy = 0.0f;

    // 2. 由粗到精匹配
    LSMMatcher matcher(maxIterPerLevel_, 1e-4f, useSIMD_);

    for (int level = actualLevels - 1; level >= 0; --level) {
        const Matrix2D& tImg = tmplPyramid.getLevel(level);
        const Matrix2D& sImg = searchPyramid.getLevel(level);

        // 设置 ROI: 假设 template 是小的 pattern，search 是大的
        // 如果 template 和 search 一样大，我们匹配整个区域
        // 这里根据 template 大小全图匹配
        // 注意：LSMMatcher 默认全图匹配，如果没设 ROI
        matcher.clearROI(); 

        // 上一层的结果乘 2 (下采样逆操作)
        if (level < actualLevels - 1) {
            currentDx *= 2.0f;
            currentDy *= 2.0f;
        }

        // 执行单层匹配
        // 注意：LSM 容易陷入局部极值，金字塔粗层提供了较好的初始位置
        
        // [OPTIMIZATION]
        // 对于 Level 0 (原始分辨率)，如果是合成的高 SNR 图像，边缘过于锐利会导致梯度计算不稳定(Aliasing)。
        // 施加轻微的高斯平滑 (sigma=0.7) 可以显著改善收敛性。
        if (level == 0) {
           Matrix2D smoothedT = tImg.gaussianSmooth(0.7f);
           Matrix2D smoothedS = sImg.gaussianSmooth(0.7f);
           LSMResult lr = matcher.match(smoothedT, smoothedS, currentDx, currentDy);
           
           result.levelResults.push_back(lr);
           result.totalIterations += lr.iterations;
           
           // Use the result
           if (lr.converged) {
               currentDx = lr.dx;
               currentDy = lr.dy;
           }
           
           result.dx = lr.dx;
           result.dy = lr.dy;
           result.correlation = lr.residualRMS; 
           result.converged = lr.converged;
           continue;
        }

        LSMResult lr = matcher.match(tImg, sImg, currentDx, currentDy);

        result.levelResults.push_back(lr);
        result.totalIterations += lr.iterations;

        if (lr.converged) {
            currentDx = lr.dx;
            currentDy = lr.dy;
        } else {
            // 如果不收敛，可能需要保留上层结果或者报错？
            // 这里继续使用不收敛的结果，但在下一层可能会修正
            currentDx = lr.dx;
            currentDy = lr.dy;
        }

        // 最精细层 (Level 0) 的结果作为最终结果
        if (level == 0) {
            result.dx = lr.dx;
            result.dy = lr.dy;
            result.correlation = lr.correlation; // LSMMatcher 当前未计算 correlation，需补充或在此计算
            result.converged = lr.converged;
            result.correlation = lr.residualRMS; // 暂用 residual替代
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = endTime - startTime;
    result.elapsedMs = elapsed.count();

    return result;
}
