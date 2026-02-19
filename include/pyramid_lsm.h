#pragma once
#ifndef PYRAMID_LSM_H
#define PYRAMID_LSM_H

#include "gaussian_pyramid.h"
#include "lsm_matcher.h"
#include <vector>

/**
 * @brief Pyramid LSM 金字塔最小二乘匹配结果
 */
struct PyramidLSMResult {
    float dx = 0.0f;              // 最终X方向亚像素偏移
    float dy = 0.0f;              // 最终Y方向亚像素偏移
    float correlation = 0.0f;     // 最终相关系数
    bool converged = false;       // 是否收敛
    int totalIterations = 0;      // 总迭代次数
    std::vector<LSMResult> levelResults;  // 各层匹配结果
    double elapsedMs = 0.0;       // 耗时(毫秒)
};

/**
 * @brief Pyramid LSM 金字塔最小二乘匹配器
 * 
 * 由粗到精的多尺度匹配策略：
 * 1. 构建模板和搜索图像的高斯金字塔
 * 2. 在最粗层进行初始匹配
 * 3. 将参数传递到下一层并精化
 * 4. 在原始分辨率层获得亚像素精度结果
 */
class PyramidLSM {
public:
    /**
     * @brief 构造函数
     * @param pyramidLevels 金字塔层数
     * @param maxIterPerLevel 每层最大迭代次数
     * @param useSIMD 是否使用 SIMD 加速
     */
    PyramidLSM(int pyramidLevels = 4, int maxIterPerLevel = 30, bool useSIMD = true);

    /**
     * @brief 执行金字塔 LSM 匹配
     * @param templateImg 模板图像
     * @param searchImg 搜索图像
     * @return PyramidLSMResult 匹配结果
     */
    PyramidLSMResult match(const Matrix2D& templateImg, const Matrix2D& searchImg);

    void setUseSIMD(bool use) { useSIMD_ = use; }
    bool getUseSIMD() const { return useSIMD_; }
    
    int getPyramidLevels() const { return pyramidLevels_; }

private:
    int pyramidLevels_;
    int maxIterPerLevel_;
    bool useSIMD_;
};

#endif // PYRAMID_LSM_H
