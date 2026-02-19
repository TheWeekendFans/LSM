#pragma once
#ifndef GAUSSIAN_PYRAMID_H
#define GAUSSIAN_PYRAMID_H

#include "matrix2d.h"
#include <vector>

/**
 * @brief 高斯金字塔构建器
 * 
 * 对图像进行多尺度分解，支持 Pyramid LSM 的由粗到精匹配策略
 */
class GaussianPyramid {
public:
    /**
     * @brief 构造函数
     * @param levels 金字塔层数 (包含原始图像)
     * @param sigma 高斯平滑核标准差
     */
    GaussianPyramid(int levels = 4, float sigma = 1.0f);

    /**
     * @brief 构建金字塔
     * @param image 原始图像（最高分辨率）
     */
    void build(const Matrix2D& image);

    /**
     * @brief 获取指定层的图像
     * @param level 层索引 (0=最高分辨率/原图)
     */
    const Matrix2D& getLevel(int level) const;
    
    /**
     * @brief 获取金字塔层数
     */
    int numLevels() const { return static_cast<int>(pyramid_.size()); }

    /**
     * @brief 获取缩放因子
     */
    float getScale(int level) const;

private:
    int levels_;
    float sigma_;
    std::vector<Matrix2D> pyramid_;
};

#endif // GAUSSIAN_PYRAMID_H
