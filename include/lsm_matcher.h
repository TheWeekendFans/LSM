#pragma once
#ifndef LSM_MATCHER_H
#define LSM_MATCHER_H

#include "matrix2d.h"
#include <cmath>

/**
 * @brief LSM 最小二乘匹配结果
 */
struct LSMResult {
    float dx = 0.0f;          // X方向亚像素偏移
    float dy = 0.0f;          // Y方向亚像素偏移
    float a0 = 0.0f, a1 = 1.0f, a2 = 0.0f;  // 仿射参数 x' = a0 + a1*x + a2*y
    float b0 = 0.0f, b1 = 0.0f, b2 = 1.0f;  // 仿射参数 y' = b0 + b1*x + b2*y
    float correlation = 0.0f;  // 相关系数
    int iterations = 0;        // 迭代次数
    bool converged = false;    // 是否收敛
    float residualRMS = 0.0f;  // 残差RMS
};

/**
 * @brief LSM 最小二乘匹配器
 * 
 * 基于灰度梯度的最小二乘匹配，6参数仿射变换模型
 * 核心算法：迭代求解法方程 A^T*A * p = A^T * l
 */
class LSMMatcher {
public:
    /**
     * @brief 构造函数
     * @param maxIter 最大迭代次数
     * @param convergenceThreshold 收敛阈值
     * @param useSIMD 是否使用 AVX2 SIMD 加速
     */
    LSMMatcher(int maxIter = 30, float convergenceThreshold = 1e-6f, bool useSIMD = true);

    /**
     * @brief 执行 LSM 匹配
     * @param templateImg 模板图像
     * @param searchImg 搜索图像
     * @param initDx 初始X偏移估计
     * @param initDy 初始Y偏移估计
     * @return LSMResult 匹配结果
     */
    LSMResult match(const Matrix2D& templateImg, const Matrix2D& searchImg,
                    float initDx = 0.0f, float initDy = 0.0f);

    /**
     * @brief 设置匹配区域 ROI
     */
    void setROI(int x, int y, int width, int height);
    void clearROI();

    void setUseSIMD(bool use) { useSIMD_ = use; }
    bool getUseSIMD() const { return useSIMD_; }

private:
    int maxIter_;
    float convergenceThreshold_;
    bool useSIMD_;
    
    // ROI 参数
    bool hasROI_ = false;
    int roiX_ = 0, roiY_ = 0, roiW_ = 0, roiH_ = 0;

    // 标准匹配核心
    LSMResult matchCore(const Matrix2D& templateImg, const Matrix2D& searchImg,
                        float initDx, float initDy);

    // SIMD 加速匹配核心
    LSMResult matchCoreSIMD(const Matrix2D& templateImg, const Matrix2D& searchImg,
                            float initDx, float initDy);

    // 计算相关系数
    float computeCorrelation(const Matrix2D& img1, const Matrix2D& img2,
                             int startX, int startY, int w, int h);

    // 解法方程 6x6
    bool solveNormalEquation(const float* AtA, const float* Atl, float* params);
};

#endif // LSM_MATCHER_H
