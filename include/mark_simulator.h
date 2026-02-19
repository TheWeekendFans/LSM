#pragma once
#ifndef MARK_SIMULATOR_H
#define MARK_SIMULATOR_H

#include "matrix2d.h"
#include "noise_generator.h"
#include <cmath>

/**
 * @brief 晶圆 Alignment Mark 模拟器
 * 
 * 生成含各种系统误差和噪声的模拟对准标记图像：
 * - Box-in-Box / 十字线 mark
 * - Mark asymmetry（不对称性）
 * - Wafer stress 变形
 * - 已知亚像素偏移
 * - Gaussian + Poisson 噪声
 */
struct MarkSimConfig {
    int imageSize = 128;           // 图像尺寸 (imageSize x imageSize)
    float markWidth = 40.0f;       // mark 外框宽度
    float markLineWidth = 4.0f;    // mark 线宽
    float innerMarkWidth = 20.0f;  // 内框宽度
    float innerLineWidth = 3.0f;   // 内框线宽

    // 偏移参数（真值）
    float trueOffsetX = 0.0f;      // X方向亚像素偏移
    float trueOffsetY = 0.0f;      // Y方向亚像素偏移

    // 噪声参数
    float snr = 30.0f;             // 信噪比 (dB)
    
    // Mark asymmetry (0~1, 0=完全对称, 0.15=15%不对称)
    float asymmetry = 0.0f;

    // Wafer stress 变形系数
    float stressCoeffX = 0.0f;     // X方向应力变形
    float stressCoeffY = 0.0f;     // Y方向应力变形

    // 基础灰度值
    float backgroundIntensity = 50.0f;
    float markIntensity = 200.0f;

    unsigned int seed = 42;
};

class MarkSimulator {
public:
    MarkSimulator();

    /**
     * @brief 生成参考标记图像(模板)
     */
    Matrix2D generateTemplate(const MarkSimConfig& config);

    /**
     * @brief 生成搜索图像（含偏移、变形、asymmetry、噪声）
     */
    Matrix2D generateSearchImage(const MarkSimConfig& config);

    /**
     * @brief 生成一对配对图像(模板+搜索)
     * @return pair<template, search>
     */
    std::pair<Matrix2D, Matrix2D> generatePair(const MarkSimConfig& config);

private:
    NoiseGenerator noiseGen_;

    // 绘制 Box-in-Box mark（抗锯齿亚像素精度）
    void drawBoxMark(Matrix2D& img, float cx, float cy,
                     float width, float lineWidth, float intensity,
                     float asymmetryShift = 0.0f);

    // 应用 wafer stress 变形
    Matrix2D applyStressDeformation(const Matrix2D& img,
                                     float stressX, float stressY);

    // 抗锯齿亚像素线条绘制
    void drawSubpixelRect(Matrix2D& img, float x0, float y0,
                          float x1, float y1, float lineWidth,
                          float intensity, float asymShift = 0.0f);
};

#endif // MARK_SIMULATOR_H
