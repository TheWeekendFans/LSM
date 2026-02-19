#pragma once
#ifndef NOISE_GENERATOR_H
#define NOISE_GENERATOR_H

#include "matrix2d.h"
#include <random>
#include <cmath>

/**
 * @brief 噪声生成器 - 支持高斯噪声、泊松噪声及混合噪声
 * 
 * 用于模拟真实晶圆成像过程中的噪声特征
 */
class NoiseGenerator {
public:
    explicit NoiseGenerator(unsigned int seed = 42);

    /**
     * @brief 添加高斯噪声
     * @param image 输入图像
     * @param sigma 噪声标准差
     * @return 添加噪声后的图像
     */
    Matrix2D addGaussianNoise(const Matrix2D& image, float sigma);

    /**
     * @brief 添加泊松噪声
     * @param image 输入图像（值代表光子计数）
     * @return 添加噪声后的图像
     */
    Matrix2D addPoissonNoise(const Matrix2D& image);

    /**
     * @brief 添加混合噪声（高斯+泊松）
     * @param image 输入图像
     * @param gaussSigma 高斯噪声标准差
     * @param poissonScale 泊松噪声缩放因子
     * @return 添加噪声后的图像
     */
    Matrix2D addMixedNoise(const Matrix2D& image, float gaussSigma, float poissonScale = 1.0f);

    /**
     * @brief 根据目标SNR添加噪声
     * @param image 干净的输入图像
     * @param targetSNR 目标信噪比(dB)
     * @return 添加噪声后的图像
     */
    Matrix2D addNoiseWithSNR(const Matrix2D& image, float targetSNR);

    /**
     * @brief 设置随机种子
     */
    void setSeed(unsigned int seed);

private:
    std::mt19937 rng_;
    
    float computeSignalPower(const Matrix2D& image);
};

#endif // NOISE_GENERATOR_H
