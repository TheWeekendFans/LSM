#include "noise_generator.h"
#include <cmath>

NoiseGenerator::NoiseGenerator(unsigned int seed) : rng_(seed) {}

void NoiseGenerator::setSeed(unsigned int seed) {
    rng_.seed(seed);
}

Matrix2D NoiseGenerator::addGaussianNoise(const Matrix2D& image, float sigma) {
    Matrix2D result(image.rows(), image.cols());
    std::normal_distribution<float> dist(0.0f, sigma);

    for (int r = 0; r < image.rows(); ++r) {
        for (int c = 0; c < image.cols(); ++c) {
            result.at(r, c) = image.at(r, c) + dist(rng_);
        }
    }
    return result;
}

Matrix2D NoiseGenerator::addPoissonNoise(const Matrix2D& image) {
    Matrix2D result(image.rows(), image.cols());

    for (int r = 0; r < image.rows(); ++r) {
        for (int c = 0; c < image.cols(); ++c) {
            float val = std::max(0.0f, image.at(r, c));
            if (val < 50.0f) {
                // 小值用精确泊松分布
                std::poisson_distribution<int> poisson(static_cast<double>(val));
                result.at(r, c) = static_cast<float>(poisson(rng_));
            } else {
                // 大值用正态近似
                std::normal_distribution<float> approx(val, std::sqrt(val));
                result.at(r, c) = approx(rng_);
            }
        }
    }
    return result;
}

Matrix2D NoiseGenerator::addMixedNoise(const Matrix2D& image, float gaussSigma, float poissonScale) {
    // 先加泊松噪声
    Matrix2D scaled(image.rows(), image.cols());
    for (int r = 0; r < image.rows(); ++r)
        for (int c = 0; c < image.cols(); ++c)
            scaled.at(r, c) = image.at(r, c) * poissonScale;

    Matrix2D poissonNoisy = addPoissonNoise(scaled);

    // 再缩放回来并加高斯噪声
    Matrix2D result(image.rows(), image.cols());
    std::normal_distribution<float> gaussDist(0.0f, gaussSigma);

    for (int r = 0; r < image.rows(); ++r) {
        for (int c = 0; c < image.cols(); ++c) {
            result.at(r, c) = poissonNoisy.at(r, c) / poissonScale + gaussDist(rng_);
        }
    }
    return result;
}

float NoiseGenerator::computeSignalPower(const Matrix2D& image) {
    double sum = 0.0, sumSq = 0.0;
    int n = image.rows() * image.cols();
    for (int r = 0; r < image.rows(); ++r) {
        for (int c = 0; c < image.cols(); ++c) {
            float v = image.at(r, c);
            sum += v;
            sumSq += v * v;
        }
    }
    double mean = sum / n;
    // 信号功率 = 方差
    return static_cast<float>(sumSq / n - mean * mean);
}

Matrix2D NoiseGenerator::addNoiseWithSNR(const Matrix2D& image, float targetSNR) {
    float signalPower = computeSignalPower(image);
    // SNR(dB) = 10 * log10(signalPower / noisePower)
    // noisePower = signalPower / 10^(SNR/10)
    float noisePower = signalPower / std::pow(10.0f, targetSNR / 10.0f);
    float noiseSigma = std::sqrt(noisePower);

    // 使用混合噪声模型（高斯为主，泊松为辅）
    float gaussSigma = noiseSigma * 0.8f;
    float poissonScale = 1.0f / (noiseSigma * 0.2f + 0.001f);

    return addMixedNoise(image, gaussSigma, poissonScale);
}
