#include "gaussian_pyramid.h"

GaussianPyramid::GaussianPyramid(int levels, float sigma)
    : levels_(levels), sigma_(sigma) {}

void GaussianPyramid::build(const Matrix2D& image) {
    pyramid_.clear();
    pyramid_.reserve(levels_);

    // Level 0 = 原始图像
    pyramid_.push_back(image);

    // 逐层平滑 + 下采样
    for (int i = 1; i < levels_; ++i) {
        const Matrix2D& prev = pyramid_[i - 1];
        if (prev.rows() < 4 || prev.cols() < 4)
            break;  // 图像太小，停止构建

        Matrix2D smoothed = prev.gaussianSmooth(sigma_);
        Matrix2D downsampled = smoothed.downsample2x();
        pyramid_.push_back(std::move(downsampled));
    }
}

const Matrix2D& GaussianPyramid::getLevel(int level) const {
    if (level < 0 || level >= static_cast<int>(pyramid_.size()))
        throw std::out_of_range("Pyramid level out of range");
    return pyramid_[level];
}

float GaussianPyramid::getScale(int level) const {
    return 1.0f / std::pow(2.0f, static_cast<float>(level));
}
