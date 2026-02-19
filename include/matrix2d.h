#pragma once
#ifndef MATRIX2D_H
#define MATRIX2D_H

#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <immintrin.h>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

/**
 * @brief 二维浮点矩阵类，支持 AVX2 对齐内存布局
 * 
 * 用于图像存储与矩阵运算，行优先存储，32字节对齐
 */
class Matrix2D {
public:
    Matrix2D();
    Matrix2D(int rows, int cols);
    Matrix2D(int rows, int cols, float initVal);
    Matrix2D(const Matrix2D& other);
    Matrix2D(Matrix2D&& other) noexcept;
    ~Matrix2D();

    Matrix2D& operator=(const Matrix2D& other);
    Matrix2D& operator=(Matrix2D&& other) noexcept;

    // 元素访问
    float& at(int r, int c);
    float at(int r, int c) const;
    float* rowPtr(int r);
    const float* rowPtr(int r) const;
    float* data();
    const float* data() const;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int stride() const { return stride_; }
    size_t totalBytes() const { return static_cast<size_t>(rows_) * stride_ * sizeof(float); }

    // 双线性插值
    float bilinearInterp(float x, float y) const;

    // 基本运算
    Matrix2D operator+(const Matrix2D& rhs) const;
    Matrix2D operator-(const Matrix2D& rhs) const;
    Matrix2D operator*(const Matrix2D& rhs) const;  // 矩阵乘法
    Matrix2D operator*(float scalar) const;

    // 转置
    Matrix2D transpose() const;

    // 求逆 (Gauss-Jordan)
    Matrix2D inverse() const;

    // 图像梯度
    void computeGradient(Matrix2D& gx, Matrix2D& gy) const;

    // 高斯平滑
    Matrix2D gaussianSmooth(float sigma = 1.0f) const;

    // 2倍下采样
    Matrix2D downsample2x() const;

    // 填充
    void fill(float val);
    void zeros();

    // 统计
    float min() const;
    float max() const;
    float mean() const;

private:
    float* data_;
    int rows_;
    int cols_;
    int stride_;  // 每行实际包含的 float 数（含padding，32字节对齐）

    void allocate(int rows, int cols);
    void deallocate();
    static int alignedStride(int cols);
};

#endif // MATRIX2D_H
