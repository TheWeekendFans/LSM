#include "matrix2d.h"
#include <omp.h>

// ============================================================
// 构造 / 析构
// ============================================================
Matrix2D::Matrix2D() : data_(nullptr), rows_(0), cols_(0), stride_(0) {}

Matrix2D::Matrix2D(int rows, int cols) : data_(nullptr), rows_(0), cols_(0), stride_(0) {
    allocate(rows, cols);
}

Matrix2D::Matrix2D(int rows, int cols, float initVal) : data_(nullptr), rows_(0), cols_(0), stride_(0) {
    allocate(rows, cols);
    fill(initVal);
}

Matrix2D::Matrix2D(const Matrix2D& other) : data_(nullptr), rows_(0), cols_(0), stride_(0) {
    if (other.data_) {
        allocate(other.rows_, other.cols_);
        std::memcpy(data_, other.data_, totalBytes());
    }
}

Matrix2D::Matrix2D(Matrix2D&& other) noexcept
    : data_(other.data_), rows_(other.rows_), cols_(other.cols_), stride_(other.stride_) {
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.stride_ = 0;
}

Matrix2D::~Matrix2D() {
    deallocate();
}

Matrix2D& Matrix2D::operator=(const Matrix2D& other) {
    if (this != &other) {
        deallocate();
        if (other.data_) {
            allocate(other.rows_, other.cols_);
            std::memcpy(data_, other.data_, totalBytes());
        }
    }
    return *this;
}

Matrix2D& Matrix2D::operator=(Matrix2D&& other) noexcept {
    if (this != &other) {
        deallocate();
        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        stride_ = other.stride_;
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
        other.stride_ = 0;
    }
    return *this;
}

// ============================================================
// 内存管理 (32字节对齐)
// ============================================================
int Matrix2D::alignedStride(int cols) {
    // 对齐到 8 个 float (32 字节)
    return ((cols + 7) / 8) * 8;
}

void Matrix2D::allocate(int rows, int cols) {
    deallocate();
    rows_ = rows;
    cols_ = cols;
    stride_ = alignedStride(cols);
    size_t bytes = static_cast<size_t>(rows_) * stride_ * sizeof(float);
    data_ = static_cast<float*>(ALIGNED_ALLOC(32, bytes));
    if (!data_) throw std::bad_alloc();
    std::memset(data_, 0, bytes);
}

void Matrix2D::deallocate() {
    if (data_) {
        ALIGNED_FREE(data_);
        data_ = nullptr;
    }
}

// ============================================================
// 元素访问
// ============================================================
float& Matrix2D::at(int r, int c) {
    return data_[r * stride_ + c];
}

float Matrix2D::at(int r, int c) const {
    return data_[r * stride_ + c];
}

float* Matrix2D::rowPtr(int r) { return data_ + r * stride_; }
const float* Matrix2D::rowPtr(int r) const { return data_ + r * stride_; }
float* Matrix2D::data() { return data_; }
const float* Matrix2D::data() const { return data_; }

// ============================================================
// 双线性插值
// ============================================================
float Matrix2D::bilinearInterp(float x, float y) const {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 边界处理
    x0 = std::max(0, std::min(x0, cols_ - 1));
    x1 = std::max(0, std::min(x1, cols_ - 1));
    y0 = std::max(0, std::min(y0, rows_ - 1));
    y1 = std::max(0, std::min(y1, rows_ - 1));

    float fx = x - std::floor(x);
    float fy = y - std::floor(y);

    float v00 = at(y0, x0);
    float v10 = at(y0, x1);
    float v01 = at(y1, x0);
    float v11 = at(y1, x1);

    return (1 - fx) * (1 - fy) * v00 +
           fx * (1 - fy) * v10 +
           (1 - fx) * fy * v01 +
           fx * fy * v11;
}

// ============================================================
// 基本运算
// ============================================================
Matrix2D Matrix2D::operator+(const Matrix2D& rhs) const {
    Matrix2D result(rows_, cols_);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_; ++r) {
        const float* a = rowPtr(r);
        const float* b = rhs.rowPtr(r);
        float* c = result.rowPtr(r);
        for (int col = 0; col < cols_; ++col)
            c[col] = a[col] + b[col];
    }
    return result;
}

Matrix2D Matrix2D::operator-(const Matrix2D& rhs) const {
    Matrix2D result(rows_, cols_);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_; ++r) {
        const float* a = rowPtr(r);
        const float* b = rhs.rowPtr(r);
        float* c = result.rowPtr(r);
        for (int col = 0; col < cols_; ++col)
            c[col] = a[col] - b[col];
    }
    return result;
}

Matrix2D Matrix2D::operator*(const Matrix2D& rhs) const {
    if (cols_ != rhs.rows_)
        throw std::invalid_argument("Matrix dimension mismatch for multiplication");

    Matrix2D result(rows_, rhs.cols_, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows_; ++i) {
        for (int k = 0; k < cols_; ++k) {
            float aik = at(i, k);
            for (int j = 0; j < rhs.cols_; ++j) {
                result.at(i, j) += aik * rhs.at(k, j);
            }
        }
    }
    return result;
}

Matrix2D Matrix2D::operator*(float scalar) const {
    Matrix2D result(rows_, cols_);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_; ++r) {
        const float* src = rowPtr(r);
        float* dst = result.rowPtr(r);
        int col = 0;
        // AVX2 vectorized
        for (; col + 8 <= cols_; col += 8) {
            __m256 v = _mm256_loadu_ps(src + col);
            __m256 s = _mm256_set1_ps(scalar);
            _mm256_storeu_ps(dst + col, _mm256_mul_ps(v, s));
        }
        for (; col < cols_; ++col)
            dst[col] = src[col] * scalar;
    }
    return result;
}

// ============================================================
// 转置
// ============================================================
Matrix2D Matrix2D::transpose() const {
    Matrix2D result(cols_, rows_);
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < cols_; ++c)
            result.at(c, r) = at(r, c);
    return result;
}

// ============================================================
// Gauss-Jordan 求逆 (小矩阵, 通常6x6)
// ============================================================
Matrix2D Matrix2D::inverse() const {
    if (rows_ != cols_)
        throw std::invalid_argument("Only square matrices can be inverted");

    int n = rows_;
    // 增广矩阵
    Matrix2D aug(n, 2 * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            aug.at(i, j) = at(i, j);
        aug.at(i, n + i) = 1.0f;
    }

    // 高斯消元
    for (int col = 0; col < n; ++col) {
        // 寻找主元
        int maxRow = col;
        float maxVal = std::fabs(aug.at(col, col));
        for (int r = col + 1; r < n; ++r) {
            float v = std::fabs(aug.at(r, col));
            if (v > maxVal) {
                maxVal = v;
                maxRow = r;
            }
        }
        if (maxVal < 1e-12f)
            throw std::runtime_error("Matrix is singular, cannot invert");

        // 行交换
        if (maxRow != col) {
            for (int j = 0; j < 2 * n; ++j)
                std::swap(aug.at(col, j), aug.at(maxRow, j));
        }

        // 归一化当前行
        float pivot = aug.at(col, col);
        for (int j = 0; j < 2 * n; ++j)
            aug.at(col, j) /= pivot;

        // 消元
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            float factor = aug.at(r, col);
            for (int j = 0; j < 2 * n; ++j)
                aug.at(r, j) -= factor * aug.at(col, j);
        }
    }

    // 提取结果
    Matrix2D result(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result.at(i, j) = aug.at(i, n + j);

    return result;
}

// ============================================================
// 图像梯度 (中心差分)
// ============================================================
void Matrix2D::computeGradient(Matrix2D& gx, Matrix2D& gy) const {
    gx = Matrix2D(rows_, cols_, 0.0f);
    gy = Matrix2D(rows_, cols_, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int r = 1; r < rows_ - 1; ++r) {
        for (int c = 1; c < cols_ - 1; ++c) {
            gx.at(r, c) = (at(r, c + 1) - at(r, c - 1)) * 0.5f;
            gy.at(r, c) = (at(r + 1, c) - at(r - 1, c)) * 0.5f;
        }
    }
}

// ============================================================
// 高斯平滑
// ============================================================
Matrix2D Matrix2D::gaussianSmooth(float sigma) const {
    // 5x5 高斯核
    int ksize = 5;
    int half = ksize / 2;
    
    // 生成高斯核
    float kernel[5][5];
    float sum = 0.0f;
    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            float val = std::exp(-(i * i + j * j) / (2.0f * sigma * sigma));
            kernel[i + half][j + half] = val;
            sum += val;
        }
    }
    for (int i = 0; i < ksize; ++i)
        for (int j = 0; j < ksize; ++j)
            kernel[i][j] /= sum;

    Matrix2D result(rows_, cols_, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            float val = 0.0f;
            for (int ki = -half; ki <= half; ++ki) {
                for (int kj = -half; kj <= half; ++kj) {
                    int rr = std::max(0, std::min(r + ki, rows_ - 1));
                    int cc = std::max(0, std::min(c + kj, cols_ - 1));
                    val += at(rr, cc) * kernel[ki + half][kj + half];
                }
            }
            result.at(r, c) = val;
        }
    }
    return result;
}

// ============================================================
// 2x 下采样
// ============================================================
Matrix2D Matrix2D::downsample2x() const {
    int nr = rows_ / 2;
    int nc = cols_ / 2;
    Matrix2D result(nr, nc);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
            result.at(r, c) = at(r * 2, c * 2);
        }
    }
    return result;
}

// ============================================================
// 填充/统计
// ============================================================
void Matrix2D::fill(float val) {
    for (int r = 0; r < rows_; ++r) {
        float* row = rowPtr(r);
        for (int c = 0; c < cols_; ++c)
            row[c] = val;
    }
}

void Matrix2D::zeros() { fill(0.0f); }

float Matrix2D::min() const {
    float m = data_[0];
    for (int r = 0; r < rows_; ++r) {
        const float* row = rowPtr(r);
        for (int c = 0; c < cols_; ++c)
            m = std::min(m, row[c]);
    }
    return m;
}

float Matrix2D::max() const {
    float m = data_[0];
    for (int r = 0; r < rows_; ++r) {
        const float* row = rowPtr(r);
        for (int c = 0; c < cols_; ++c)
            m = std::max(m, row[c]);
    }
    return m;
}

float Matrix2D::mean() const {
    double sum = 0.0;
    for (int r = 0; r < rows_; ++r) {
        const float* row = rowPtr(r);
        for (int c = 0; c < cols_; ++c)
            sum += row[c];
    }
    return static_cast<float>(sum / (rows_ * cols_));
}
