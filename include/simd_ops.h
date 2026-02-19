#pragma once
#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <immintrin.h>
#include <cstdint>

/**
 * @brief AVX2 SIMD 加速操作集
 * 
 * 提供矩阵运算、图像插值等核心计算的 SIMD 加速版本
 * 所有函数假设输入数据按 32 字节对齐
 */
namespace simd {

/**
 * @brief AVX2 向量点积
 * @param a 向量a (对齐)
 * @param b 向量b (对齐)
 * @param n 向量长度
 * @return 点积结果
 */
float dot_product(const float* a, const float* b, int n);

/**
 * @brief AVX2 矩阵乘法 C = A * B
 * @param A 矩阵A (M x K, 行优先)
 * @param B 矩阵B (K x N, 行优先)
 * @param C 结果矩阵 (M x N, 行优先)
 * @param M A的行数
 * @param K A的列数/B的行数
 * @param N B的列数
 * @param strideA A每行的stride
 * @param strideB B每行的stride
 * @param strideC C每行的stride
 */
void matrix_multiply(const float* A, const float* B, float* C,
                     int M, int K, int N,
                     int strideA, int strideB, int strideC);

/**
 * @brief AVX2 批量双线性插值
 * @param src 源图像数据
 * @param srcW 源图像宽度
 * @param srcH 源图像高度
 * @param srcStride 源图像stride
 * @param xs X坐标数组
 * @param ys Y坐标数组
 * @param output 输出数组
 * @param n 插值点数
 */
void batch_bilinear_interp(const float* src, int srcW, int srcH, int srcStride,
                           const float* xs, const float* ys, float* output, int n);

/**
 * @brief AVX2 图像梯度计算 (Sobel)
 * @param src 源图像数据
 * @param gx X方向梯度输出
 * @param gy Y方向梯度输出
 * @param width 宽度
 * @param height 高度
 * @param srcStride 源stride
 * @param dstStride 目标stride
 */
void compute_gradient(const float* src, float* gx, float* gy,
                      int width, int height, int srcStride, int dstStride);

/**
 * @brief AVX2 法方程组装 (用于 LSM)
 * 
 * 对每个像素点，计算 A^T*A 和 A^T*l 的累加
 * @param gx X梯度
 * @param gy Y梯度
 * @param diff 灰度差
 * @param xs 归一化X坐标
 * @param ys 归一化Y坐标
 * @param AtA 输出 6x6 法方程矩阵 (上三角存储)
 * @param Atl 输出 6x1 法方程右端
 * @param n 像素数
 */
void assemble_normal_equations(const float* gx, const float* gy,
                                const float* diff,
                                const float* xs, const float* ys,
                                float* AtA, float* Atl, int n);

/**
 * @brief AVX2 向量加权和
 */
void weighted_sum(const float* a, const float* b, float* c,
                  float wa, float wb, int n);

/**
 * @brief 检测 CPU 是否支持 AVX2
 */
bool cpu_supports_avx2();

} // namespace simd

#endif // SIMD_OPS_H
