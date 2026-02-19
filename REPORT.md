# 半导体晶圆对准引擎技术报告

## 1. 算法原理
本系统采用 **Pyramid Least Squares Matching (LSM)** 算法。

### 1.1 最小二乘匹配 (LSM)
LSM 通过最小化模板图像 $T(x,y)$ 与变形后的搜索图像 $I(x',y')$ 之间的灰度差异平方和来求解最佳变换参数。
目标函数：
$$ E = \sum_{x,y} [ T(x,y) - I(a_0 + a_1 x + a_2 y, b_0 + b_1 x + b_2 y) ]^2 $$
通过泰勒展开线性化，建立法方程组 $A^T A \Delta p = A^T l$，迭代求解参数增量 $\Delta p$。
其中设计矩阵 $A$ 包含图像梯度信息，观测向量 $l$ 为灰度残差。

### 1.2 金字塔策略
为解决 LSM 收敛范围小（仅数个像素）的问题，引入高斯金字塔：
1.  构建 $L$ 层金字塔（下采样率 2）。
2.  从顶层（低分辨率）开始匹配，初始位移为 0。
3.  将第 $i$ 层计算的偏移量 $(\Delta x, \Delta y)$ 放大 2 倍传递给第 $i-1$ 层作为初值。
4.  在第 0 层（原始分辨率）进行最终精化。

## 2. 优化实现

### 2.1 SIMD 指令集优化 (AVX2)
核心计算瓶颈在于：
1.  **双线性插值**: 每次迭代需对整个 ROI 进行重采样。
2.  **法方程组装**: 需累加 $Gx^2, GxGy, Gx \cdot \text{diff}$ 等项。

**优化方案**:
- 使用 `_mm256_loadu_ps` 批量加载 8 个像素。
- 使用 AVX2 FMA 指令 (`_mm256_fmadd_ps`) 加速卷积和累加。
- 对齐内存 (`ALIGNED_ALLOC`) 确保 Load 效率。
- 在 `simd_ops.cpp` 中实现了 `batch_bilinear_interp` 和 `assemble_normal_equations`。

### 2.2 OpenMP 并行
在 `Evaluator` 模块中，对大规模蒙特卡洛仿真（Monte Carlo Simulation）采用 OpenMP 并行化。
```cpp
#pragma omp parallel for
for (int i = 0; i < numTrials; ++i) { ... }
```
显著缩短了评估时间。

## 3. 评估结果

### 3.1 信噪比 (SNR) 鲁棒性
测试在不同 SNR (10dB - 60dB) 下的匹配精度。
- **高 SNR (>=30dB)**: 平均误差接近 0，标准差约为 0.14 像素。
- **低 SNR (<20dB)**: 误差显著增加，但在 20dB 时仍能保持收敛。
*注：当前精度受限于双线性插值的系统误差，若需达到 0.01 像素，建议升级为 Bicubic 插值或 B-Spline 插值。*

### 3.2 Mark 不对称性 (Asymmetry)
测试在不同 Asymmetry (0 - 0.25) 条件下的表现。
- 当 Asymmetry < 0.1 时，平均误差线性增加，但仍能保持亚像素对准。
- LSM 对线性变形具有一定的抵抗力，但非对称边缘会引入系统性偏差 (Bias)。

### 3.3 性能测试
- **无 SIMD**: ~19.4 ms / 次
- **有 SIMD**: ~11.3 ms / 次
- **加速比**: ~1.71x
（注：当前 SIMD 优化主要集中在 Core LSM，金字塔构建和部分内存操作仍有优化空间）

## 4. 结论与展望
本项目成功实现了基于 Pyramid LSM 的晶圆对准引擎，验证了 SIMD 加速的有效性。
**改进方向**:
1.  引入更高阶插值（Bicubic/B-Spline）以消除亚像素偏差，冲击 0.01 精度目标。
2.  改进 Asymmetry 补偿算法，例如引入权重函数忽略边缘异常。
3.  进一步优化 SIMD 流水线，减少非对齐内存访问开销。
