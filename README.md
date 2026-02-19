# Pyramid LSM: High-Precision Wafer Alignment Engine
# 面向半导体光刻制程的晶圆对准与套刻误差量测引擎

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![SIMD](https://img.shields.io/badge/SIMD-AVX2-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **核心亮点**: 本项目实现了一个工业级精度的晶圆对准算法，采用 Pyramid Least Squares Matching (LSM) 策略，结合 AVX2/OpenMP 高性能优化。在模拟真实晶圆噪声与畸变的环境下，实现了 **<0.02 像素** 的超高重复精度和 **~11ms** 的高速匹配。

## 核心测试结果 (Results Show)

### 1. 精度与鲁棒性 (Precision & Robustness)
针对“图像越干净精度越差”的经典混叠问题，我们引入了 Pre-smoothing 策略，实现了全信噪比范围的高精度覆盖。

| 环境条件 (SNR) | 优化前精度 (StdDev) | **优化后精度 (StdDev)** | 提升幅度 |
| :--- | :--- | :--- | :--- |
| **60dB (Clean)** | 0.150 px | **0.018 px** | **8.3x ** |
| 30dB (Normal) | 0.137 px | **0.018 px** | **7.6x ** |
| 10dB (Noisy) | 0.027 px | **0.020 px** | 1.35x |

> **结论**: 系统在所有噪声水平下均能保持 **<0.02 像素** (3σ < 0.06 px) 的重复精度，完全满足半导体前道制程的对准需求。

### 2. 性能加速 (Performance)
利用 AVX2 指令集对双线性插值、法方程组装等核心算子进行向量化，并结合 OpenMP 多线程技术。

| 模式 | 平均耗时 (ms) | Speedup |
| :--- | :--- | :--- |
| Scala (无SIMD) | 18.90 ms | 1.0x |
| **SIMD (AVX2)** | **10.94 ms** | **1.72x** |

### 3. 不对称性分析 (Asymmetry Sensitivity)
针对工艺导致的 Mark 不对称 (Asymmetry) 问题进行定量分析：
*   **线性偏差**: 匹配位置随 Asymmetry (0~0.25) 呈线性偏移，符合物理预期。
*   **高稳定性**: 即便存在 Asymmetry，匹配的重复精度 (StdDev) 仍保持在 **0.018 px**。这意味着系统误差是稳定的，可以通过 Calibration 轻松校准。

---

##  项目简介 (Introduction)

本项目基于 C++17 实现，不依赖 OpenCV 等第三方重型库，是一个轻量级、高性能的算法核心。

### 主要功能
1.  **Pyramid LSM 算法**: 4层高斯金字塔，由粗到精 (Coarse-to-Fine) 搜索，兼顾大范围捕获与高精度收敛。
2.  **6参数仿射模型**: 能够校正缩放、旋转、剪切 (Shear) 等复杂几何畸变。
3.  **工业级仿真器**: 内置 `MarkSimulator`，支持生成含 Box-in-Box/Cross Mark、应力变形 (Stress)、非对称 (Asymmetry) 及混合噪声 (Gaussian/Poisson) 的仿真数据。
4.  **评估框架**: 自动化测试 SNR 鲁棒性、Asymmetry 敏感度及 SIMD 加速比。

##  编译与运行 (Build & Run)

### 环境要求
*   Windows (MSVC 2019+) 或 Linux (GCC 7+)
*   CMake 3.10+
*   支持 AVX2 的 CPU

### 一键运行 (Windows)
我们提供了批处理脚本，会自动配置 MSVC 环境并编译运行：
```cmd
./build.bat
```

### 手动编译 (CMake)
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
./PyramidLSM
```
##  目录结构
*   `include/`: 核心算法头文件
*   `src/`: 源代码实现 (LSM Matcher, Pyramid, Simulator, AVX2 Ops)
*   `eval_*.csv`: 自动生成的评估报告

##  技术报告
详细的算法原理、优化策略及实验分析请参阅 [REPORT.md](./REPORT.md) 和 [结果分析_优化后.md](./结果分析_优化后.md)。
