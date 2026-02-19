#pragma once
#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "pyramid_lsm.h"
#include "mark_simulator.h"
#include <string>
#include <vector>
#include <functional>

/**
 * @brief 单组评估结果
 */
struct EvalResult {
    float snr;                 // 信噪比
    float asymmetry;           // mark 不对称性
    float meanErrorX;          // X方向平均误差
    float meanErrorY;          // Y方向平均误差
    float stdErrorX;           // X方向标准差(重复精度)
    float stdErrorY;           // Y方向标准差
    float maxErrorX;           // X方向最大误差
    float maxErrorY;           // Y方向最大误差
    float rmseX;               // X方向RMSE
    float rmseY;               // Y方向RMSE
    float meanCorrelation;     // 平均相关系数
    double avgTimeMs;          // 平均匹配耗时(ms)
    int numTrials;             // 试验次数
};

/**
 * @brief 加速对比结果
 */
struct SpeedupResult {
    double timeWithoutSIMD;    // 无SIMD耗时(ms)
    double timeWithSIMD;       // 有SIMD耗时(ms)
    double speedup;            // 加速比
    int numTrials;
};

/**
 * @brief 系统评估器
 * 
 * 系统性评估 Pyramid LSM 在不同条件下的性能：
 * - SNR 扫描
 * - Asymmetry 扫描
 * - 重复精度测试
 * - SIMD 加速比测试
 */
class Evaluator {
public:
    using ProgressCallback = std::function<void(int current, int total, const std::string& msg)>;

    Evaluator(int numTrials = 100);

    /**
     * @brief SNR 扫描评估
     */
    std::vector<EvalResult> evaluateSNR(
        const std::vector<float>& snrValues,
        float fixedAsymmetry = 0.0f,
        ProgressCallback progress = nullptr);

    /**
     * @brief Asymmetry 扫描评估
     */
    std::vector<EvalResult> evaluateAsymmetry(
        const std::vector<float>& asymmetryValues,
        float fixedSNR = 30.0f,
        ProgressCallback progress = nullptr);

    /**
     * @brief 综合评估 (SNR x Asymmetry)
     */
    std::vector<EvalResult> evaluateGrid(
        const std::vector<float>& snrValues,
        const std::vector<float>& asymmetryValues,
        ProgressCallback progress = nullptr);

    /**
     * @brief SIMD 加速比评估
     */
    SpeedupResult evaluateSpeedup(float snr = 30.0f, int trials = 50);

    /**
     * @brief 保存结果到 CSV
     */
    static void saveResultsCSV(const std::vector<EvalResult>& results,
                                const std::string& filename);

    /**
     * @brief 保存加速结果
     */
    static void saveSpeedupCSV(const SpeedupResult& result,
                                const std::string& filename);

    /**
     * @brief 打印结果摘要
     */
    static void printSummary(const std::vector<EvalResult>& results);

    void setNumTrials(int n) { numTrials_ = n; }

private:
    int numTrials_;

    EvalResult runSingleEvaluation(const MarkSimConfig& baseConfig, bool useSIMD = true);
};

#endif // EVALUATOR_H
