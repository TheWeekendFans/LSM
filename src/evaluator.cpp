#include "evaluator.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <iomanip>

Evaluator::Evaluator(int numTrials) : numTrials_(numTrials) {}

EvalResult Evaluator::runSingleEvaluation(const MarkSimConfig& baseConfig, bool useSIMD) {
    MarkSimulator sim;
    PyramidLSM lsm(4, 30, useSIMD);

    // 随机生成真实偏移 -0.5 ~ 0.5 像素
    // 为了更好地测试亚像素精度，我们在 (-2, 2) 范围内随机
    // MarkSimConfig 中已经有 trueX, trueY，这里我们在 loop 外面生成随机配置
    // 但 Evaluator 的接口是批量的，所以我们在 evaluateX 中循环生成不同的配置

    // 这里 runSingle 需要明确的 config
    auto pair = sim.generatePair(baseConfig);
    
    PyramidLSMResult result = lsm.match(pair.first, pair.second);
    
    EvalResult res;
    res.snr = baseConfig.snr;
    res.asymmetry = baseConfig.asymmetry;
    res.meanErrorX = result.dx - baseConfig.trueOffsetX;
    res.meanErrorY = result.dy - baseConfig.trueOffsetY;
    res.stdErrorX = 0; // 单次没有std
    res.stdErrorY = 0;
    res.maxErrorX = std::fabs(res.meanErrorX);
    res.maxErrorY = std::fabs(res.meanErrorY);
    res.rmseX = res.meanErrorX * res.meanErrorX;
    res.rmseY = res.meanErrorY * res.meanErrorY;
    res.meanCorrelation = result.correlation;
    res.avgTimeMs = result.elapsedMs;
    res.numTrials = 1;

    return res;
}

std::vector<EvalResult> Evaluator::evaluateSNR(const std::vector<float>& snrValues,
                                               float fixedAsymmetry,
                                               ProgressCallback progress) {
    std::vector<EvalResult> results;
    int totalSteps = static_cast<int>(snrValues.size()) * numTrials_;
    int processed = 0;

    for (float snr : snrValues) {
        EvalResult batchRes = {};
        batchRes.snr = snr;
        batchRes.asymmetry = fixedAsymmetry;
        batchRes.numTrials = numTrials_;

        std::vector<double> errorsX, errorsY, times;

        // 并行测试 multiple trials
        // 注意：随机数生成器需要线程安全或者独立
        // MarkSimulator 内部有自己的 rng，每次重新构造或者 setSeed
        // 为了随机性，我们需要不同的种子
        
        #pragma omp parallel for
        for (int i = 0; i < numTrials_; ++i) {
            MarkSimConfig config;
            config.snr = snr;
            config.asymmetry = fixedAsymmetry;
            config.seed = 1000 + i; // 确保每次不同
            
            // 随机位移
            srand(config.seed);
            config.trueOffsetX = (float)(rand() % 1000 - 500) / 500.0f; // -1.0 ~ 1.0
            config.trueOffsetY = (float)(rand() % 1000 - 500) / 500.0f;

            MarkSimulator localSim; // thread local
            PyramidLSM localLSM(4, 30, true);

            auto pair = localSim.generatePair(config);
            PyramidLSMResult res = localLSM.match(pair.first, pair.second);

            double errX = res.dx - config.trueOffsetX;
            double errY = res.dy - config.trueOffsetY;

            #pragma omp critical
            {
                errorsX.push_back(errX);
                errorsY.push_back(errY);
                times.push_back(res.elapsedMs);
                batchRes.meanCorrelation += res.correlation; // accumulation
                processed++;
                if (progress && processed % 10 == 0) {
                    progress(processed, totalSteps, "Evaluating SNR=" + std::to_string(snr));
                }
            }
        }

        // 统计
        double sumErrX = 0, sumErrY = 0, sumSqErrX = 0, sumSqErrY = 0, sumTime = 0;
        double maxErrX = 0, maxErrY = 0;

        for (size_t i = 0; i < errorsX.size(); ++i) {
            double ex = errorsX[i];
            double ey = errorsY[i];
            sumErrX += ex;
            sumErrY += ey;
            sumSqErrX += ex * ex;
            sumSqErrY += ey * ey;
            sumTime += times[i];
            maxErrX = std::max(maxErrX, std::fabs(ex));
            maxErrY = std::max(maxErrY, std::fabs(ey));
        }

        batchRes.meanErrorX = static_cast<float>(sumErrX / numTrials_);
        batchRes.meanErrorY = static_cast<float>(sumErrY / numTrials_);
        batchRes.avgTimeMs = sumTime / numTrials_;
        batchRes.maxErrorX = static_cast<float>(maxErrX);
        batchRes.maxErrorY = static_cast<float>(maxErrY);

        // StdDev = sqrt(E[x^2] - (E[x])^2)
        double mseX = sumSqErrX / numTrials_;
        double mseY = sumSqErrY / numTrials_;
        batchRes.stdErrorX = static_cast<float>(std::sqrt(mseX - batchRes.meanErrorX * batchRes.meanErrorX));
        batchRes.stdErrorY = static_cast<float>(std::sqrt(mseY - batchRes.meanErrorY * batchRes.meanErrorY));
        batchRes.rmseX = static_cast<float>(std::sqrt(mseX));
        batchRes.rmseY = static_cast<float>(std::sqrt(mseY));
        batchRes.meanCorrelation /= numTrials_;

        results.push_back(batchRes);
    }

    return results;
}

std::vector<EvalResult> Evaluator::evaluateAsymmetry(const std::vector<float>& asymValues,
                                                     float fixedSNR,
                                                     ProgressCallback progress) {
    std::vector<EvalResult> results;
    int totalSteps = static_cast<int>(asymValues.size()) * numTrials_;
    int processed = 0;

    for (float asym : asymValues) {
        EvalResult batchRes = {};
        batchRes.snr = fixedSNR;
        batchRes.asymmetry = asym;
        batchRes.numTrials = numTrials_;

        std::vector<double> errorsX, errorsY;

        #pragma omp parallel for
        for (int i = 0; i < numTrials_; ++i) {
            MarkSimConfig config;
            config.snr = fixedSNR;
            config.asymmetry = asym;
            config.seed = 2000 + i;
            
            srand(config.seed);
            config.trueOffsetX = (float)(rand() % 1000 - 500) / 500.0f;
            config.trueOffsetY = (float)(rand() % 1000 - 500) / 500.0f;

            MarkSimulator localSim;
            PyramidLSM localLSM(4, 30, true);

            auto pair = localSim.generatePair(config);
            PyramidLSMResult res = localLSM.match(pair.first, pair.second);

            #pragma omp critical
            {
                errorsX.push_back(res.dx - config.trueOffsetX);
                errorsY.push_back(res.dy - config.trueOffsetY);
                processed++;
                if (progress && processed % 10 == 0) progress(processed, totalSteps, "Evaluating Asym=" + std::to_string(asym));
            }
        }

        double sumErrX = 0, sumSqErrX = 0;
        for (double e : errorsX) { sumErrX += e; sumSqErrX += e * e; }
        
        batchRes.meanErrorX = static_cast<float>(sumErrX / numTrials_);
        batchRes.stdErrorX = static_cast<float>(std::sqrt(sumSqErrX / numTrials_ - batchRes.meanErrorX * batchRes.meanErrorX));
        // Fill others...
        results.push_back(batchRes);
    }
    return results;
}

SpeedupResult Evaluator::evaluateSpeedup(float snr, int trials) {
    SpeedupResult res;
    res.numTrials = trials;
    
    MarkSimConfig config;
    config.snr = snr;
    
    // Warmup
    {
        MarkSimulator sim;
        PyramidLSM lsm(4, 30, true);
        auto pair = sim.generatePair(config);
        lsm.match(pair.first, pair.second);
    }

    // SIMD
    double totalTimeSIMD = 0;
    for(int i=0; i<trials; ++i) {
        MarkSimulator sim;
        config.seed = i;
        config.trueOffsetX = 0.5f; 
        auto pair = sim.generatePair(config);
        PyramidLSM lsm(4, 30, true);
        PyramidLSMResult r = lsm.match(pair.first, pair.second);
        totalTimeSIMD += r.elapsedMs;
    }
    res.timeWithSIMD = totalTimeSIMD / trials;

    // No SIMD
    double totalTimeNoSIMD = 0;
    for(int i=0; i<trials; ++i) {
        MarkSimulator sim;
        config.seed = i;
        config.trueOffsetX = 0.5f;
        auto pair = sim.generatePair(config);
        PyramidLSM lsm(4, 30, false);
        PyramidLSMResult r = lsm.match(pair.first, pair.second);
        totalTimeNoSIMD += r.elapsedMs;
    }
    res.timeWithoutSIMD = totalTimeNoSIMD / trials;
    
    res.speedup = res.timeWithoutSIMD / res.timeWithSIMD;
    return res;
}

void Evaluator::saveResultsCSV(const std::vector<EvalResult>& results, const std::string& filename) {
    std::ofstream ofs(filename);
    ofs << "SNR,Asymmetry,MeanErrorX,MeanErrorY,StdErrorX,StdErrorY,RMSE_X,RMSE_Y,MaxErrorX,MaxErrorY,AvgTimeMs\n";
    for (const auto& r : results) {
        ofs << r.snr << "," << r.asymmetry << ","
            << r.meanErrorX << "," << r.meanErrorY << ","
            << r.stdErrorX << "," << r.stdErrorY << ","
            << r.rmseX << "," << r.rmseY << ","
            << r.maxErrorX << "," << r.maxErrorY << ","
            << r.avgTimeMs << "\n";
    }
}

void Evaluator::saveSpeedupCSV(const SpeedupResult& result, const std::string& filename) {
    std::ofstream ofs(filename);
    ofs << "TimeNoSIMD_ms,TimeSIMD_ms,Speedup,Trials\n";
    ofs << result.timeWithoutSIMD << ","
        << result.timeWithSIMD << ","
        << result.speedup << ","
        << result.numTrials << "\n";
}

void Evaluator::printSummary(const std::vector<EvalResult>& results) {

    std::cout << "\n=== Evaluation Summary ===\n";
    std::cout << std::left << std::setw(10) << "SNR" 
              << std::setw(10) << "Asym" 
              << std::setw(12) << "MeanErrX" 
              << std::setw(12) << "StdErrX" 
              << std::setw(12) << "RMSE_X" << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(10) << r.snr 
                  << std::setw(10) << r.asymmetry 
                  << std::setw(12) << r.meanErrorX 
                  << std::setw(12) << r.stdErrorX 
                  << std::setw(12) << r.rmseX << "\n";
    }
    std::cout << "==========================\n";
}
