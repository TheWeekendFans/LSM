#include "evaluator.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "Starting Pyramid LSM Evaluation..." << std::endl;
    
    // 1. 设置评估参数
    Evaluator evaluator;
    evaluator.setNumTrials(1000); // 1000 trials for robustness

    // Output directory
    std::string outDir = "./";

    // 2. 评估 SNR 鲁棒性
    std::cout << "\n[Test 1] SNR Robustness Evaluation" << std::endl;
    std::vector<float> snrValues = {10, 15, 20, 25, 30, 40, 50, 60};
    auto snrResults = evaluator.evaluateSNR(snrValues, 0.0f, 
        [](int curr, int total, const std::string& msg) {
            if (curr % 100 == 0) std::cout << "\r" << msg << " [" << curr << "/" << total << "]" << std::flush;
        }
    );
    std::cout << "\nSNR Evaluation Done." << std::endl;
    Evaluator::printSummary(snrResults);
    Evaluator::saveResultsCSV(snrResults, outDir + "eval_snr.csv");

    // 3. 评估 Asymmetry 鲁棒性 (Fixed SNR = 30dB)
    std::cout << "\n[Test 2] Mark Asymmetry Robustness Evaluation (SNR=30dB)" << std::endl;
    std::vector<float> asymValues = {0.0f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f};
    auto asymResults = evaluator.evaluateAsymmetry(asymValues, 30.0f,
         [](int curr, int total, const std::string& msg) {
            if (curr % 100 == 0) std::cout << "\r" << msg << " [" << curr << "/" << total << "]" << std::flush;
        }
    );
    std::cout << "\nAsymmetry Evaluation Done." << std::endl;
    Evaluator::printSummary(asymResults);
    Evaluator::saveResultsCSV(asymResults, outDir + "eval_asymmetry.csv");

    // 4. SIMD 加速比测试
    std::cout << "\n[Test 3] SIMD Performance Benchmarking" << std::endl;
    auto speedup = evaluator.evaluateSpeedup(30.0f, 200);
    std::cout << "Time No SIMD: " << speedup.timeWithoutSIMD << " ms" << std::endl;
    std::cout << "Time SIMD:    " << speedup.timeWithSIMD << " ms" << std::endl;
    std::cout << "Speedup:      " << speedup.speedup << "x" << std::endl;
    Evaluator::saveSpeedupCSV(speedup, outDir + "perf_speedup.csv");

    std::cout << "\nAll evaluations completed. Results saved to local directory." << std::endl;
    return 0;
}
