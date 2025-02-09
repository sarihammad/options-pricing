#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>
#include <chrono>
#include "BlackScholes.h"
#include "MonteCarlo.h"
#include "BinomialTree.h"

class Benchmark {
public:
    struct BenchmarkResult {
        std::string modelName;
        size_t numThreads;
        size_t numOptions;
        double executionTimeMs;
        double speedupFactor;  // Relative to single-threaded
    };

    explicit Benchmark(const std::vector<size_t>& threadCounts = {1, 2, 4, 8});

    // Run all benchmarks
    std::vector<BenchmarkResult> runAllBenchmarks(size_t numOptions = 10000);

    // Individual model benchmarks
    std::vector<BenchmarkResult> benchmarkBlackScholes(size_t numOptions);
    std::vector<BenchmarkResult> benchmarkMonteCarlo(size_t numOptions);
    std::vector<BenchmarkResult> benchmarkBinomialTree(size_t numOptions);

    // Save results to file
    void saveResults(const std::string& filename,
                    const std::vector<BenchmarkResult>& results);

private:
    // Generate random test data
    std::vector<double> generateRandomData(size_t size, double min, double max);
    
    // Thread configurations to test
    std::vector<size_t> threadCounts;
    
    // Timer utilities
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    
    double measureExecutionTime(const TimePoint& start, const TimePoint& end);
};

#endif // BENCHMARK_H 