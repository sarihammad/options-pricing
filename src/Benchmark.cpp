#include "Benchmark.h"
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>

Benchmark::Benchmark(const std::vector<size_t>& threadCounts)
    : threadCounts(threadCounts) {
    // Sort thread counts in ascending order
    std::sort(this->threadCounts.begin(), this->threadCounts.end());
}

std::vector<double> Benchmark::generateRandomData(size_t size, double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    
    std::vector<double> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

double Benchmark::measureExecutionTime(const TimePoint& start, const TimePoint& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<Benchmark::BenchmarkResult> Benchmark::benchmarkBlackScholes(size_t numOptions) {
    std::vector<BenchmarkResult> results;
    
    // Generate random test data
    auto spots = generateRandomData(numOptions, 50.0, 150.0);
    auto strikes = generateRandomData(numOptions, 40.0, 160.0);
    auto rates = generateRandomData(numOptions, 0.01, 0.05);
    auto volatilities = generateRandomData(numOptions, 0.1, 0.5);
    auto timesToMaturity = generateRandomData(numOptions, 0.1, 2.0);
    
    double baselineTime = 0.0;
    
    for (size_t threads : threadCounts) {
        BlackScholes bs(threads);
        
        auto start = Clock::now();
        auto prices = bs.batchCalculateCall(spots, strikes, rates,
                                          volatilities, timesToMaturity);
        auto end = Clock::now();
        
        double executionTime = measureExecutionTime(start, end);
        
        if (threads == threadCounts.front()) {
            baselineTime = executionTime;
        }
        
        results.push_back({
            "Black-Scholes",
            threads,
            numOptions,
            executionTime,
            baselineTime / executionTime
        });
    }
    
    return results;
}

std::vector<Benchmark::BenchmarkResult> Benchmark::benchmarkMonteCarlo(size_t numOptions) {
    std::vector<BenchmarkResult> results;
    
    // Generate random test data
    auto spots = generateRandomData(numOptions, 50.0, 150.0);
    auto strikes = generateRandomData(numOptions, 40.0, 160.0);
    auto rates = generateRandomData(numOptions, 0.01, 0.05);
    auto volatilities = generateRandomData(numOptions, 0.1, 0.5);
    auto timesToMaturity = generateRandomData(numOptions, 0.1, 2.0);
    
    double baselineTime = 0.0;
    
    for (size_t threads : threadCounts) {
        MonteCarlo mc(threads);
        
        auto start = Clock::now();
        auto prices = mc.batchSimulateCall(spots, strikes, rates,
                                         volatilities, timesToMaturity);
        auto end = Clock::now();
        
        double executionTime = measureExecutionTime(start, end);
        
        if (threads == threadCounts.front()) {
            baselineTime = executionTime;
        }
        
        results.push_back({
            "Monte Carlo",
            threads,
            numOptions,
            executionTime,
            baselineTime / executionTime
        });
    }
    
    return results;
}

std::vector<Benchmark::BenchmarkResult> Benchmark::benchmarkBinomialTree(size_t numOptions) {
    std::vector<BenchmarkResult> results;
    
    // Generate random test data
    auto spots = generateRandomData(numOptions, 50.0, 150.0);
    auto strikes = generateRandomData(numOptions, 40.0, 160.0);
    auto rates = generateRandomData(numOptions, 0.01, 0.05);
    auto volatilities = generateRandomData(numOptions, 0.1, 0.5);
    auto timesToMaturity = generateRandomData(numOptions, 0.1, 2.0);
    
    double baselineTime = 0.0;
    
    for (size_t threads : threadCounts) {
        BinomialTree bt(threads);
        
        auto start = Clock::now();
        auto prices = bt.batchCalculateCall(spots, strikes, rates,
                                          volatilities, timesToMaturity);
        auto end = Clock::now();
        
        double executionTime = measureExecutionTime(start, end);
        
        if (threads == threadCounts.front()) {
            baselineTime = executionTime;
        }
        
        results.push_back({
            "Binomial Tree",
            threads,
            numOptions,
            executionTime,
            baselineTime / executionTime
        });
    }
    
    return results;
}

std::vector<Benchmark::BenchmarkResult> Benchmark::runAllBenchmarks(size_t numOptions) {
    std::vector<BenchmarkResult> results;
    
    // Run benchmarks for each model
    auto bsResults = benchmarkBlackScholes(numOptions);
    auto mcResults = benchmarkMonteCarlo(numOptions);
    auto btResults = benchmarkBinomialTree(numOptions);
    
    // Combine results
    results.insert(results.end(), bsResults.begin(), bsResults.end());
    results.insert(results.end(), mcResults.begin(), mcResults.end());
    results.insert(results.end(), btResults.begin(), btResults.end());
    
    return results;
}

void Benchmark::saveResults(const std::string& filename,
                          const std::vector<BenchmarkResult>& results) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    // Write CSV header
    file << "Model,Threads,NumOptions,ExecutionTime(ms),SpeedupFactor\n";
    
    // Write results
    for (const auto& result : results) {
        file << result.modelName << ","
             << result.numThreads << ","
             << result.numOptions << ","
             << std::fixed << std::setprecision(2) << result.executionTimeMs << ","
             << std::fixed << std::setprecision(2) << result.speedupFactor << "\n";
    }
} 