#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <vector>
#include <memory>
#include <random>
#include "ThreadPool.h"

class MonteCarlo {
public:
    explicit MonteCarlo(size_t numThreads = 1, size_t defaultSimulations = 100000);
    ~MonteCarlo() = default;

    // Single option pricing
    double simulateCallPrice(double spot, double strike, double rate,
                           double volatility, double timeToMaturity,
                           size_t numSimulations = 0);
    
    double simulatePutPrice(double spot, double strike, double rate,
                          double volatility, double timeToMaturity,
                          size_t numSimulations = 0);

    // Batch pricing using multithreading
    std::vector<double> batchSimulateCall(
        const std::vector<double>& spots,
        const std::vector<double>& strikes,
        const std::vector<double>& rates,
        const std::vector<double>& volatilities,
        const std::vector<double>& timesToMaturity,
        size_t numSimulations = 0);

    std::vector<double> batchSimulatePut(
        const std::vector<double>& spots,
        const std::vector<double>& strikes,
        const std::vector<double>& rates,
        const std::vector<double>& volatilities,
        const std::vector<double>& timesToMaturity,
        size_t numSimulations = 0);

private:
    // Helper functions for simulation
    double simulatePath(double spot, double strike, double rate,
                       double volatility, double timeToMaturity,
                       std::mt19937& gen, bool isCall);
    
    std::vector<double> simulatePathBatch(
        const std::vector<double>& spots,
        const std::vector<double>& strikes,
        const std::vector<double>& rates,
        const std::vector<double>& volatilities,
        const std::vector<double>& timesToMaturity,
        size_t startIdx, size_t endIdx,
        size_t numSimulations, bool isCall);

    // Thread pool for parallel computation
    std::unique_ptr<ThreadPool> threadPool;
    
    // Default number of simulations if not specified
    size_t defaultNumSimulations;
    
    // Batch size for parallel processing
    static constexpr size_t BATCH_SIZE = 100;
    
    // Random number generation
    static thread_local std::mt19937 generator;
    static thread_local std::normal_distribution<double> normalDist;
};

#endif // MONTE_CARLO_H 