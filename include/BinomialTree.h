#ifndef BINOMIAL_TREE_H
#define BINOMIAL_TREE_H

#include <vector>
#include <memory>
#include "ThreadPool.h"

class BinomialTree {
public:
    explicit BinomialTree(size_t numThreads = 1, size_t defaultSteps = 1000);
    ~BinomialTree() = default;

    // Single option pricing
    double calculateCallPrice(double spot, double strike, double rate,
                            double volatility, double timeToMaturity,
                            size_t numSteps = 0);
    
    double calculatePutPrice(double spot, double strike, double rate,
                           double volatility, double timeToMaturity,
                           size_t numSteps = 0);

    // Batch pricing using multithreading
    std::vector<double> batchCalculateCall(
        const std::vector<double>& spots,
        const std::vector<double>& strikes,
        const std::vector<double>& rates,
        const std::vector<double>& volatilities,
        const std::vector<double>& timesToMaturity,
        size_t numSteps = 0);

    std::vector<double> batchCalculatePut(
        const std::vector<double>& spots,
        const std::vector<double>& strikes,
        const std::vector<double>& rates,
        const std::vector<double>& volatilities,
        const std::vector<double>& timesToMaturity,
        size_t numSteps = 0);

private:
    // Helper functions for tree construction and traversal
    std::vector<double> buildTree(double spot, double rate, double volatility,
                                double timeToMaturity, size_t numSteps);
    
    double traverseTree(const std::vector<double>& tree, double strike,
                       double probability, bool isCall);

    // Thread pool for parallel computation
    std::unique_ptr<ThreadPool> threadPool;
    
    // Default number of time steps if not specified
    size_t defaultNumSteps;
    
    // Batch size for parallel processing
    static constexpr size_t BATCH_SIZE = 100;
};

#endif // BINOMIAL_TREE_H 