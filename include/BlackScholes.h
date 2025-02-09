#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include <vector>
#include <memory>
#include "ThreadPool.h"

class BlackScholes {
public:
    explicit BlackScholes(size_t numThreads = 1);
    ~BlackScholes() = default;

    // Single option pricing
    double calculateCallPrice(double spot, double strike, double rate, 
                            double volatility, double timeToMaturity) const;
    
    double calculatePutPrice(double spot, double strike, double rate, 
                           double volatility, double timeToMaturity) const;

    // Batch pricing using multithreading
    std::vector<double> batchCalculateCall(
        const std::vector<double>& spots,
        const std::vector<double>& strikes,
        const std::vector<double>& rates,
        const std::vector<double>& volatilities,
        const std::vector<double>& timesToMaturity);

    std::vector<double> batchCalculatePut(
        const std::vector<double>& spots,
        const std::vector<double>& strikes,
        const std::vector<double>& rates,
        const std::vector<double>& volatilities,
        const std::vector<double>& timesToMaturity);

private:
    // Helper functions for the Black-Scholes formula
    static double normalCDF(double x);
    static double d1(double spot, double strike, double rate, 
                    double volatility, double timeToMaturity);
    static double d2(double d1Value, double volatility, double timeToMaturity);

    // Thread pool for parallel computation
    std::unique_ptr<ThreadPool> threadPool;
    
    // Batch size for parallel processing
    static constexpr size_t BATCH_SIZE = 1000;
};

#endif // BLACK_SCHOLES_H 