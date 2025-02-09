#include "BlackScholes.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// Constants for numerical calculations
constexpr double SQRT_2PI = 2.506628274631;
constexpr double SQRT_2 = 1.4142135623731;

BlackScholes::BlackScholes(size_t numThreads) 
    : threadPool(std::make_unique<ThreadPool>(numThreads)) {}

double BlackScholes::normalCDF(double x) {
    // Approximation of the cumulative distribution function
    // Using Abramowitz and Stegun approximation
    if (x < 0) {
        return 1.0 - normalCDF(-x);
    }
    
    double k = 1.0 / (1.0 + 0.2316419 * x);
    double poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + 
                  k * (-1.821255978 + k * 1.330274429))));
                  
    return 1.0 - (1.0 / SQRT_2PI) * std::exp(-0.5 * x * x) * poly;
}

double BlackScholes::d1(double spot, double strike, double rate, 
                       double volatility, double timeToMaturity) {
    if (timeToMaturity <= 0.0) {
        throw std::invalid_argument("Time to maturity must be positive");
    }
    if (volatility <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }
    if (spot <= 0.0 || strike <= 0.0) {
        throw std::invalid_argument("Spot and strike prices must be positive");
    }

    double volSqrtT = volatility * std::sqrt(timeToMaturity);
    return (std::log(spot/strike) + (rate + 0.5 * volatility * volatility) * timeToMaturity) 
           / volSqrtT;
}

double BlackScholes::d2(double d1Value, double volatility, double timeToMaturity) {
    return d1Value - volatility * std::sqrt(timeToMaturity);
}

double BlackScholes::calculateCallPrice(double spot, double strike, double rate,
                                      double volatility, double timeToMaturity) const {
    double d1Value = d1(spot, strike, rate, volatility, timeToMaturity);
    double d2Value = d2(d1Value, volatility, timeToMaturity);
    
    return spot * normalCDF(d1Value) - 
           strike * std::exp(-rate * timeToMaturity) * normalCDF(d2Value);
}

double BlackScholes::calculatePutPrice(double spot, double strike, double rate,
                                     double volatility, double timeToMaturity) const {
    double d1Value = d1(spot, strike, rate, volatility, timeToMaturity);
    double d2Value = d2(d1Value, volatility, timeToMaturity);
    
    return strike * std::exp(-rate * timeToMaturity) * normalCDF(-d2Value) - 
           spot * normalCDF(-d1Value);
}

std::vector<double> BlackScholes::batchCalculateCall(
    const std::vector<double>& spots,
    const std::vector<double>& strikes,
    const std::vector<double>& rates,
    const std::vector<double>& volatilities,
    const std::vector<double>& timesToMaturity) {
    
    size_t numOptions = spots.size();
    if (numOptions != strikes.size() || numOptions != rates.size() ||
        numOptions != volatilities.size() || numOptions != timesToMaturity.size()) {
        throw std::invalid_argument("All input vectors must have the same size");
    }

    std::vector<double> results(numOptions);
    std::vector<std::future<void>> futures;

    // Split work into batches
    for (size_t i = 0; i < numOptions; i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, numOptions);
        futures.push_back(
            threadPool->submit([this, i, batchEnd, &spots, &strikes, &rates, 
                              &volatilities, &timesToMaturity, &results]() {
                for (size_t j = i; j < batchEnd; ++j) {
                    results[j] = calculateCallPrice(spots[j], strikes[j], rates[j],
                                                 volatilities[j], timesToMaturity[j]);
                }
            })
        );
    }

    // Wait for all batches to complete
    for (auto& future : futures) {
        future.get();
    }

    return results;
}

std::vector<double> BlackScholes::batchCalculatePut(
    const std::vector<double>& spots,
    const std::vector<double>& strikes,
    const std::vector<double>& rates,
    const std::vector<double>& volatilities,
    const std::vector<double>& timesToMaturity) {
    
    size_t numOptions = spots.size();
    if (numOptions != strikes.size() || numOptions != rates.size() ||
        numOptions != volatilities.size() || numOptions != timesToMaturity.size()) {
        throw std::invalid_argument("All input vectors must have the same size");
    }

    std::vector<double> results(numOptions);
    std::vector<std::future<void>> futures;

    // Split work into batches
    for (size_t i = 0; i < numOptions; i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, numOptions);
        futures.push_back(
            threadPool->submit([this, i, batchEnd, &spots, &strikes, &rates, 
                              &volatilities, &timesToMaturity, &results]() {
                for (size_t j = i; j < batchEnd; ++j) {
                    results[j] = calculatePutPrice(spots[j], strikes[j], rates[j],
                                                volatilities[j], timesToMaturity[j]);
                }
            })
        );
    }

    // Wait for all batches to complete
    for (auto& future : futures) {
        future.get();
    }

    return results;
} 