#include "MonteCarlo.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <chrono>

// Initialize thread-local random number generators
thread_local std::mt19937 MonteCarlo::generator(
    std::chrono::system_clock::now().time_since_epoch().count());
thread_local std::normal_distribution<double> MonteCarlo::normalDist(0.0, 1.0);

MonteCarlo::MonteCarlo(size_t numThreads, size_t defaultSimulations)
    : threadPool(std::make_unique<ThreadPool>(numThreads)),
      defaultNumSimulations(defaultSimulations) {}

double MonteCarlo::simulatePath(double spot, double strike, double rate,
                              double volatility, double timeToMaturity,
                              std::mt19937& gen, bool isCall) {
    double drift = (rate - 0.5 * volatility * volatility) * timeToMaturity;
    double diffusion = volatility * std::sqrt(timeToMaturity) * normalDist(gen);
    double finalSpot = spot * std::exp(drift + diffusion);
    
    if (isCall) {
        return std::max(finalSpot - strike, 0.0);
    } else {
        return std::max(strike - finalSpot, 0.0);
    }
}

double MonteCarlo::simulateCallPrice(double spot, double strike, double rate,
                                   double volatility, double timeToMaturity,
                                   size_t numSimulations) {
    if (timeToMaturity <= 0.0) {
        throw std::invalid_argument("Time to maturity must be positive");
    }
    if (volatility <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }
    if (spot <= 0.0 || strike <= 0.0) {
        throw std::invalid_argument("Spot and strike prices must be positive");
    }

    size_t actualSimulations = numSimulations > 0 ? numSimulations : defaultNumSimulations;
    double sumPayoffs = 0.0;
    
    for (size_t i = 0; i < actualSimulations; ++i) {
        sumPayoffs += simulatePath(spot, strike, rate, volatility, timeToMaturity,
                                 generator, true);
    }
    
    return std::exp(-rate * timeToMaturity) * (sumPayoffs / actualSimulations);
}

double MonteCarlo::simulatePutPrice(double spot, double strike, double rate,
                                  double volatility, double timeToMaturity,
                                  size_t numSimulations) {
    if (timeToMaturity <= 0.0) {
        throw std::invalid_argument("Time to maturity must be positive");
    }
    if (volatility <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }
    if (spot <= 0.0 || strike <= 0.0) {
        throw std::invalid_argument("Spot and strike prices must be positive");
    }

    size_t actualSimulations = numSimulations > 0 ? numSimulations : defaultNumSimulations;
    double sumPayoffs = 0.0;
    
    for (size_t i = 0; i < actualSimulations; ++i) {
        sumPayoffs += simulatePath(spot, strike, rate, volatility, timeToMaturity,
                                 generator, false);
    }
    
    return std::exp(-rate * timeToMaturity) * (sumPayoffs / actualSimulations);
}

std::vector<double> MonteCarlo::simulatePathBatch(
    const std::vector<double>& spots,
    const std::vector<double>& strikes,
    const std::vector<double>& rates,
    const std::vector<double>& volatilities,
    const std::vector<double>& timesToMaturity,
    size_t startIdx, size_t endIdx,
    size_t numSimulations, bool isCall) {
    
    std::vector<double> results(endIdx - startIdx);
    std::mt19937 localGen(generator());  // Create a local copy of the generator
    
    for (size_t i = startIdx; i < endIdx; ++i) {
        double sumPayoffs = 0.0;
        for (size_t sim = 0; sim < numSimulations; ++sim) {
            sumPayoffs += simulatePath(spots[i], strikes[i], rates[i],
                                     volatilities[i], timesToMaturity[i],
                                     localGen, isCall);
        }
        results[i - startIdx] = std::exp(-rates[i] * timesToMaturity[i]) * 
                               (sumPayoffs / numSimulations);
    }
    
    return results;
}

std::vector<double> MonteCarlo::batchSimulateCall(
    const std::vector<double>& spots,
    const std::vector<double>& strikes,
    const std::vector<double>& rates,
    const std::vector<double>& volatilities,
    const std::vector<double>& timesToMaturity,
    size_t numSimulations) {
    
    size_t numOptions = spots.size();
    if (numOptions != strikes.size() || numOptions != rates.size() ||
        numOptions != volatilities.size() || numOptions != timesToMaturity.size()) {
        throw std::invalid_argument("All input vectors must have the same size");
    }

    size_t actualSimulations = numSimulations > 0 ? numSimulations : defaultNumSimulations;
    std::vector<double> results(numOptions);
    std::vector<std::future<std::vector<double>>> futures;

    // Split work into batches
    for (size_t i = 0; i < numOptions; i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, numOptions);
        futures.push_back(
            threadPool->submit(&MonteCarlo::simulatePathBatch, this,
                             std::ref(spots), std::ref(strikes), std::ref(rates),
                             std::ref(volatilities), std::ref(timesToMaturity),
                             i, batchEnd, actualSimulations, true)
        );
    }

    // Collect results
    size_t currentIdx = 0;
    for (auto& future : futures) {
        auto batchResults = future.get();
        std::copy(batchResults.begin(), batchResults.end(),
                 results.begin() + currentIdx);
        currentIdx += batchResults.size();
    }

    return results;
}

std::vector<double> MonteCarlo::batchSimulatePut(
    const std::vector<double>& spots,
    const std::vector<double>& strikes,
    const std::vector<double>& rates,
    const std::vector<double>& volatilities,
    const std::vector<double>& timesToMaturity,
    size_t numSimulations) {
    
    size_t numOptions = spots.size();
    if (numOptions != strikes.size() || numOptions != rates.size() ||
        numOptions != volatilities.size() || numOptions != timesToMaturity.size()) {
        throw std::invalid_argument("All input vectors must have the same size");
    }

    size_t actualSimulations = numSimulations > 0 ? numSimulations : defaultNumSimulations;
    std::vector<double> results(numOptions);
    std::vector<std::future<std::vector<double>>> futures;

    // Split work into batches
    for (size_t i = 0; i < numOptions; i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, numOptions);
        futures.push_back(
            threadPool->submit(&MonteCarlo::simulatePathBatch, this,
                             std::ref(spots), std::ref(strikes), std::ref(rates),
                             std::ref(volatilities), std::ref(timesToMaturity),
                             i, batchEnd, actualSimulations, false)
        );
    }

    // Collect results
    size_t currentIdx = 0;
    for (auto& future : futures) {
        auto batchResults = future.get();
        std::copy(batchResults.begin(), batchResults.end(),
                 results.begin() + currentIdx);
        currentIdx += batchResults.size();
    }

    return results;
} 