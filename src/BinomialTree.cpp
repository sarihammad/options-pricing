#include "BinomialTree.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <future>

BinomialTree::BinomialTree(size_t numThreads, size_t defaultSteps)
    : threadPool(std::make_unique<ThreadPool>(numThreads)),
      defaultNumSteps(defaultSteps) {}

std::vector<double> BinomialTree::buildTree(double spot, double rate,
                                          double volatility, double timeToMaturity,
                                          size_t numSteps) {
    double dt = timeToMaturity / numSteps;
    double up = std::exp(volatility * std::sqrt(dt));
    double down = 1.0 / up;
    
    // Pre-allocate the entire tree
    std::vector<double> tree((numSteps + 1) * (numSteps + 2) / 2);
    
    // Initialize the root
    tree[0] = spot;
    
    // Build the tree level by level
    size_t levelStart = 0;
    for (size_t step = 1; step <= numSteps; ++step) {
        size_t prevLevelStart = levelStart;
        levelStart += step;
        
        // First node in level comes from previous first node
        tree[levelStart] = tree[prevLevelStart] * down;
        
        // Interior nodes
        for (size_t j = 1; j <= step; ++j) {
            tree[levelStart + j] = tree[prevLevelStart + j - 1] * up;
        }
    }
    
    return tree;
}

double BinomialTree::traverseTree(const std::vector<double>& tree, double strike,
                                double probability, bool isCall) {
    size_t numSteps = static_cast<size_t>(std::sqrt(2 * tree.size()) - 1);
    std::vector<double> values(numSteps + 1);
    
    // Initialize terminal values
    size_t levelStart = tree.size() - numSteps - 1;
    for (size_t i = 0; i <= numSteps; ++i) {
        double spotPrice = tree[levelStart + i];
        values[i] = isCall ? std::max(spotPrice - strike, 0.0)
                          : std::max(strike - spotPrice, 0.0);
    }
    
    // Backward induction
    double oneMinusProb = 1.0 - probability;
    for (size_t step = numSteps; step > 0; --step) {
        for (size_t i = 0; i < step; ++i) {
            values[i] = probability * values[i + 1] + oneMinusProb * values[i];
        }
    }
    
    return values[0];
}

double BinomialTree::calculateCallPrice(double spot, double strike, double rate,
                                      double volatility, double timeToMaturity,
                                      size_t numSteps) {
    if (timeToMaturity <= 0.0) {
        throw std::invalid_argument("Time to maturity must be positive");
    }
    if (volatility <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }
    if (spot <= 0.0 || strike <= 0.0) {
        throw std::invalid_argument("Spot and strike prices must be positive");
    }

    size_t actualSteps = numSteps > 0 ? numSteps : defaultNumSteps;
    double dt = timeToMaturity / actualSteps;
    double up = std::exp(volatility * std::sqrt(dt));
    double probability = (std::exp(rate * dt) - 1.0/up) / (up - 1.0/up);
    
    auto tree = buildTree(spot, rate, volatility, timeToMaturity, actualSteps);
    double price = traverseTree(tree, strike, probability, true);
    
    return std::exp(-rate * timeToMaturity) * price;
}

double BinomialTree::calculatePutPrice(double spot, double strike, double rate,
                                     double volatility, double timeToMaturity,
                                     size_t numSteps) {
    if (timeToMaturity <= 0.0) {
        throw std::invalid_argument("Time to maturity must be positive");
    }
    if (volatility <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }
    if (spot <= 0.0 || strike <= 0.0) {
        throw std::invalid_argument("Spot and strike prices must be positive");
    }

    size_t actualSteps = numSteps > 0 ? numSteps : defaultNumSteps;
    double dt = timeToMaturity / actualSteps;
    double up = std::exp(volatility * std::sqrt(dt));
    double probability = (std::exp(rate * dt) - 1.0/up) / (up - 1.0/up);
    
    auto tree = buildTree(spot, rate, volatility, timeToMaturity, actualSteps);
    double price = traverseTree(tree, strike, probability, false);
    
    return std::exp(-rate * timeToMaturity) * price;
}

std::vector<double> BinomialTree::batchCalculateCall(
    const std::vector<double>& spots,
    const std::vector<double>& strikes,
    const std::vector<double>& rates,
    const std::vector<double>& volatilities,
    const std::vector<double>& timesToMaturity,
    size_t numSteps) {
    
    size_t numOptions = spots.size();
    if (numOptions != strikes.size() || numOptions != rates.size() ||
        numOptions != volatilities.size() || numOptions != timesToMaturity.size()) {
        throw std::invalid_argument("All input vectors must have the same size");
    }

    size_t actualSteps = numSteps > 0 ? numSteps : defaultNumSteps;
    std::vector<double> results(numOptions);
    std::vector<std::future<void>> futures;

    // Split work into batches
    for (size_t i = 0; i < numOptions; i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, numOptions);
        futures.push_back(
            threadPool->submit([this, i, batchEnd, &spots, &strikes, &rates,
                              &volatilities, &timesToMaturity, actualSteps, &results]() {
                for (size_t j = i; j < batchEnd; ++j) {
                    results[j] = calculateCallPrice(spots[j], strikes[j], rates[j],
                                                 volatilities[j], timesToMaturity[j],
                                                 actualSteps);
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

std::vector<double> BinomialTree::batchCalculatePut(
    const std::vector<double>& spots,
    const std::vector<double>& strikes,
    const std::vector<double>& rates,
    const std::vector<double>& volatilities,
    const std::vector<double>& timesToMaturity,
    size_t numSteps) {
    
    size_t numOptions = spots.size();
    if (numOptions != strikes.size() || numOptions != rates.size() ||
        numOptions != volatilities.size() || numOptions != timesToMaturity.size()) {
        throw std::invalid_argument("All input vectors must have the same size");
    }

    size_t actualSteps = numSteps > 0 ? numSteps : defaultNumSteps;
    std::vector<double> results(numOptions);
    std::vector<std::future<void>> futures;

    // Split work into batches
    for (size_t i = 0; i < numOptions; i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, numOptions);
        futures.push_back(
            threadPool->submit([this, i, batchEnd, &spots, &strikes, &rates,
                              &volatilities, &timesToMaturity, actualSteps, &results]() {
                for (size_t j = i; j < batchEnd; ++j) {
                    results[j] = calculatePutPrice(spots[j], strikes[j], rates[j],
                                                volatilities[j], timesToMaturity[j],
                                                actualSteps);
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