#include <iostream>
#include <iomanip>
#include <thread>
#include "BlackScholes.h"
#include "MonteCarlo.h"
#include "BinomialTree.h"
#include "Benchmark.h"

void printOptionPrice(const std::string& modelName, const std::string& optionType,
                     double price) {
    std::cout << std::setw(15) << std::left << modelName
              << std::setw(10) << std::left << optionType
              << "Price: " << std::fixed << std::setprecision(4) << price
              << std::endl;
}

void runSingleOptionExample() {
    // Example option parameters
    double spot = 100.0;        // Current stock price
    double strike = 100.0;      // Strike price
    double rate = 0.05;         // Risk-free rate
    double volatility = 0.2;    // Volatility
    double timeToMaturity = 1.0; // Time to maturity in years

    std::cout << "\nSingle Option Pricing Example:\n"
              << "--------------------------------\n"
              << "Spot Price: " << spot << "\n"
              << "Strike Price: " << strike << "\n"
              << "Risk-free Rate: " << rate << "\n"
              << "Volatility: " << volatility << "\n"
              << "Time to Maturity: " << timeToMaturity << " years\n\n";

    // Black-Scholes pricing
    BlackScholes bs(1);
    double bsCall = bs.calculateCallPrice(spot, strike, rate, volatility, timeToMaturity);
    double bsPut = bs.calculatePutPrice(spot, strike, rate, volatility, timeToMaturity);
    
    printOptionPrice("Black-Scholes", "Call", bsCall);
    printOptionPrice("Black-Scholes", "Put", bsPut);

    // Monte Carlo simulation
    MonteCarlo mc(1, 100000);
    double mcCall = mc.simulateCallPrice(spot, strike, rate, volatility, timeToMaturity);
    double mcPut = mc.simulatePutPrice(spot, strike, rate, volatility, timeToMaturity);
    
    printOptionPrice("Monte Carlo", "Call", mcCall);
    printOptionPrice("Monte Carlo", "Put", mcPut);

    // Binomial Tree model
    BinomialTree bt(1, 1000);
    double btCall = bt.calculateCallPrice(spot, strike, rate, volatility, timeToMaturity);
    double btPut = bt.calculatePutPrice(spot, strike, rate, volatility, timeToMaturity);
    
    printOptionPrice("Binomial Tree", "Call", btCall);
    printOptionPrice("Binomial Tree", "Put", btPut);
}

void runBenchmarks() {
    std::cout << "\nRunning Performance Benchmarks:\n"
              << "--------------------------------\n";

    // Get number of available hardware threads
    size_t maxThreads = std::thread::hardware_concurrency();
    std::vector<size_t> threadCounts;
    
    // Create thread counts: 1, 2, 4, ..., up to maxThreads
    for (size_t threads = 1; threads <= maxThreads; threads *= 2) {
        threadCounts.push_back(threads);
    }

    // Create and run benchmarks
    Benchmark benchmark(threadCounts);
    size_t numOptions = 10000;
    
    std::cout << "Testing with " << numOptions << " options...\n\n";
    
    auto results = benchmark.runAllBenchmarks(numOptions);
    
    // Print results
    std::cout << std::setw(15) << std::left << "Model"
              << std::setw(10) << std::right << "Threads"
              << std::setw(15) << std::right << "Time (ms)"
              << std::setw(15) << std::right << "Speedup"
              << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    for (const auto& result : results) {
        std::cout << std::setw(15) << std::left << result.modelName
                  << std::setw(10) << std::right << result.numThreads
                  << std::setw(15) << std::right << std::fixed 
                  << std::setprecision(2) << result.executionTimeMs
                  << std::setw(15) << std::right << std::fixed 
                  << std::setprecision(2) << result.speedupFactor
                  << std::endl;
    }

    // Save results to file
    benchmark.saveResults("benchmark_results.csv", results);
    std::cout << "\nBenchmark results saved to 'benchmark_results.csv'\n";
}

int main() {
    try {
        std::cout << "Options Pricing Model Demonstration\n"
                  << "===================================\n";

        // Run single option example
        runSingleOptionExample();

        // Run performance benchmarks
        runBenchmarks();

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 