# High-Performance Options Pricing Model

A high-performance C++ implementation of various options pricing models utilizing multithreading for optimal performance.

## Features

- Multiple pricing models:
  - Black-Scholes (closed-form solutions)
  - Monte Carlo simulation
  - Binomial Tree model
- Multithreaded computation for improved performance
- Thread pool implementation for efficient resource management
- Comprehensive benchmarking system
- Lock-free data structures for high concurrency

## Requirements

- C++17 or later
- CMake 3.15 or later
- A modern C++ compiler (GCC, Clang, or MSVC)
- Google Test (for unit testing)

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Project Structure

```
/options_pricing
│── /src                       # Source files
│── /include                   # Header files
│── /tests                     # Unit tests
│── /data                      # Sample data and benchmarks
│── /build                     # Build directory
│── CMakeLists.txt            # CMake configuration
│── README.md                 # This file
```

## Usage

```cpp
#include "BlackScholes.h"
#include "MonteCarlo.h"
#include "BinomialTree.h"

// Example usage of Black-Scholes model
BlackScholes bs;
double price = bs.calculateCallPrice(spot, strike, rate, volatility, timeToMaturity);

// Example usage of Monte Carlo simulation
MonteCarlo mc(numThreads);
double mcPrice = mc.simulateCallPrice(spot, strike, rate, volatility, timeToMaturity, numSimulations);
```

## Performance Benchmarks

The multithreaded implementation shows significant performance improvements:
- Monte Carlo: Up to 8x speedup with 8 threads
- Binomial Tree: Up to 4x speedup with optimal thread allocation
- Batch Black-Scholes: Near-linear scaling for large option batches

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 