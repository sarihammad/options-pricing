cmake_minimum_required(VERSION 3.15)
project(options_pricing VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable optimization flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add threading support
find_package(Threads REQUIRED)

# Create directories if they don't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

# Add source files
set(SOURCES
    src/main.cpp
    src/BlackScholes.cpp
    src/MonteCarlo.cpp
    src/BinomialTree.cpp
    src/ThreadPool.cpp
    src/Benchmark.cpp
    src/Utilities.cpp
)

# Add header files
set(HEADERS
    include/BlackScholes.h
    include/MonteCarlo.h
    include/BinomialTree.h
    include/ThreadPool.h
    include/Benchmark.h
    include/Utilities.h
)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create main executable
add_executable(options_pricing ${SOURCES} ${HEADERS})

# Link against threading library
target_link_libraries(options_pricing PRIVATE Threads::Threads)

# Enable testing
enable_testing()

# Add test executable
set(TEST_SOURCES
    tests/test_BlackScholes.cpp
    tests/test_MonteCarlo.cpp
    tests/test_BinomialTree.cpp
    tests/test_ThreadPool.cpp
)

# Find GTest package
find_package(GTest REQUIRED)

# Create test executable
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests PRIVATE
    GTest::GTest
    GTest::Main
    Threads::Threads
)

# Add tests
add_test(NAME unit_tests COMMAND run_tests)

# Installation rules
install(TARGETS options_pricing DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include/options_pricing) 