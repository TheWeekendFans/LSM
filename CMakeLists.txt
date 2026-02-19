cmake_minimum_required(VERSION 3.10)
project(PyramidLSM)

set(CMAKE_CXX_STANDARD 17) # Use C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set SIMD flags based on compiler
if(MSVC)
    add_compile_options(/arch:AVX2 /openmp /O2)
else()
    add_compile_options(-mavx2 -fopenmp -O3)
endif()

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.h")

add_executable(PyramidLSM ${SOURCES} ${HEADERS})

# OpenMP linking
if(NOT MSVC)
    find_package(OpenMP REQUIRED)
    target_link_libraries(PyramidLSM PRIVATE OpenMP::OpenMP_CXX)
endif()
