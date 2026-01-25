cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)

project(gpu_blast LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.6)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_executable(test
        gpu_blast.cu
        tests_gpu_blast.cu
)