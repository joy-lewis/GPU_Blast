cmake_minimum_required(VERSION 3.18)

# Specific Anaconda CUDA compiler
set(CMAKE_CUDA_COMPILER /home/joy/anaconda3/envs/bw312/bin/nvcc)

project(gpu_blast LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Defines the main executable
add_executable(gpu_blast
        main.cu
        gpu_blast.cu
        gpu_blast.h
        lookup_table.cuh
)

# Links CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpu_blast PRIVATE CUDA::cudart)

# Copys data to build folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ncbi_data
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})