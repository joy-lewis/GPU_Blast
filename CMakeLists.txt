#cmake_minimum_required(VERSION 3.18)
#project(gpu_blast LANGUAGES CXX CUDA)
#
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 17)
#
#find_package(CUDAToolkit REQUIRED)

###
#add_executable(gpu_blast_app
#        gpu_blast.cu
#)
#
#add_executable(test
#        tests_gpu_blast.cu
#        # plus any .cu/.cpp files that contain functions the tests need,
#        # BUT do NOT include gpu_blast.cu if it contains main()
#)

#target_link_libraries(gpu_blast_app PRIVATE CUDA::cudart)
#target_link_libraries(test         PRIVATE CUDA::cudart)

cmake_minimum_required(VERSION 3.18)
project(gpu_blast LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# This creates the single executable "gpu_blast"
add_executable(gpu_blast
        main.cu             # The file with the real main()
        gpu_blast.cu        # Contains blast_main() and encoder()
        tests_gpu_blast.cu  # Contains test_main()
)

# FIX: Link the library to "gpu_blast", not the old names
target_link_libraries(gpu_blast PRIVATE CUDA::cudart)