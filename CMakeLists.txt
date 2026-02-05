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

# Links the library to "gpu_blast"
target_link_libraries(gpu_blast PRIVATE CUDA::cudart)

# This copies the 'ncbi_data' folder from the source root into the build folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ncbi_data
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})