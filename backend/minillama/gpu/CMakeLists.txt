cmake_minimum_required(VERSION 3.16)
project(LLAMA-GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -m64 -fopenmp")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17" )
# find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(Threads REQUIRED)
find_package(CUDA 11.7 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
set(CUTLASS_ENABLE_EXAMPLES FALSE)
set(CUTLASS_ENABLE_LIBRARY  OFF)
set(CUTLASS_ENABLE_PROFILER OFF)
add_subdirectory(ext/cutlass)

add_library(LLAMA-GPU
    gpu_linear_helper.cu
    gpu_mem.cu
    gpu_matmul.cu
    gpu_conv2d.cu
    gpu_relu_truncate.cu
    conv2d_layer.cpp 
    fc_layer.cpp 
    maxpool_layer.cpp 
    relu_sign_extend_layer.cpp 
    gpu_sgd.cpp 
    gpu_truncate.cpp 
    gpu_relu.cpp 
    gpu_select.cpp 
    gpu_and.cpp 
    gpu_fss_utils.cpp 
    gpu_dcf.cpp 
    gpu_comms.cpp 
    gpu_file_utils.cpp
)

set_property(TARGET LLAMA-GPU PROPERTY CUDA_STANDARD 14)

target_link_libraries(LLAMA-GPU CUTLASS cutlass_tools_util_includes ${CUDA_LIBRARY_DIRS} cudart LLAMA)
# target_link_libraries (LLAMA-GPU Eigen3::Eigen Threads::Threads)
target_include_directories(LLAMA-GPU PRIVATE ../cryptoTools)

