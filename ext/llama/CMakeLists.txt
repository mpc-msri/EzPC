cmake_minimum_required(VERSION 3.16)
project(LLAMA)

if(APPLE)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME}
    src/llama/config.cpp
    src/llama/comms.cpp
    src/llama/input_prng.cpp
    src/llama/prng.cpp
    src/llama/stats.cpp
    src/llama/utils.cpp
    and.cpp
    api.cpp
    conv.cpp
    dcf.cpp
    float.cpp
    mic.cpp
    msnzb.cpp
    mult.cpp
    pubdiv.cpp
    relutruncate.cpp
    relu.cpp
    reluextend.cpp
    signextend.cpp
    select.cpp
    taylor.cpp
    truncate.cpp
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads cryptoTools)
target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)