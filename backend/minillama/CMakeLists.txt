cmake_minimum_required(VERSION 3.16)
project(LLAMA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# add_subdirectory(gpu)

add_library(LLAMA
    cryptoTools/cryptoTools/Common/Defines.cpp
    cryptoTools/cryptoTools/Common/Log.cpp
    cryptoTools/cryptoTools/Crypto/AES.cpp
    cryptoTools/cryptoTools/Crypto/PRNG.cpp
    and.cpp
    api.cpp
    comms.cpp
    config.cpp
    conv.cpp
    dcf.cpp
    float.cpp
    input_prng.cpp
    mic.cpp
    msnzb.cpp
    mult.cpp
    prng.cpp
    pubdiv.cpp
    relutruncate.cpp
    relu.cpp
    reluextend.cpp
    signextend.cpp
    select.cpp
    stats.cpp
    taylor.cpp
    truncate.cpp
    utils.cpp
)

target_link_libraries (LLAMA Eigen3::Eigen Threads::Threads)
target_include_directories(LLAMA PRIVATE cryptoTools)
