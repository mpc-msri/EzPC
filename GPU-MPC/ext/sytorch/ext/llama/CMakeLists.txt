# Authors: Kanav Gupta, Neha Jawalkar
# Copyright:
# 
# Copyright (c) 2024 Microsoft Research
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.16)
project(LLAMA)

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
    mult.cpp
    pubdiv.cpp
    relu.cpp
    signextend.cpp
    pubcmp.cpp
    clip.cpp
    dpf.cpp
    lut.cpp
    select.cpp
    fixtobfloat16.cpp
    wrap.cpp
    float.cpp
    taylor.cpp
    mic.cpp
    msnzb.cpp
    truncate.cpp
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads cryptoTools bitpack)
target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)