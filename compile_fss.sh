#!/bin/bash

# Authors: Saksham Gupta.

# Copyright:
# Copyright (c) 2020 Microsoft Research
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

FSS_CPP_FILE=$1
EZPC_SRC_PATH=$(dirname $0)

if [ ! -e "$FSS_CPP_FILE" ]; then
  echo "Please specify file name of the generated .cpp file using CompileONNXGraph.py";
  exit;
fi

BINARY_NAME=$(basename $FSS_CPP_FILE .cpp)
DIR="$(dirname "${FSS_CPP_FILE}")" 

pd=$(pwd)
rm -rf build_dir
mkdir build_dir
cd build_dir
eval `opam config env`

path=$(dirname $0)
alt_root=$2
echo "
cmake_minimum_required (VERSION 3.13) 
project(fptraining)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations\")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
add_subdirectory($alt_root/fptraining/backend/minillama $pd/minillama_build)
add_executable($BINARY_NAME
    ../$FSS_CPP_FILE 
    $alt_root/fptraining/backend/minillama/add.cpp
    $alt_root/fptraining/backend/minillama/lib.cpp
)
target_link_libraries ($BINARY_NAME Eigen3::Eigen )
target_link_libraries ($BINARY_NAME Threads::Threads )
target_link_libraries ($BINARY_NAME  LLAMA)
target_include_directories($BINARY_NAME PRIVATE  $alt_root/fptraining/backend/minillama/cryptoTools)

" > CMakeLists.txt

cmake -DCMAKE_BUILD_TYPE=Release .
make -j4
rm -rf ../$BINARY_NAME 
mv $BINARY_NAME ../$DIR
cd ..
# rm -rf build_dir



if [ -e "../$BINARY_NAME" ]; then
  echo "Compilation failed"
  exit
else
  echo "Output binary: $(dirname $BINARY_NAME)/$BINARY_NAME"  
fi