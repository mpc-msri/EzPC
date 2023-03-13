#!/bin/bash

FSS_CPP_FILE=$1
EZPC_SRC_PATH=$(dirname $0)

if [ ! -e "$FSS_CPP_FILE" ]; then
  echo "Please specify file name of the generated .cpp file using CompileONNXGraph.py";
  exit;
fi

BINARY_NAME=$(basename $FSS_CPP_FILE .cpp)
DIR="$(dirname "${FSS_CPP_FILE}")" 

script_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# echo The script is located at $script_path
alt_root=$script_path/../..
# echo The alt_root is $alt_root

rm -rf build_dir
mkdir build_dir
cd build_dir
pd=$(pwd)

echo "
cmake_minimum_required (VERSION 3.13) 
project(sytorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp\")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
add_subdirectory($alt_root/ext/cryptoTools $pd/cryptoTools)
add_subdirectory($alt_root/ext/llama $pd/llama)
add_executable($BINARY_NAME 
    ../$FSS_CPP_FILE $alt_root/src/sytorch/random.cpp $alt_root/src/sytorch/backend/cleartext.cpp
)
target_include_directories($BINARY_NAME
PUBLIC
    \$<BUILD_INTERFACE:$alt_root/include>
    \$<INSTALL_INTERFACE:\${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries ($BINARY_NAME Eigen3::Eigen Threads::Threads LLAMA cryptoTools)
" > CMakeLists.txt

cmake -DCMAKE_BUILD_TYPE=Release .
make -j4
rm -rf ../$BINARY_NAME 
mv $BINARY_NAME ../$DIR
cd ..
rm -rf build_dir



if [ -e "../$BINARY_NAME" ]; then
  echo "Compilation failed"
  exit
else
  echo "Output binary: $(dirname $BINARY_NAME)/$BINARY_NAME"  
fi