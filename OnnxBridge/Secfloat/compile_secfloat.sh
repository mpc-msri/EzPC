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

SECFLOAT_CPP_FILE=$1
EZPC_SRC_PATH=$(dirname $0)

if [ ! -e "$SECFLOAT_CPP_FILE" ]; then
  echo "Please specify file name of the generated .cpp file using ./ezpc.sh";
  exit;
fi

BINARY_NAME=$(basename $SECFLOAT_CPP_FILE .cpp)
DIR="$(dirname "${SECFLOAT_CPP_FILE}")" 


rm -rf build_dir
mkdir build_dir
cd build_dir
eval `opam config env`

path=$(dirname $0)
echo "
cmake_minimum_required (VERSION 3.13) 
project (BUILD_IT)
set(CMAKE_MODULE_PATH \${CMAKE_CURRENT_SOURCE_DIR})
find_package(SCI REQUIRED PATHS \"$path/../../SCI/build/install\") 
add_executable($BINARY_NAME ../$SECFLOAT_CPP_FILE)
target_include_directories($BINARY_NAME PUBLIC)
target_compile_options($BINARY_NAME PRIVATE -fconcepts -g)
target_link_libraries($BINARY_NAME SCI::SCI-SecfloatML )
" > CMakeLists.txt

cmake --log-level=ERROR .

cmake --build . --parallel
exit
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
