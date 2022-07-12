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

EMP_CPP_FILE=$1
EZPC_SRC_PATH=$(dirname $0)

if [ ! -e ${EMP_CPP_FILE} ]; then
  echo "Please specify file name of the generated .cpp file using ./ezpc.sh";
  exit;
fi

BINARY_NAME=$(basename $EMP_CPP_FILE .cpp)

g++  ${EMP_CPP_FILE} \
        -I ${EZPC_SRC_PATH}/emp-sh2pc \
        -DEMP_CIRCUIT_PATH=/usr/local/include/emp-tool/circuits/files/ \
        -fconcepts -pthread -funroll-loops -Wno-ignored-attributes \
        -Wno-unused-result -Wno-write-strings -march=native -maes -mrdseed -std=c++14 -O3 \
        -Wl,-rpath,/usr/local/lib /usr/local/lib/libemp-tool.so \
        -lssl -lcrypto \
	-o $BINARY_NAME

if [ $? -ne 0 ]; then
  echo "Compilation failed"
  exit
else
  echo "Output binary: $(dirname $BINARY_NAME)/$BINARY_NAME"  
fi