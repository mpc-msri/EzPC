#!/usr/bin/bash
EMP_CPP_FILE=$1
EZPC_SRC_PATH=$(dirname $0)

if [ -z ${EMP_CPP_FILE} ]; then
  echo "Please specify file name of the generated .cpp file using ./ezpc.sh";
  exit;
fi

BINARY_NAME=$(basename $EMP_CPP_FILE .cpp)

g++  ${EMP_CPP_FILE} \
        -I ${EZPC_SRC_PATH}/emp-sh2pc \
        -DEMP_CIRCUIT_PATH=/usr/local/include/emp-tool/circuits/files/ \
        -fconcepts -pthread -Wall -funroll-loops -Wno-ignored-attributes \
        -Wno-unused-result -march=native -maes -mrdseed -std=c++14 -O3 \
        -Wl,-rpath,/usr/local/lib /usr/local/lib/libemp-tool.so \
        -lssl -lcrypto \
	-o $BINARY_NAME

if [ $? -ne 0 ]; then
  echo "Compilation failed"
  exit
else
  echo "Output binary: $(dirname $BINARY_NAME)/$BINARY_NAME"  
fi