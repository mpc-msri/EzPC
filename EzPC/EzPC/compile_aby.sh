#!/usr/bin/bash
ABY_CPP_FILE=$1
EZPC_SRC_PATH=$(dirname $0)

if [ -z ${ABY_CPP_FILE} ]; then
  echo "Please specify file name of the generated .cpp file using ./ezpc.sh";
  exit;
fi

BINARY_NAME=$(basename $ABY_CPP_FILE .cpp)

g++ -w ${EZPC_SRC_PATH}/ABY_example/millionaire_prob_test.cpp ${ABY_CPP_FILE} \
        -I ${EZPC_SRC_PATH}/ABY_example/common \
        -I ${EZPC_SRC_PATH}/ABY/build/install/include/ \
        -I ${EZPC_SRC_PATH}/ABY/build/extern/ENCRYPTO_utils/include \
        -L ${EZPC_SRC_PATH}/ABY/build/install/lib  \
        -laby -lencrypto_utils -lrelic_s -lotextension -lgmp -lgmpxx \
        -lpthread -lssl -lcrypto -lboost_system -lboost_thread \
	-o $BINARY_NAME

if [ $? -ne 0 ]; then
  echo "Compilation failed"
  exit
else
  echo "Output binary: $(dirname $BINARY_NAME)/$BINARY_NAME"  
fi
