#!/bin/bash

cd ../../ABY-latest/ABY/build/bin/
rm -r client
mkdir client
cp docker_binop_example* client/.
cd client
pwd

./docker_binop_example -r 0

cd ../../../../../EzPC
