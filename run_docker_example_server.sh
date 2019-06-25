#!/bin/bash

cd ../ABY-latest/ABY/build/bin/
rm -r server
mkdir server
cp docker_binop_example* server/.
cd server
pwd

./docker_binop_example -r 1

cd ../../../../../ezpc
