#!/bin/sh

# Switch ocaml version
echo "================================================================"
echo "ocaml version switching"
echo "================================================================"
opam switch 4.06.1
eval `opam config env`
echo "SWITCHED"

# make EzPC again
cd EzPC/
make
cd ..

echo "================================================================"
echo "Compiling docker binary ops example to ABY"
echo "================================================================"
cd EzPC/docker_test
../ezpc docker_bin_example.ezpc

echo "================================================================"
echo "Copying generated files to latest ABY examples directory"
cp docker_bin_example0.cpp ../../../ABY-latest/ABY/src/examples/docker-test/common/millionaire_prob.cpp

# Add all perceptron files to CMakeLists
cd ../../../ABY-latest/ABY/src/examples/docker-test/
> CMakeLists.txt
echo 'add_executable(docker_binop_example millionaire_prob_test.cpp common/millionaire_prob.cpp common/millionaire_prob.h)' >> CMakeLists.txt
echo 'target_link_libraries(docker_binop_example ABY::aby ENCRYPTO_utils::encrypto_utils)' >> CMakeLists.txt

# Build the ABY files of docker-test
cd ../../../build
make

cd ../../../ezpc/

echo "DONE"
