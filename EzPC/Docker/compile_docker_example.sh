#!/bin/sh

# Switch ocaml version
echo "================================================================"
echo "ocaml version switching"
echo "================================================================"
opam switch 4.06.1
eval `opam config env`
echo "SWITCHED"

# make EzPC again
cd ../EzPC/
make
cd ..

cd EzPC/docker_test

echo "Select which example to compile"
read -p "Enter 0 for binary example or 1 for arithmetic example: " choice

if [ $choice -lt 1 ]
then
	echo "================================================================"
	echo "Compiling docker binary ops example to ABY"
	echo "================================================================"
	../ezpc docker_bin_example.ezpc

	echo "================================================================"
	echo "Copying generated files to latest ABY examples directory"
	echo "================================================================"
	cp docker_bin_example0.cpp ../../../ABY-latest/ABY/src/examples/docker-test/common/millionaire_prob.cpp
else
	echo "================================================================"
	echo "Compiling docker arithmetic ops example to ABY"
	echo "================================================================"
	../ezpc docker_arith_example.ezpc

	echo "================================================================"
	echo "Copying generated files to latest ABY examples directory"
	echo "================================================================"
	cp docker_arith_example0.cpp ../../../ABY-latest/ABY/src/examples/docker-test/common/millionaire_prob.cpp
fi


# Add all example files to CMakeLists
cd ../../../ABY-latest/ABY/src/examples/docker-test/
> CMakeLists.txt
echo 'add_executable(docker_binop_example millionaire_prob_test.cpp common/millionaire_prob.cpp common/millionaire_prob.h)' >> CMakeLists.txt
echo 'target_link_libraries(docker_binop_example ABY::aby ENCRYPTO_utils::encrypto_utils)' >> CMakeLists.txt

# Build the ABY files of docker-test
cd ../../../build
make

cd ../../../EzPC/

echo "DONE"
echo "================================================================"
echo "Executables are ready. They can be found in ABY/build/bin"
echo "================================================================"

