#!/bin/sh

# Authors: Mayank Rathee.

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

cd EzPC/

echo "Select which example to compile"
read -p "Enter 0 for binary example or 1 for arithmetic example: " choice

if [ $choice -lt 1 ]
then
	echo "================================================================"
	echo "Compiling docker binary ops example to ABY"
	echo "================================================================"
	pwd
	cp docker_test/docker_bin_example.ezpc .
	ls
	./ezpc.sh docker_bin_example.ezpc

	echo "================================================================"
	echo "Copying generated files to latest ABY examples directory"
	echo "================================================================"
	cp docker_bin_example0.cpp ../../../ABY-latest/ABY/src/examples/docker-test/common/millionaire_prob.cpp
else
	echo "================================================================"
	echo "Compiling docker arithmetic ops example to ABY"
	echo "================================================================"
	cp docker_test/docker_arith_example.ezpc .
	./ezpc.sh docker_arith_example.ezpc

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

