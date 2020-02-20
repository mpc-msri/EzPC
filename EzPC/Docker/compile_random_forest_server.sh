#!/bin/sh

# Authors: Mayank Rathee.

# Copyright:
# Copyright (c) 2018 Microsoft Research
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

# get the latest version
git pull

# Switch ocaml version
echo "================================================================"
echo "ocaml version switching"
echo "================================================================"
opam switch 4.06.1
eval "opam config env"
echo "SWITCHED"

# make EzPC again
cd ../EzPC/
make
chmod +x ezpc.sh
cd ..

cd EzPC/seclud_random_forest
rm pickle_model.pickle

# Check the number of input arguments to this script
# 1st argument is the task type: cla for classification and reg for regression
# 2nd argument is the number of features in the dataset
# 3rd argument is the model type: tree for decision tree and forest for random forest
# 4th argument is port no. of server
# 5th argument, if exists, is pickle file path. If not exist, default path of ../../../pickle_model.pickle is taken

if [ $# -lt 4 ]
then
	echo "ERROR. 4 args expected. Usage is <script>.sh cla/reg #features Server_Port [Pickle File Absolute Path]"
	exit 1

elif [ $# -eq 4 ]
then
	echo "Taking pickle file from /ezpc-workdir/EzPC/pickle_model.pickle"

elif [ $# -eq 5 ]
then
	echo "Taking pickle file from the 5th argument value supplied"
	cp $5 ../../../pickle_model.pickle
fi

echo "================================================================"
echo "Copying model files to correct directories"
echo "================================================================"

cp ../../../pickle_model.pickle .

# To make sure that client does not read stale stats
rm decision_tree_stat.txt
rm decision_tree_stat1.txt

#python convert_pickle_to_graphviz.py $1
python3 convert_pickle_to_graphviz.py pickle_model.pickle $1 $3

echo "================================================================"
echo "Compiling to ABY"
echo "================================================================"

pwd
./master_script_random_forest.sh $1 $2

echo "================================================================"
echo "Copying preprocessed model to correct directory"
echo "================================================================"
cp ezpc_parsed_tree.txt ../../../../ABY-latest/ABY/build/bin/ezpc_parsed_tree.txt

echo "================================================================"
echo "Copying generated files to latest ABY examples directory"
echo "================================================================"
cp random_forest_main0.cpp ../../../../ABY-latest/ABY/src/examples/docker-test/common/millionaire_prob.cpp

# Add all example files to CMakeLists
cd ../../../../ABY-latest/ABY/src/examples/docker-test/
> CMakeLists.txt
echo 'add_executable(random_forest millionaire_prob_test.cpp common/millionaire_prob.cpp common/millionaire_prob.h)' >> CMakeLists.txt
echo 'target_link_libraries(random_forest ABY::aby ENCRYPTO_utils::encrypto_utils)' >> CMakeLists.txt

# Build the ABY files of docker-test
cd ../../../build
make

cd ../../../EzPC/EzPC

echo "DONE"
echo "================================================================"
echo "Executables are ready. They can be found in ABY/build/bin"
echo "================================================================"

cp Docker/transfer_data_to_client.py .

echo "================================================================"
echo "Running HTTP server on Server machine to communicate tree depth and #trees to client"
echo "================================================================"
python3 transfer_data_to_client.py $4
rm transfer_data_to_client.py

(true || rm port_server_s.txt)
echo $4 > port_server_s.txt

echo "\n[STATUS] Success!"

