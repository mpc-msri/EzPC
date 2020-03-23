#!/bin/bash

# Authors: Shubham Ugare.

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

# This script will 
# 1) compile the ONNX model to SeeDot AST 
# 2) Compile the SeeDot AST to ezpc
# 3) Convert the ezpc code to cpp and then run it on the given dataset

modelName=$1
debugOnnxNode=$2

EzPCDir="../../EzPC"
ONNX_dir="../../Athos/ONNXCompiler"	
BITLEN="64"
SCALINGFACTOR="12"
COMPILATIONTARGET="CPP"
ezpcOutputFullFileName=${modelName}'.ezpc'
compilationTargetLower=$(echo "$COMPILATIONTARGET" | awk '{print tolower($0)}')
compilationTargetHigher=$(echo "$COMPILATIONTARGET" | awk '{print toupper($0)}')

python3 ../SeeDot/SeeDot.py -p astOutput.pkl --astFile astOutput.pkl --outputFileName "$ezpcOutputFullFileName" 
finalCodeOutputFileName=${modelName}'0.cpp'
inputFileName=${modelName}'_input.h'
seedotASTName=${modelName}'.pkl'

# modelname_input.npy and modelname_output.npy
onnxInputFileName=${modelName}'_input.npy'
onnxOutputFileName=${modelName}'_output.npy'

if [ -f "$inputFileName" ] && [ -f "$seedotASTName" ] && [ -z "$debugOnnxNode" ]; then
    echo "$inputFileName and $seedotASTName already exist, skipping process_onnx"
else 
	echo "Starting onnx run to gemerate input and output"
	python3 "onnx_run.py" ${modelName}'.onnx' ${debugOnnxNode}
	echo "Finished onnx run"
    echo "Starting process_onnx"
    python3 "process_onnx.py" ${modelName}'.onnx' 
    echo "Finished process_onnx"
fi

if [ -z "$debugOnnxNode" ]; then 
	python3 ../SeeDot/SeeDot.py -p $seedotASTName --astFile $seedotASTName --outputFileName "$ezpcOutputFullFileName" --consSF ${SCALINGFACTOR}
else 	
	debugSeedotNode=$(python3 -c "import common; common.get_seedot_name_from_onnx_name(\"${debugOnnxNode}\")")
	echo "${debugSeedotNode} is the corresponding SeeDot name"
	python3 ../SeeDot/SeeDot.py -p $seedotASTName --astFile $seedotASTName --outputFileName "$ezpcOutputFullFileName" --consSF ${SCALINGFACTOR} --debugVar ${debugSeedotNode}	
fi 

python3 -c 'import common; common.merge_name_map()'


cat "../TFEzPCLibrary/Library${BITLEN}_cpp.ezpc" "../TFEzPCLibrary/Library${BITLEN}_common.ezpc" "$ezpcOutputFullFileName" > temp
mv temp "$ezpcOutputFullFileName"

cp "$ezpcOutputFullFileName" "$EzPCDir/EzPC"
cd "$EzPCDir/EzPC"
eval `opam config env`
./ezpc.sh "$ezpcOutputFullFileName" --bitlen "$BITLEN" --codegen "$compilationTargetHigher" --disable-tac

if [ "$compilationTargetLower" == "cpp" ]; then
	# cd "$fullDirPath"
	mv "$finalCodeOutputFileName" "$ONNX_dir"
	rm '$EzPCDir/EzPC/'${modelName}'*'
	cd "$ONNX_dir"
	g++ -O3 -g "$finalCodeOutputFileName" -o ${modelName}'.out'
	rm "debug/cpp_output_raw.txt"
	eval './'${modelName}'.out' < ${inputFileName} > "debug/cpp_output_raw.txt"
	python3 -c "import common; common.parse_output(${SCALINGFACTOR})"
	echo -e "All compilation done."
fi
