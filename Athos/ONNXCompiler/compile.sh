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

EzPCDir="../../EzPC"
ONNX_dir="../../Athos/ONNXCompiler"	
BITLEN="64"
SCALINGFACTOR="24"
COMPILATIONTARGET="CPP"
ezpcOutputFullFileName=${modelName}'.ezpc'
compilationTargetLower=$(echo "$COMPILATIONTARGET" | awk '{print tolower($0)}')
compilationTargetHigher=$(echo "$COMPILATIONTARGET" | awk '{print toupper($0)}')
finalCodeOutputFileName=${modelName}'0.cpp'
inputFileName=${modelName}'_input.h'
seedotASTName=${modelName}'.pkl'

# modelname_input.npy and modelname_output.npy
onnxInputFileName=${modelName}'_input.npy'
onnxOutputFileName=${modelName}'_output.npy'

if [ -f "$onnxInputFileName" ] && [ -f "$onnxOutputFileName" ]; then
    echo "$onnxInputFileName and $onnxOutputFileName already exist, skipping onnx run"
else 
    echo "Starting onnx run to gemerate input and output"
    python3 "onnx_run.py" ${modelName}'.onnx' 
    echo "Finished onnx run"
fi


if [ -f "$inputFileName" ] && [ -f "$seedotASTName" ]; then
    echo "$inputFileName and $seedotASTName already exist, skipping process_onnx"
    cp '$inputFileName' 'models/$inputFileName'
else 
    echo "Starting process_onnx"
    python3 "process_onnx.py" ${modelName}'.onnx' 
    echo "Finished process_onnx"
fi

python3 ../SeeDot/SeeDot.py -p $seedotASTName --astFile $seedotASTName --outputFileName "$ezpcOutputFullFileName" --consSF ${SCALINGFACTOR}

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
	g++ -O3 "$finalCodeOutputFileName" -o ${modelName}'.out'
	echo -e "All compilation done."
fi

