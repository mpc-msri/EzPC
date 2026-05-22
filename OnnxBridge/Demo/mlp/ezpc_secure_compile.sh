#!/bin/bash

onnx_model_path=${1:-"mlp_model.onnx"}
work_dir="EzPC/OnnxBridge/workdir"

model_basename=$(basename $onnx_model_path)
mkdir -p $work_dir
# TODO: use symbolic link instead
cp $onnx_model_path $work_dir/mlp_model.onnx
cd $work_dir

# ------------------SERVER------------------
# Compile the onnx model
python ../main.py --path $model_basename --generate "executable" --backend LLAMA --scale 15 --bitlength 64

# ------------------DEALER------------------
# generate keys
./mlp_model_LLAMA_15 1
