#!/bin/bash


imgdir="./images"

for fullFileName in $imgdir/*.jpg; do
    baseFileName=$(basename -- $fullFileName)
    actualFileName="${baseFileName%.*}"
    python helper/pre_process.py $fullFileName
    python helper/convert_np_to_float_inp.py --inp images/$actualFileName.npy --output images/${actualFileName}_input.inp
    python helper/run_onnx.py images/${actualFileName}.npy
    ./onnx_files/lenet_secfloat r=2 < onnx_files/lenet_input_weights_.inp &
    ./onnx_files/lenet_secfloat r=1 < images/${actualFileName}_input.inp  > onnx_files/output.txt &
    wait
    python helper/make_np_arr.py onnx_files/output.txt
    python helper/compare_np_arrs.py -i onnx_files/output.npy onnx_output/input.npy >> match_result.txt
done