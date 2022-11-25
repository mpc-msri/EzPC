#!/bin/bash


imgdir="./images"
executable="model"
fullFileName="${imgdir}/input.jpg"
baseFileName=$(basename -- $fullFileName)
actualFileName="${baseFileName%.*}"
echo ${actualFileName}
python helper/pre_process.py $fullFileName
python helper/convert_np_to_float_inp.py --inp images/$actualFileName.npy --output images/${actualFileName}_input.inp
python helper/run_onnx.py images/${actualFileName}.npy > onnx_output/output.txt
./onnx_files/${executable}_secfloat r=2 port=9000  < onnx_files/${executable}_input_weights_.inp &
./onnx_files/${executable}_secfloat r=1 port=9000  < images/${actualFileName}_input.inp  > onnx_files/output.txt &
wait
python helper/make_np_arr.py onnx_files/output.txt
python helper/compare_np_arrs.py -i onnx_files/output.npy onnx_output/${actualFileName}.npy  -v >> debug.txt
exit
# done