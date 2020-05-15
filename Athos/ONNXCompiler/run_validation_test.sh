#!/bin/bash
DIR=val/*
rm debug/val_results.txt
touch debug/val_results.txt
for d in $DIR
do
  echo "Processing $d directory..."
	FILES=$d/*
  # take action on each file. $f store current file name
	for f in $FILES
	do
		echo "Processing $f file..."
		python3 preprocess_images.py $f > /dev/null 2>&1
    python3 load_input.py cov7.onnx 24 prep_input_covid_BGR.npy > /dev/null 2>&1
		python3 onnx_run.py cov7.onnx > /dev/null 2>&1
		echo "$d" >> debug/val_results.txt
		python3 float_parse_result.py >> debug/val_results.txt
	done
done
