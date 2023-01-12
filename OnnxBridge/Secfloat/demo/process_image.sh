#!/bin/bash

image=$1

actualFileName="${image%.*}"
python ../../helper/pre_process.py $image
python ../../helper/convert_np_to_float_inp.py --inp $actualFileName.npy --output ${actualFileName}_input.inp
