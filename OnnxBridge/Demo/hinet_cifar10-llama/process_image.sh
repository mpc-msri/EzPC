#!/bin/bash

image=$1

actualFileName="${image%.*}"
python preprocess.py $image
python ../../helper/convert_np_to_float_inp.py --inp $actualFileName.npy --output ${actualFileName}_input.inp
