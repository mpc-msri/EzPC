#!/bin/bash
# Authors: Pratik Bhatu.

# Copyright:
# Copyright (c) 2021 Microsoft Research
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

if [ -z "$1" ]; then
	scale=12
else
	scale=$1
fi

filename="tf-densenet121.tar.gz"
if [ ! -f "PreTrainedModel/${filename}" ]; then
  echo "----------------------------------"
  echo "Downloading trained DenseNet Model"
  echo "----------------------------------"
  fileid=0B_fUSpodN0t0eW1sVk1aeWREaDA
  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o PreTrainedModel/${filename}
  rm cookie
fi
if [[  ! -f "PreTrainedModel/tf-densenet121.ckpt.data-00000-of-00001"  ||  ! -f "PreTrainedModel/tf-densenet121.ckpt.index"  ||  ! -f "PreTrainedModel/tf-densenet121.ckpt.meta"  ]]; then
  cd PreTrainedModel
  tar -xvzf ${filename}
  cd -
fi
#exit
echo -e "\n\n"
echo "--------------------------------------------------------------------------------"
echo "Running DenseNet network and dumping computation graph, inputs and model weights"
echo "This will take some time"
echo "--------------------------------------------------------------------------------"
echo -e "\n\n"
python3 DenseNet_main.py --runPrediction True --scalingFac $scale --saveImgAndWtData True
echo -e "\n\n"
