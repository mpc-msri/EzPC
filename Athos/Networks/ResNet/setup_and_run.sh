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
filename=resnet_v2_fp32_savedmodel_NHWC.tar.gz
if [ ! -f "PreTrainedModel/${filename}" ]; then
  echo "--------------------------------"
  echo "Downloading trained ResNet Model"
  echo "--------------------------------"
  wget "http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz" -P ./PreTrainedModel
fi

if [[  ! -d "PreTrainedModel/resnet_v2_fp32_savedmodel_NHWC" ]]; then
  cd PreTrainedModel
  tar -xvzf ${filename}
  cd -
fi
#exit
echo -e "\n\n"
echo "--------------------------------------------------------------------------------"
echo "Running ResNet network and dumping computation graph, inputs and model weights"
echo "This will take some time"
echo "--------------------------------------------------------------------------------"
echo -e "\n\n"
python3 ResNet_main.py --runPrediction True --scalingFac $scale --saveImgAndWtData True
echo -e "\n\n"
