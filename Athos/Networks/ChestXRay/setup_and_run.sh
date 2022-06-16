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

filename="chexray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5"
if [ ! -f "PreTrainedModel/KerasModel/${filename}" ]; then
  echo "----------------------------------"
  echo "Downloading trained ChestXRay Model"
  echo "----------------------------------"
  curl "https://chestxray.blob.core.windows.net/chestxraytutorial/tutorial_xray/chexray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5" -o PreTrainedModel/KerasModel
  if [ ! -f "PreTrainedModel/TFModel/model.pb" ]; then
    cd PreTrainedModel
    if [ ! -d "keras_to_tensorflow" ]; then 
      git clone https://github.com/amir-abdi/keras_to_tensorflow
    fi
    echo -e "Starting keras to TF model conversion....\n"
    python3 keras_to_tensorflow/keras_to_tensorflow.py --output_meta_ckpt=True --save_graph_def=True --input_model="KerasModel/${filename}" --output_model="TFModel/model.pb"
  fi
fi

#exit
echo -e "\n\n"
echo "--------------------------------------------------------------------------------"
echo "Running ChestXRay network and dumping computation graph, inputs and model weights"
echo "This will take some time"
echo "--------------------------------------------------------------------------------"
echo -e "\n\n"
python3 ChestXRay_tf_main.py --runPrediction True --scalingFac $scale --saveImgAndWtData True
echo -e "\n\n"
