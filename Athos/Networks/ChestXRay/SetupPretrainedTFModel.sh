#!/bin/bash

# Authors: Nishant Kumar.

# Copyright:
# Copyright (c) 2020 Microsoft Research
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

kerasModelLink="https://chestxray.blob.core.windows.net/chestxraytutorial/tutorial_xray/chexray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5"
axel -a -n 3 -c --output PreTrainedModel/KerasModel "$kerasModelLink"
cd PreTrainedModel
git clone https://github.com/amir-abdi/keras_to_tensorflow
cd keras_to_tensorflow
echo -e "Starting keras to TF model conversion....\n"
python3 keras_to_tensorflow.py --output_meta_ckpt=True --save_graph_def=True --input_model="../KerasModel/chexray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5" --output_model="../TFModel/model.pb"
