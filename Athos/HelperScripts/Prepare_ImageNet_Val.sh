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

#Prepare imagenet validation set 

ImageNetValidationSetUrl="http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar"
ImageNetValidationSetBBoxUrl="http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_val_v3.tgz"
ImageNetValidationSetSynSetLabels="https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/imagenet_2012_validation_synset_labels.txt"

axel -a -n 3 -c --output ImageNet_ValData "$ImageNetValidationSetUrl" 
axel -a -n 3 -c --output ImageNet_ValData "$ImageNetValidationSetBBoxUrl" 
axel -a -n 3 -c --output ImageNet_ValData "$ImageNetValidationSetSynSetLabels"
cd ImageNet_ValData
mkdir img
tar -xvf ILSVRC2012_img_val.tar --directory=img
tar -xvzf ILSVRC2012_bbox_val_v3.tgz
mv val bbox
cd ..
python3 Convert_WnId_To_TrainId.py

