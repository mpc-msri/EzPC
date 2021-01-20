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

if [ ! -f "PreProcessedImages/preprocess_batch_1.p" ] ; then
	echo -e "\n\n"
	echo "--------------------------------------------------------------------------------"
	echo "One-time set up of CIFAR10 dataset for SqueezeNet Training"
	echo "This will take some time"
	echo "--------------------------------------------------------------------------------"
	echo -e "\n\n"
	cd "../../HelperScripts" 
	./SetupCIFAR10.sh
	cd -
	python3 Util.py
fi

if [ ! -f "TrainedModel/model.meta" ] ; then
	echo -e "\n\n"
	echo "--------------------------------------------------------------------------------"
	echo "Training SqueezeNet network for 1 epoch"
	echo "This will take some time"
	echo "--------------------------------------------------------------------------------"
	echo -e "\n\n"
	python3 Squeezenet_model.py train
fi

echo -e "\n\n"
echo "--------------------------------------------------------------------------------"
echo "Running SqueezeNetCIFAR10 network and dumping computation graph, inputs and model weights"
echo "This will take some time"
echo "--------------------------------------------------------------------------------"
echo -e "\n\n"
python3 Squeezenet_model.py savegraph
python3 Squeezenet_model.py testSingleTestInpAndSaveData 1 1
echo -e "\n\n"