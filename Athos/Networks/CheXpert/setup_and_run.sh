#!/bin/bash
# Authors: Saksham Gupta.

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
	scale=23
else
	scale=$1
fi

if [ ! -f "Data_batch/preprocess_valid_batch.p" ] ; then
	echo -e "\n\n"
	echo "--------------------------------------------------------------------------------"
	echo "One-time set up of ChexPert dataset"
	echo "This will take some time"
    echo "Processed and scaled data is dumped in Data_batch/preprocess_valid_batch.p"
    echo "Data can be retrieved with function - Util.load_preprocess_validation_data()"
	echo "--------------------------------------------------------------------------------"
	echo -e "\n\n"
	cd "../../HelperScripts" 
	./SetupCheXpert.sh
	cd -
	python Util.py $scale
fi

echo "--------------------------------------------------------------------------------"
echo "Preparing images in .inp format"
echo "This will take some time"
echo "--------------------------------------------------------------------------------"
echo -e "\n\n"
python Prepare_Model.py