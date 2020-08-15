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

TOTALIMG=50000

. ./Paths.config
imgFolderName=$(realpath "$imgFolderName")
bboxFolderName=$(realpath "$bboxFolderName")
preProcessedImgSaveDirName=$(realpath "$preProcessedImgSaveDirName")

echo -e "********Running preprocessing for ResNet********"
echo -e "ImgFolderName - $imgFolderName"
echo -e "bboxFolderName - $bboxFolderName"
echo -e "Image name prefix - $imgFilePrefix"
echo -e "PreProcessed images save folder - $preProcessedImgSaveDirName"

cd ../PreProcessingImages
for ((imgNum=1;imgNum<=$TOTALIMG;imgNum++)); do
	filePath='../AccuracyAnalysisHelper/PreProcessedImages/ImageNum''_'"$imgNum"'.inp'
	imgNumPlusPlus=$((imgNum + 1))
	if [ ! -f "$filePath" ]; then
		echo -e "$imgNum", "$imgNumPlusPlus"
		python3 DenseNet_preprocess_main.py "$imgFolderName" "$bboxFolderName" "$imgFilePrefix" "$preProcessedImgSaveDirName" "$imgNum" "$imgNumPlusPlus"
	fi
done

