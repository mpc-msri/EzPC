#!/bin/bash

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

