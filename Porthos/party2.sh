#!/bin/sh

if [ $# -lt 1 ]
then
	echo "Please specify the binary to run."
	echo "Usage: ./party2.sh [SqNetImgNet/ResNet50/DenseNet121]"
	exit 1

elif [ $# -gt 1 ]
then
	echo "Incorrect usage. Only requires one argument."
	echo "Usage: ./party2.sh [SqNetImgNet/ResNet50/DenseNet121]"
	exit 1
fi

./src/build/bin/$1 2 files/addresses


