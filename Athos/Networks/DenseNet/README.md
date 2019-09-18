This folder contains code for DenseNet based deep neural network. The original source code for the model and the pretrained network itself is taken from [this github repo](https://github.com/pudae/tensorflow-densenet). The code is modified slightly to make it compatible for compilation by Athos.

## Setup
- Download the DenseNet-121 pretrained model from [this link](https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA) given in the [above github repo](https://github.com/pudae/tensorflow-densenet) and place the same in the `PreTrainedModel` folder. This should be a file of the name `tf-densenet121.tar.gz`.
- Extract the model: `cd PreTrainedModel && tar -xvzf tf-densenet121.tar.gz && cd - `
- Command to run the TensorFlow DenseNet model to dump the metadata:
`python3 DenseNet_main.py --runPrediction True --scalingFac 12 --saveImgAndWtData True`