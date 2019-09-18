This folder contains code for the ResNet based deep neural network. The network implementation itself as well as the pretrained model is taken from the [official github repository](https://github.com/tensorflow/models/tree/master/official/r1/resnet) of TensorFlow. The code is modified slightly to make it compatible for compilation by Athos.

## Setup
- Download the pretrained model from the [official repository](https://github.com/tensorflow/models/tree/master/official/r1/resnet) of TensorFlow (specifically Resnet-50 v2, fp32, Acc 76.47%, (NHWC)). Run the following to download the pretrained model from the above webpage and place the same in `./PreTrainedModel` folder:
`axel -a -n 5 -c --output ./PreTrainedModel http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz`.
- Next, extract the tar file:
`cd PreTrainedModel && tar -xvzf resnet_v2_fp32_savedmodel_NHWC.tar.gz && cd ..`
- Finally, run the ResNet-50 model to dump the metadata:
`python3 ResNet_main.py --runPrediction True --scalingFac 12 --saveImgAndWtData True`

## Notes
To compile ResNet-n for some n in [18,34,50,101,152,200], modify the parameter to `ImagenetModel()` in `ResNet_main.py::infer()` passing the desired value of n.
