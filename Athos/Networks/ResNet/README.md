- Source of Pre-trained model: https://github.com/tensorflow/models/tree/master/official/r1/resnet -- Resnet-50 v2, fp32, Acc 76.47%, (NHWC)
- Download and extract the pretrained model from the above webpage and place the same in `./PreTrainedModel` folder.
`axel -a -n 5 -c --output ./PreTrainedModel http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz`
- Next extract the same:
`cd PreTrainedModel && tar -xvzf resnet_v2_fp32_savedmodel_NHWC.tar.gz && cd ..`
- To run the ResNet-50 model to dump the metadata, use the following commands:
`python3 ResNet_main.py --runPrediction True --scalingFac 12 --saveImgAndWtData True`