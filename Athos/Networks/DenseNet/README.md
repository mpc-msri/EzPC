- Source of Pretrained model: https://github.com/pudae/tensorflow-densenet.
Download the DenseNet-121 pretrained model from [link](https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA) given in the above website and place the same in the `PreTrainedModel` folder. This should be a file of the name `tf-densenet121.tar.gz`.
- Extract the model: `cd PreTrainedModel && tar -xvzf tf-densenet121.tar.gz && cd - `
- Command to run the TensorFlow DenseNet model to dump the metadata:
`python3 DenseNet_main.py --runPrediction True --scalingFac 12 --saveImgAndWtData True`