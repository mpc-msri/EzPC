This folder contains code for the SqueezeNet neural network on the ImageNet dataset. Original source code taken from [this](https://github.com/avoroshilov/tf-squeezenet) github repo and modified for our purposes.

## Setup
- First, download the pretrained model:
`axel -a -n 5 -c --output ./PreTrainedModel https://github.com/avoroshilov/tf-squeezenet/raw/master/sqz_full.mat`
- The TensorFlow model requires scipy v1.1.0. To install the same:
`pip3 install scipy==1.1.0`
- Run the squeezenet model to dump metadata:
`python3 squeezenet_main.py --in ./SampleImages/n02109961_36.JPEG --saveTFMetadata True`
- Run the squeezenet model to dump the pre-trained model and image in a format which can later be understood by Porthos.
`python3 squeezenet_main.py --in ./SampleImages/n02109961_36.JPEG --scalingFac 12 --saveImgAndWtData True` 
