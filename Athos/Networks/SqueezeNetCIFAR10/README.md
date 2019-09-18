This folder contains code for the SqueezeNet network on the CIFAR10 dataset. Original source code of model is taken from [this](https://github.com/kaizouman/tensorsandbox/tree/master/cifar10/models/squeeze) github repo. For preprocessing the code is based on [this github repo](https://github.com/deep-diver/CIFAR10-img-classification-tensorflow). Both the model and preprocessing code were further modified for our purposes.

## Setup
- Setup CIFAR10 dataset first using the following command: `cd ../../HelperScripts && ./SetupCIFAR10.sh && cd -`. This sets up the CIFAR10 dataset in this folder:`../../HelperScripts/CIFAR10/`.
- Next run `python3 Util.py`, which will preprocess the CIFAR10 dataset images as required by the model.
- Run `python3 Squeezenet_model.py train` to train the model for 1 epoch. This will place the trained model in `./TrainedModel` folder.
- Run the following command to dump the TensorFlow metadata required by Athos for further compilation: `python3 Squeezenet_model.py savegraph`.
- Finally, run this command to save the image and model in a format which can be used later by Porthos: `python3 Squeezenet_model.py testSingleTestInpAndSaveData 1 1`. This will save a file by the name `SqNet_CIFAR_input.inp`.
