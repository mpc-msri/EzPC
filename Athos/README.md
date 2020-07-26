# Introduction
This folder contains the code for Athos - an end-to-end compiler from TensorFlow to a variety of secure computation protocols.

# Requirements/Setup 
Below we list some of the packages required to get Athos up and running. This is a non-exhaustive list and we only mention the packages on which the system has been tested so far.
- axel: download manager used to download pre-trained models `sudo apt-get install axel`.
- python3.6
- TensorFlow 1.11
- Numpy

Athos also makes use of the EzPC compiler internally (please check `../EzPC/README.md` for corresponding dependencies).

# Directory structure
The codebase is organized as follows:
- `HelperScripts`: This folder contains numerous helper scripts which help from automated setup of ImageNet/CIFAR10 dataset to finding accuracy from output files. Please refer to each of the scripts for further instructions on how to use them.
- `Networks`: This folder contains the code in TensorFlow of the various benchmarks/networks we run in CrypTFlow. Among other networks, it includes code for ResNet, DenseNet, SqueezeNet for ImageNet dataset, SqueezeNet for CIFAR10 dataset, Lenet, Logistic Regression, and a chest x-ray demo network.
- `SeeDot`: This contains code for SeeDot, a high-level intermediate language on which Athos performs various optimizations before compiling to MPC protocols.
- `TFCompiler`: This contains python modules which are called from the TensorFlow code for the dumping of TensorFlow metadata (required by Athos for compilation to MPC protocols).
- `TFEzPCLibrary`: This contains library code written in EzPC for the TensorFlow nodes required during compilation.
- `CompileTF.sh`: The Athos compilation script. Try `./CompileTF.sh --help` for options.
- `CompileTFGraph.sh`: The Athos compilation script for protobuf models. Try `./CompileTFGraph.sh --help` for options.
- `Paths.config`: This can be used to override the default folders for EzPC and Porthos.
- `CompilerScripts`: This folder contains scripts used for processing and compiling dumped models.

# Usage
Here we provide an example on how to use Athos to compile TensorFlow based ResNet-50 code to Porthos semi-honest 3PC protocol and subsequently run it. The relevant TensorFlow code for ResNet-50 can be found in `./Networks/ResNet/ResNet_main.py`.
- Refer to `./Networks/ResNet/README.md` for instructions on how to download and extract the ResNet-50 pretrained model from the official TensorFlow model page.
- `cd ./Networks/ResNet && python3 ResNet_main.py --runPrediction True --scalingFac 12 --saveImgAndWtData True && cd -`
Runs the ResNet-50 code written in TensorFlow to dump the metadata which is required by Athos for further compilation. 
This command execution should result in 2 files which will be used for further compilation - `./Networks/ResNet/graphDef.mtdata` and `./Networks/ResNet/sizeInfo.mtdata`. In addition, the image and the model are also saved in fixed-point format, which can be later input into the compiled code - `./Networks/ResNet/ResNet_img.inp` which contains the image and `./Networks/ResNet/ResNet_weights.inp` which contains the model.
- The next step is to perform the compilation itself. The compilation script internally makes use of the `ezpc` executable. So, before continuing please ensure that you have built `ezpc` (please check the `../EzPC/README.md` for further instructions on that).
- Once EzPC has been built, run this to compile the model to Porthos - `./CompileTF.sh -b 64 -s 12 -t PORTHOS -f ./Networks/ResNet/ResNet_main.py`. This should result in creation of the file - `./Networks/ResNet/ResNet_main_64_porthos0.cpp`.
- `cp ./Networks/ResNet/ResNet_main_64_porthos0.cpp ../Porthos/src/main.cpp`
Copy the compiled file to Porthos.
- `cd ../Porthos && make clean && make -j` 
- Finally run the 3 parties. Open 3 terminals and run the following in each for the 3 parties.
`./party0.sh < ../Athos/Networks/ResNet/ResNet_img.inp` ,
`./party1.sh < ../Athos/Networks/ResNet/ResNet_weights.inp` ,
`./party2.sh`.
Once the above runs, the final answer for prediction should appear in the output of party0, the client inputting the image. For the sample image, this answer should be 249 for ResNet and 248 for DenseNet/SqueezeNet.

Instructions on how to run the particular TensorFlow model in `./Networks` can vary. Please refer to the appropriate readme in each model folder to get more insights. But once that is done, the further compilation commands are the same.

# Preprocessing images and running inference on ImageNet validation dataset
- First setup the ImageNet validation dataset using the script provided in `./HelperScripts/Prepare_ImageNet_Val.sh`. This sets up the ImageNet validation dataset in the folder - `./HelperScripts/ImageNet_ValData`.
- Each of the network folders - `./Networks/ResNet`, `./Networks/DenseNet` and `./Networks/SqueezeNetImgNet` is provided with these folders:
	* `PreProcessingImages`: This folder contains code for preprocessing the images. Code borrowed from the appropriate repository from where the model code is taken
and modified for our purposes (check the apt network folder for more details on the source of the model).
	* `AccuracyAnalysisHelper`: This contains further scripts for automating ImageNet dataset preprocessing and inference. Check the apt scripts for more information.
