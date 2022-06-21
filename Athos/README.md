- [Introduction](#introduction)
- [Requirements/Setup](#requirementssetup)
- [Usage](#usage)
  * [Compiling random forest/decision tree](#compiling-random-forestdecision-tree)
  * [Compiling a tensorflow model](#compiling-a-tensorflow-model)
  * [Compiling an ONNX model](#compiling-an-onnx-model)
  * [Compiling and Running Models in Networks Directory Automatically](#compiling-and-running-models-in-networks-directory-automatically)
      - [Running manually (non-tmux-mode)](#running-manually-non-tmux-mode)
  * [Compiling and Running Models in Networks Directory Manually](#compiling-and-running-models-in-networks-directory-manually)
- [Directory structure](#directory-structure)
- [Preprocessing images and running inference on ImageNet validation dataset](#preprocessing-images-and-running-inference-on-imagenet-validation-dataset)

# Introduction
This folder contains the code for Athos - an end-to-end compiler from TensorFlow to a variety of secure computation protocols.

# Requirements/Setup 
If you used the `setup_env_and_build.sh` script the below would already have been installed in the `mpc_venv` environment. We require the below packages to run Athos.
- python3.7
- TensorFlow 1.15
- Numpy
- pytest, pytest-cov (For running tests)
- onnx, onnx-simplifier

Athos also makes use of the EzPC compiler internally (please check `../EzPC/README.md` for corresponding dependencies).

# Usage
Please source the virtual environment in `mpc_venv` if you used the `setup_env_and_build.sh` script to setup and build.

`source mpc_venv/bin/activate`

## Compiling random forest/decision tree
Use `CompileRandomForests.py` according to the readme in [Athos/RandomForests](https://github.com/mpc-msri/EzPC/tree/master/Athos/RandomForests)

## Compiling a tensorflow model
The `CompileTFGraph.py` script can compile tensorflow models (v1.15). You can dump your tensorflow model as a frozen graph. Run [convert_variables_to_constants](https://www.tensorflow.org/api_docs/python/tf/compat/v1/graph_util/convert_variables_to_constants) on your model graph and then dump the output graph_def as a protobuf (see `dump_graph_def_pb` in `CompilerScripts/tf_graph_io.py`). Once you have the model.pb file simply do:
```
python CompileTFGraph.py --config config.json --role server
```
See `python CompileTFGraph.py --help` for additional details on the config.json parameters. A sample config could be:
```
{
  "model_name": "model.pb",
  "output_tensors": [ "output1" ],
  "target": "SCI",
  "backend": "OT",
}
```
You will see the output messages of the compiler and a `model_SCI_OT.out` binary will be generated. You will also see this in the output:
```
Use as input to server (model weights): model_input_weights_fixedpt_scale_12.inp.
Share client.zip file with the client
```
Use the `model_input_weights_fixedpt_scale_12.inp` file as input for the server party. The additional client.zip file contains a version of the model without model weights and additionally contains the config file. This zip file should be sent to the client and after unzipping, they can compile the model with:
```
python CompileTFGraph.py --config model.config --role client
```
For model input you can create a random input using `CompilerScripts/create_tf_input.py` or pass your actual input as a numpy array to the `dumpImageDataInt` function in `TFCompiler/DumpTFMtData.py`. For both scripts you need to pass the scaling factor for conversion of floating point to fixed point (we use 12 for ResNet). Refer to [Running manually (non-tmux-mode)](#running-manually-non-tmux-mode) on how to run the MPC protocol or this [blog post](https://pratik-bhatu.medium.com/privacy-preserving-machine-learning-for-healthcare-using-cryptflow-cc6c379fbab7) for a more detailed walkthrough.

## Compiling an ONNX model
Similar to how we compile tensorflow graphs, we have a `CompileONNXGraph.py` script that can compile onnx models. The usage is exactly the same as the `CompileTFGraph.py` script.

### Supported nodes

Some of the supported nodes in ONNX models are:

```Cast
Pad
Concat
HardSigmoid ( Only in SCI )
Relu
Div
Add
Sub
Mul
Clip ( Only in SCI )
Gather
ArgMax
Gemm
Constant
Transpose
Split
ReduceMean
MatMul
BatchNormalization
Unsqueeze
Reshape
Flatten
Conv
MaxPool
AvgPool
AveragePool
GlobalAveragePool
ConvTranspose
```

There are some also additional limitations in some of the nodes. The compiler will exit with information about the limitation when the model with an unsupported node is compiled.

## Compiling and Running Models in Networks Directory Automatically
The `CompileSampleNetworks.py` script can compile and optionally run models in the Networks directory like ResNet-50, DenseNet, SqueezeNet, etc..
To compile and run ResNet with the Porthos semi-honest 3PC protocol we do:

```python CompileSampleNetworks.py --config Networks/sample_network.config```

The script takes a config file as input. The contents of the config are:
```
{
  "network_name":"ResNet",
  "target":"PORTHOS",
  "run_in_tmux": true
}
```
- *network_name*: Can be any network in the Networks directory. 
- *target*: This is the secure protocol the model will run in. The possible values are:
	- **PORTHOS**: The semi-honest 3PC protocol.
	- **SCI**: The semi-honest 2PC protocol in SCI.
	- **CPP**: A non-secure debug backend which outputs plain C++ to test for correctness.
- *run_in_tmux*: If true, the script spawns a tmux session to run the network. There is a terminal pane for each party.
You can modify the config file according to which network and backend you want to compile for. See ```python CompileSampleNetworks.py --help``` for more information about the parameters of the config file.

**Output:**
After connecting to the session with ```tmux a -t ResNet``` you should see output similar to the following after the computation is complete. Numbers will vary based on the specs of your machine.

| | |
|-|-|
|-------------------------------------------------------<br>                  **ResNet results [Client]**<br>-------------------------------------------------------<br>Model outputs:<br>MPC PORTHOS (3PC) output:        249<br>Tensorflow output:               249<br><br>Execution summary for Client:<br>Communication for execution, P0: 2377.13MB (sent) 1825.32MB (recv)<br>Peak Memory Usage:               432156 KB (.41GB)<br>Total time taken:                70.50 seconds<br>Total work time:                 68.57 seconds (97.27%)<br>Time spent waiting:              1.91 seconds (2.72%)<br>Time taken by tensorflow:        0.31 seconds |  
|-------------------------------------------------------<br>                  **ResNet results [Server]**<br>-------------------------------------------------------<br>Execution summary for Server:<br>Communication for execution, P1: 2377.13MB (sent) 2155.96MB (recv)<br>Peak Memory Usage:               432428 KB (.41GB)<br>Total time taken:                70.60 seconds<br>Total work time:                 68.77 seconds (97.42%)<br>Time spent waiting:              1.81 seconds (2.57%)   |-------------------------------------------------------<br>                  **ResNet results [Helper]**<br>-------------------------------------------------------<br>Execution summary for Helper:<br>Communication for execution, P2: 2113.81MB (sent) 2886.8MB (recv)<br>Peak Memory Usage:               427508 KB (.40GB)<br>Total time taken:                70.53 seconds<br>Total work time:                 64.06 seconds (90.83%) <br>Time spent waiting:              6.46 seconds (9.16%)  
 
#### Running manually (non-tmux-mode)

If run_in_tmux is false and you want to run the network manually, you will need the following files that are generated by the script in the `Networks/ResNet` directory:
- *ResNet_PORTHOS.out*:		binary of the compiled network.
- *model_input_scale_12.inp*: 	image input to the model.
- *model_weights_scale_12.inp*: model weights.

Running it manually will not give you a neat summary as shown above but will print the computed output on the client terminal.

To run the network in **3PC mode (PORTHOS)**, open 3 terminals and do the following for each party:

- Party 0 [Client]:
	
	``` ./Networks/ResNet/ResNet_PORTHOS.out 0 ../Porthos/files/addresses ../Porthos/files/keys < model_input_scale_12.inp" ```
- Party 1 [Server]: 
	
	``` ./Networks/ResNet/ResNet_PORTHOS.out 1 ../Porthos/files/addresses ../Porthos/files/keys < model_weights_scale_12.inp" ```
- Party 2 [Helper]:
	
	``` ./Networks/ResNet/ResNet_PORTHOS.out 1 ../Porthos/files/addresses ../Porthos/files/keys" ```

To run the network in **2PC mode (SCI)**, open 2 terminals and do the following for each party:

- Party 0 [Server]:
	
	``` ./Networks/ResNet/ResNet_SCI_OT.out r=1 p=12345 < model_weights_scale_12.inp" ```
- Party 1 [Client]:
	
	``` ./Networks/ResNet/ResNet_SCI_OT.out r=2 ip=127.0.0.1 p=12345 < model_input_scale_12.inp" ```

To run the network in **CPP mode (1PC-debug-non-secure)**, open a terminal and do the following:
- ``` ./Networks/ResNet/ResNet_CPP.out < <(cat model_input_scale_12.inp model_weights_scale_12.inp) ```

## Compiling and Running Models in Networks Directory Manually
To better understand what the `CompileSampleNetworks.py` script is doing under the hood, we can step through each step manually. Here we provide an example on how to use Athos to compile TensorFlow based ResNet-50 code to Porthos semi-honest 3PC protocol and subsequently run it. The relevant TensorFlow code for ResNet-50 can be found in `./Networks/ResNet/ResNet_main.py`.
- Refer to `./Networks/ResNet/README.md` for instructions on how to download and extract the ResNet-50 pretrained model from the official TensorFlow model page.
- `cd ./Networks/ResNet && python3 ResNet_main.py --runPrediction True --scalingFac 12 --saveImgAndWtData True && cd -`
Runs the ResNet-50 code written in TensorFlow to dump the metadata which is required by Athos for further compilation. 
This command execution should result in 2 files which will be used for further compilation - `./Networks/ResNet/graphDef.mtdata` and `./Networks/ResNet/sizeInfo.mtdata`. In addition, the image and the model are also saved in fixed-point format, which can be later input into the compiled code - `./Networks/ResNet/model_input_scale_12.inp` which contains the image and `./Networks/ResNet/model_weights_scale_12.inp` which contains the model weights.
- The next step is to perform the compilation itself. The compilation script internally makes use of the `ezpc` executable. So, before continuing please ensure that you have built `ezpc` (please check the `../EzPC/README.md` for further instructions on that).
- Once EzPC has been built, run this to compile the model to Porthos - `./CompileTF.sh -b 64 -s 12 -t PORTHOS -f ./Networks/ResNet/ResNet_main.py`. This should result in creation of the file - `./Networks/ResNet/ResNet_main_64_porthos0.cpp`.
- `cp ./Networks/ResNet/ResNet_main_64_porthos0.cpp ../Porthos/src/main.cpp`
Copy the compiled file to Porthos.
- `cd ../Porthos && make clean && make -j` 
- Finally run the 3 parties. Go to the porthos directory and open 3 terminals and run the following in each for the 3 parties.
`./party0.sh < ../Athos/Networks/ResNet/ResNet_img.inp` ,
`./party1.sh < ../Athos/Networks/ResNet/ResNet_weights.inp` ,
`./party2.sh`.
Once the above runs, the final answer for prediction should appear in the output of party0, the client inputting the image. For the sample image, this answer should be 249 for ResNet and 248 for DenseNet/SqueezeNet.

Instructions on how to run the particular TensorFlow model in `./Networks` can vary. Please refer to the appropriate readme in each model folder to get more insights. But once that is done, the further compilation commands are the same.

# Directory structure
The codebase is organized as follows:
- `HelperScripts`: This folder contains numerous helper scripts which help from automated setup of ImageNet/CIFAR10 dataset to finding accuracy from output files. Please refer to each of the scripts for further instructions on how to use them.
- `Networks`: This folder contains the code in TensorFlow of the various benchmarks/networks we run in CrypTFlow. Among other networks, it includes code for ResNet, DenseNet, SqueezeNet for ImageNet dataset, SqueezeNet for CIFAR10 dataset, Lenet, Logistic Regression, and a chest x-ray demo network.
- `SeeDot`: This contains code for SeeDot, a high-level intermediate language on which Athos performs various optimizations before compiling to MPC protocols.
- `TFCompiler`: This contains python modules which are required by Athos for compilation of tensorflow models to MPC protocols.
- `ONNXCompiler`: This contains python modules which are required by Athos for compilation of ONNX models to MPC protocols.
- `TFEzPCLibrary`: This contains library code written in EzPC for the TensorFlow nodes required during compilation.
- `CompileTF.sh`: The Athos compilation script. Try `./CompileTF.sh --help` for options.
- `CompileTFGraph.py`: The Athos compilation script for tensorflow models. Try `python CompileTFGraph.py --help` for options.
- `CompileONNXGraph.py`: The Athos compilation script for ONNX models. Try `python CompileONNXGraph.py --help` for options.
- `Paths.config`: This can be used to override the default folders for EzPC and Porthos.
- `CompilerScripts`: This folder contains scripts used for processing and compiling dumped models.

# Preprocessing images and running inference on ImageNet validation dataset
- First setup the ImageNet validation dataset using the script provided in `./HelperScripts/Prepare_ImageNet_Val.sh`. This sets up the ImageNet validation dataset in the folder - `./HelperScripts/ImageNet_ValData`.
- Each of the network folders - `./Networks/ResNet`, `./Networks/DenseNet` and `./Networks/SqueezeNetImgNet` is provided with these folders:
	* `PreProcessingImages`: This folder contains code for preprocessing the images. Code borrowed from the appropriate repository from where the model code is taken
and modified for our purposes (check the apt network folder for more details on the source of the model).
	* `AccuracyAnalysisHelper`: This contains further scripts for automating ImageNet dataset preprocessing and inference. Check the apt scripts for more information.
