# Sytorch

This GitHub repository contains a script that runs a secure Multi-Party Computation (MPC) model to process an image. The script requires certain arguments to be set in order to run correctly.

## Prerequisites
Before running the script, ensure that you have the following:
- The MPC model file in ONNX format
- The image file to be processed in JPG format
- The preprocess.py file to preprocess the image
- The IP address of the server
- Further server and client IP should be whitelisted so they cam communicate over TCP/IP Protocol.

We require the below packages to run OnnxBridge.
- onnx==1.12.0
- onnxruntime==1.12.1
- onnxsim==0.4.8
- numpy==1.21.0
- protobuf==3.20.1
- torchvision==0.13.1
- idx2numpy==1.2.3

Above dependencies can be installed using the [requirements.txt](OnnxBridge/requirements.txt) file as below:
```bash
pip3 install -r OnnxBridge/requirements.txt
```

## Required Arguments
The script requires the following arguments to be set:
- `MODEL_PATH`: the full path to the ONNX MPC model file
- `IMAGE_PATH`: the full path to the input image file
- `PREPROCESS`: the full path to the preprocess.py file
- `SERVER_IP`: the IP address of the server

If any of these arguments are not set, the script will display an error message and exit.

## Optional Arguments
The script also supports the following optional arguments:
- `-b <backend>`: the MPC backend to use (default: `LLAMA`)
- `-scale <scale>`: the scaling factor for the model input (default: `15`)
- `-bl <bitlength>`: the bitlength to use for the MPC computation (default: `40`)

## Running the Script
To run the script, use the following command:
```bash
./ezpc-cli.sh -m <full-path/model.onnx> -preprocess <full-path/preprocess_image_file> -s <server-ip> -i <full-path/image>
```
The above script only works to generate steps for server and client in form of a bash script,
 which can be then run on two VM having Server and Client files respectively.
This script generates :
- server.sh -> For server machine and can be run directly using ```./server.sh```.
- ```server.sh``` also generates ```client_model.zip``` which needs to be sent to client VM in the same folder where ```client.sh``` will be executed.
- client.sh -> For client machine and can be run directly using ```./client.sh```.
