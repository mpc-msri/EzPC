# Sytorch

Sytorch is a frontend (like pytorch) for secure machine-learning which can support multiple crypto-backends. Currently it supports inference tasks and includes LLAMA and Cleartext (no crypto) as backends. Sytorch allows users to describe machine learning models in C++ using a pytorch like API. It also supports conversion of ONNX models into sytorch using OnnxBridge.

## Dependencies
Sytorch requires Eigen3, cmake and a C++ compiler with OpenMP enabled. To use Sytorch with OnnxBridge, OnnxBridge's python depenedencies need to be installed using the [requirements.txt](OnnxBridge/requirements.txt) file using the command:
```bash
pip3 install -r OnnxBridge/requirements.txt
```

## Quick start using OnnxBridge

Given an model onnx file, OnnxBridge can be used to generate an executable which can be run on two VMs, server and client (owning the model weights and input image respectively), to get the secure inference output. To do this, use the `ezpc-cli.sh` script by running the following command locally (not neccesarily on a VM):

```bash
./ezpc-cli.sh -m /absolute/path/to/model.onnx -preprocess /absolute/path/to/preprocess.py -s server-ip -i /absolute/path/to/image.jpg
```

In the above command, the paths are not local, but are the locations on the respective VMs. That is, `/absolute/path/to/model.onnx` is the path of model.onnx file on the server VM, `/absolute/path/to/preprocess.py` is the path of preprocessing script on the server VM, and `/absolute/path/to/image.jpg` is the path of image on the client VM. To write the preprocessing script for your use case, refer to the preprocessing file of the [chexpert demo](/Athos/demos/onnx/pre_process.py). If your preprocessing script uses some additional python packages, make sure they are installed on the server and client VMs. Also, ensure that the client can communicate with the server through the IP address provided on the ports between the range 42002-42100. Optionally, you can also pass the following arguments:

- `-b <backend>`: the MPC backend to use (default: `LLAMA`)
- `-scale <scale>`: the scaling factor for the model input (default: `15`)
- `-bl <bitlength>`: the bitlength to use for the MPC computation (default: `40`)

The script generates 4 scripts:

- `server-offline.sh` - Transfer this script to the server VM in any empty directory. Running this script (without any argument) reads the ONNX file, strips model weights out of it, dumps sytorch code, zips the code required to be sent to the client and waits for the client to download the zip. Once the zip is transfered, the script generates the preprocessing key material.
- `client-offline.sh` - Transfer this script to the client VM in any empty directory. Running this script fetches the stripped code from server and generates the preprocessing key material. This script must be run on client VM parallely while server VM is running it's server script. 
- `server-online.sh` - Transfer this script to the server VM in the same directory. Running this script waits for client and starts the inference once the client connects.
- `client-online.sh` - Transfer this script to the client VM in the same directory. Running this script preprocesses the input, connects with the server and starts the inference. After the secure inference is complete, inference output is printed and saved in `output.txt` file.
