# Sytorch

Sytorch is a frontend (like pytorch) for secure machine-learning which can support multiple crypto-backends. Currently it supports inference tasks and includes LLAMA and Cleartext (no crypto) as backends. Sytorch allows users to describe machine learning models in C++ using a pytorch like API. It also supports conversion of ONNX models into sytorch using OnnxBridge.

## Dependencies
Sytorch requires Eigen3, cmake and a C++ compiler with OpenMP enabled. 
```bash
sudo apt update
sudo apt install libeigen3-dev cmake build-essential git
```
To use Sytorch with OnnxBridge, OnnxBridge's python depenedencies need to be installed using the [requirements.txt](OnnxBridge/requirements.txt) file using the command:
```bash
pip3 install -r OnnxBridge/requirements.txt
```

## Quick start using OnnxBridge

Given an model onnx file, OnnxBridge can be used to generate an executable which can be run on two VMs, server and client (owning the model weights and input image respectively), to get the secure inference output. 

To do this two scripts are available:
1. [Single Inference](/sytorch/Toy%20example-%20single%20inference.md) - This script is ideal for a single inference scenario.
2. [Multiple Inference](/sytorch/Toy%20example-%20multiple%20inference.md) - This script is ideal for multiple inference usecases.


