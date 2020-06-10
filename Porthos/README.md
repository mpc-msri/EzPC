# Introduction
This folder contains code for Porthos - a semi-honest 3 party secure computation protocol.

# Setup
(Ubuntu): 
* sudo apt-get install libssl-dev
* sudo apt-get install g++
* sudo apt-get install make
* sudo apt-get install cmake

# Running the protocol
- First setup Eigen library, used for fast matrix multiplication by Porthos, by running `./setup-eigen.sh`.
- Currently the codebase contains precompiled code for the following 3 neural networks: ResNet-50, DenseNet-121 and SqueezeNet for ImageNet, checked into the following folder: `./src/example_neural_nets`.
- To compile, do the following (from Porthos root):
```
cd src/
mkdir build
cd build
cmake ..
make -j
```
- To run the MPC code for SqueezeNet, ResNet50 or DenseNet121, from Porthos root, use the following commands: (example shown for ResNet50)
`./party0.sh ResNet50 < ../Athos/Networks/ResNet/ResNet_img.inp`,
`./party1.sh ResNet50 < ../Athos/Networks/ResNet/ResNet_weights.inp`, and
`./party2.sh ResNet50`.
The above commands make use of fixed-point input files generated from Athos. Please refer to the `README.md` of Athos for instructions on how to generate the same. Also, note that in the scenario of secure inference, `party0.sh` represents the client, which inputs the image, `party1.sh` represents the server, which inputs the model and `party2.sh` represents the helper party which doesn't have any input. The output is learned by `party0`, which represents the client.

# External Code
- `basicSockets` files contain code writen by Roi Inbar. We have modified this code for crypTFlow.
- `tools.cpp` contains some code written by Anner Ben-Efraim and Satyanarayana. This code is majorly modified for crypTFlow.

# Dependencies
- Eigen library: Used in CrypTFlow for faster matrix multiplication. Can be installed by running `./setup-eigen.sh` as mentioned in **Running the protocol** section of this README.
- OpenSSL: Used in CrypTFlow for AES PRG calls, utilizing AES_NI. To see if your CPU supports AES-NI instruction set, run: `cpuid | grep -i aes`. You will see the AES NI supported `true` or `false` flag for each core in your CPU. This was already installed if you ran the instructions mentioned in the **Setup** section of this README (`sudo apt-get install libssl-dev`).

# Notes
- If porthosSecretType != uint64_t, porthosSecretType multiplication function won't work.
- Possible instructions for installation of libssl on Mac, though untested.
-- brew install libssl
-- cd /usr/local/include
-- ln -s ../opt/openssl/include/openssl . 
Also, use the second build command for Mac to be safe. Read the Caveats part of brew install instructions, since Apple has deprecated the use of OpenSSL.

- When extracting bits out of \_\_m128i, val[0] corresponds to the LSB. The set functions of \_\_m128i take the first argument as val[0].

- Matrix multiplication assembly code only works for Intel C/C++ compiler. The non-assembly code has correctness issues when both the multiplicands are large uint64_t's

# Acknowledgement
This codebase is based on an early fork of [this github repo](https://github.com/snwagh/securenn-public).
