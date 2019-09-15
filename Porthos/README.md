# Introduction
This folder contains code for Porthos - a semi-honest 3 party secure computation protocol.

# Setup
(Ubuntu): 
* sudo apt-get install libssl-dev
* sudo apt-get install g++
* sudo apt-get install make

# Running the protocol
- To compile use `make clean && make -j`.
- Currently the codebase contains precompiled code the following 3 neural networks: ResNet-50, DenseNet-121 and SqueezeNet for ImageNet, checked into the following folder: `./src/example_neural_nets`. Toggle the flag in `./src/example_neural_nets/network_config.h` to switch the network which runs. Note that if there is more than network flag uncommented or if there is already a main file in src, the compilation will error out.
- For example, to run the ResNet-50 code, use the following commands:
`./party0.sh < ../Athos/Networks/ResNet/ResNet_img_input.inp`
`./party1.sh`
`./party2.sh`

Note that the first command makes use of input generated from Athos. Please refer to the Athos readme on how to do the same. 

# Notes
- If porthosSecretType != uint64_t, porthosSecretType multiplication function won't work.
- Possible instructions for installation of libssl on Mac, though untested.
* brew install libssl
* cd /usr/local/include
* ln -s ../opt/openssl/include/openssl . 
Also, use the second build command for Mac to be safe. Read the Caveats part of brew install instructions, since Apple has deprecated the use of OpenSSL.

- libmiracl.a is compiled locally, if it does not work try the libmiracl_old.a or download the source files from https://github.com/miracl/MIRACL.git and compile miracl.a yourself (and rename to libmriacl.a when copying into this repo)

- When extracting bits out of \_\_m128i, val[0] corresponds to the LSB. The set functions of \_\_m128i take the first argument as val[0].

- If porthosSecretType != uint64_t, porthosSecretType multiplication function won't work.

- Matrix multiplication assembly code only works for Intel C/C++ compiler. The non-assembly code has correctness issues when both the multiplicands are large uint64_t's


