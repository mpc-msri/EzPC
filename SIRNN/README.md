# SIRNN: Secure Inference for Recurrent Neural Networks

SIRNN is an application of the SCI MPC protocols for inference over RNNs. This README discusses the inference over [FastGRNN](https://github.com/microsoft/EdgeML/blob/master/docs/publications/FastGRNN.pdf) models.  

## Setup

For setting up SIRNN, set-up:
1. [SCI](https://github.com/mpc-msri/EzPC/tree/master/SCI)
2. [EzPC](https://github.com/mpc-msri/EzPC/tree/master/EzPC)
3. [SeeDot](https://github.com/microsoft/EdgeML/tools/SeeDot)

## Control Flow

SIRNN performs inference on quantized machine learning models. 
This quantization is carried out using the [SeeDot](https://www.microsoft.com/en-us/research/publication/shiftry-rnn-inference-in-2kb-of-ram/) framework.
**Seedot** generates an output that is in the [EzPC](https://www.microsoft.com/en-us/research/project/ezpc-easy-secure-multi-party-computation/) language.
The EzPC compiler is then used to translate the code into SIRNN. 

SIRNN code is essentially a sequence of function calls to the [SCI](https://www.microsoft.com/en-us/research/publication/cryptflow2-practical-2-party-secure-inference/)  library. 

SIRNN Summary: SeeDot->EzPC->SCI.

## Running Instructions

This section provided a step-by-step runthrough of **SIRNN** using the **FastGRNN** model and the [*Google-30*](https://arxiv.org/abs/1804.03209) dataset. 

### Step 1: Training FastGRNN and Quantizing using SeeDot

1. #### Clone EdgeML and checkout sirnn branch:

```
git clone https://github.com/microsoft/EdgeML.git
git checkout sirnn

```

2. #### Follow the instructions in **SeeDot**'s [README](https://github.com/microsoft/EdgeML/blob/master/tools/SeeDot/README.md) to install SeeDot's dependencies. 


3. #### Download [Google-30](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz) dataset. 

```
cd EdgeML/examples/tf/FastCells/
mkdir Google-30/ && cd Google-30/
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

tar -xvf  speech_commands_v0.01.tar.gz
cd ../
```

4. #### Process the dataset:

```
python process_g30.py -d Google-30/ -s Google-30/

```

5. #### Train FastGRNN model:

```
python fastcell_example.py -id 32 -hd 100 -rW 16 -rU 35 -sU 0.2 -sW 0.2 -dir Google-30/

```

6. #### Quantize using SeeDot and generate input:

```
cd ../../../tools/SeeDot/
mkdir -p model/fastgrnn/Google-30/
mkdir -p datasets/fastgrnn/Google-30/

cp -r ../../examples/tf/FastCells/Google-30/FastGRNNResults/<timestamp>/* model/fastgrnn/Google-30/

cp -r ../../examples/tf/FastCells/Google-30/train.npy datasets/fastgrnn/Google-30/
cp -r ../../examples/tf/FastCells/Google-30/test.npy datasets/fastgrnn/Google-30/

python fixSeeDotInput.py --seedot_file seedot/compiler/input/fastgrnn-g30.sd --model_dir model/fastgrnn/Google-30/ --dataset_dir datasets/fastgrnn/Google-30/ -n 1 --normalise_data

python SeeDot-dev.py -a fastgrnn -e fixed -d Google-30 -m red_disagree -t EzPC -n 1

export SEEDOT_DIR=$(pwd) # For accessing the files in the following steps
```
##### Warning: The above SeeDot run may fail in some devices due to a large number of threads being created. For resolution, reduce the number of rows in `datasets/fastgrnn/Google-30/train.npy`.

### Step 2: Generating SIRNN code using EzPC

1. #### Clone EzPC repo: 

```
git clone https://github.com/mpc-msri/EzPC.git
```

2. #### Install **EzPC**'s dependencies and initialise OPAM environment.

3. #### Set directory environment variable for EzPC:

```
cd EzPC/EzPC/EzPC/

# Build EzPC compiler, if not already done.
make 

# The directory from where EzPC can be run.
export EZPC_DIR=$(pwd) 

# Folder for storing EzPC's output when run on SeeDot generated EzPC code. 

# This can be any convenient location
mkdir -p sirnn/  



```

4. #### Compile SeeDot generated EzPC code using the following steps:

```
cd ../../SIRNN/
mkdir FastGRNN/

python preProcessSIRNN.py --seedot_dir ${SEEDOT_DIR} --results_dir FastGRNN/ --predict_dir ${EZPC_DIR}/sirnn --dataset Google-30 --sci_build_location FastGRNN/build
python secureCodegen.py --predict_dir ${EZPC_DIR}/sirnn --ezpc_dir ${EZPC_DIR} --results_dir FastGRNN/Google-30/ 

mkdir -p FastGRNN/build/

```

Explanation:
    `preProcessSIRNN.py` script creates a folder with name 'dataset' in 'results_dir' folder. Then copies the SeeDot generated files to the 'dataset' folder. 
    It copies the predict.ezpc file to a location indicated by 'predict_dir'.
    `sci_build_location` is the path that was specified for installing the SCI library. (If the location is not included within the `$PATH` environment variable.)
    `securecppcodegne.py`  script runs the EzPC compiler on predict.ezpc, specified by 'predict_dir', and stored the output in 'sirnn_fixed.cpp' at 'results_dir'.

Note: The keywords in '' indicate the argument placeholders to the python scripts. 

### Step 3: Inference using SCI

1. #### Install SCI using the instructions in [README](https://github.com/mpc-msri/EzPC/tree/master/SCI).
    While installing SCI, the library should be installed to a location on the `$PATH` environment variable.
    It is preferable to use `-DCMAKE_INSTALL_PREFIX` flag and install the libraries to the location indicated by 'sci_build_location' [above](https://github.com/mpc-msri/EzPC/blob/new-SCI/SIRNN/README.md#compile-seedot-generated-ezpc-code-using-the-following-steps). 
    

2. #### Compile SIRNN code

```
cd FastGRNN/build/
cmake .. # If compiling for the first time
make

```

3. #### Running SIRNN:

```
# For SERVER
./bin/Google-30 r=1 addr=0.0.0.0 port=8000 nt=4 inputDir=../Google-30/input/

# For CLIENT
./bin/Google-30 r=2 addr=0.0.0.0 port=8000 nt=4 inputDir=../Google-30/input/

```


Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
