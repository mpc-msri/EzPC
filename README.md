# CrypTFlow: An End-to-end System for Secure TensorFlow Inference [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mpc-msri/EzPC/issues)

**Reference Papers:**  

[SecFloat: Accurate Floating-Point meets Secure 2-Party Computation](https://eprint.iacr.org/2022/322)  
Deevashwer Rathee, Anwesh Bhattacharya, Rahul Sharma, Divya Gupta, Nishanth Chandran, Aseem Rastogi  
*IEEE S&P 2022*

[SIRNN: A Math Library for Secure RNN Inference](https://eprint.iacr.org/2021/459)  
Deevashwer Rathee, Mayank Rathee, Rahul Kranti Kiran Goli, Divya Gupta, Rahul Sharma, Nishanth Chandran, Aseem Rastogi  
*IEEE S&P 2021*

[CrypTFlow2: Practical 2-Party Secure Inference](https://eprint.iacr.org/2020/1002)  
Deevashwer Rathee, Mayank Rathee, Nishant Kumar, Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma  
*ACM CCS 2020*

[CrypTFlow: Secure TensorFlow Inference](https://eprint.iacr.org/2019/1049)  
Nishant Kumar, Mayank Rathee, Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma  
*IEEE S&P 2020*

[EzPC: Programmable, Efficient, and Scalable Secure Two-Party Computation for Machine Learning](https://eprint.iacr.org/2017/1109.pdf)  
Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma, Shardul Tripathi  
*IEEE EuroS&P 2019*

**Project webpage:** <https://aka.ms/ezpc>

## Introduction
This repository has the following components:  

- **EzPC**: a language for secure machine learning.
- **Athos** (part of **CrypTFlow**): an end-to-end compiler from TensorFlow to a variety of semi-honest MPC protocols. Athos leverages EzPC as a low-level intermediate language.
- **SIRNN**: an end-to-end framework for performing inference over quantized RNN models using semi-honest 2-party computation protocols.
- **Porthos** (part of **CrypTFlow**): a semi-honest 3 party computation protocol which is geared towards TensorFlow-like applications.
- **Aramis** (part of **CrypTFlow**): a novel technique that uses hardware with integrity guarantees to convert any semi-honest MPC protocol into an MPC protocol that provides malicious security.
- **SCI** (part of **CrypTFlow2**, **SIRNN** and **SecFloat**): a semi-honest 2-party computation library for secure (fixed-point) inference on deep neural networks and secure floating-point computation.

Each one of the above is independent and usable in their own right and more information can be found in the readme of each of the components. But together these combine to make **CrypTFlow** a powerful system for end-to-end secure inference of deep neural networks written in TensorFlow.

With these components in place, we are able to run for the first time secure inference on the [ImageNet dataset]([http://www.image-net.org) with the pre-trained models of the following deep neural nets: ResNet-50, DenseNet-121 and SqueezeNet for ImageNet. For an end-to-end tutorial on running models with CrypTFlow please refer to this [blog post](https://pratik-bhatu.medium.com/privacy-preserving-machine-learning-for-healthcare-using-cryptflow-cc6c379fbab7).

## Setup
For setup instructions, please refer to each of the components' readme.

Alternatively you can use the **setup_env_and_build.sh** script. It installs dependencies and builds each component. It also creates a virtual environment in a *mpc_venv* folder with all the required packages. If you want to do setup with default paths and settings do ``./setup_env_and_build.sh quick``, otherwise if you want to manually choose paths you can use ``./setup_env_and_build.sh``.

Please do ``source mpc_venv/bin/activate`` before using the toolchain.

## Secure AI Validation

To setup the repo with modified SCI build such that only secret shares are revealed at the end of 2PC, run the setup script as ``./setup_env_and_build.sh quick NO_REVEAL_OUTPUT``.
Alternatively, just rebuild SCI. For instructions to build modified SCI, see README for SCI.

To build docker image for Secure AI Validation, use the `Dockerfile_AI_Validation` dockerfile.

```docker build -t ezpc_modified - < path/to/EzPC/Dockerfile_AI_Validation```


### Docker
You can use a pre-built docker image from docker hub using ``docker pull ezpc/ezpc:latest``. We occasionally push stable images to that channel. However, if you want a docker image with the latest code, you can build it yourself using:

```docker build -t ezpc_image - < path/to/EzPC/Dockerfile```

## Wiki
Wiki section of this repository provides coding practices and examples to get started with EzPC.

## Issues/Bugs
For bugs and support, please create an issue on the issues page.


