# CrypTFlow: An End-to-end System for Secure TensorFlow Inference [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mpc-msri/EzPC/issues) [![HitCount](http://hits.dwyl.io/mpc-msri/EzPC.svg)](http://hits.dwyl.io/mpc-msri/EzPC)

**Reference Papers:**  

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
- **Porthos** (part of **CrypTFlow**): a semi-honest 3 party computation protocol which is geared towards TensorFlow-like applications.
- **Aramis** (part of **CrypTFlow**): a novel technique that uses hardware with integrity guarantees to convert any semi-honest MPC protocol into an MPC protocol that provides malicious security.
- **SCI** (part of **CrypTFlow2**): a semi-honest 2-party computation library for secure inference on deep neural networks.

Each one of the above is independent and usable in their own right and more information can be found in the readme of each of the components. But together these combine to make **CrypTFlow** a powerful system for end-to-end secure inference of deep neural networks written in TensorFlow.

With these components in place, we are able to run for the first time secure inference on the [ImageNet dataset]([http://www.image-net.org) with the pre-trained models of the following deep neural nets: ResNet-50, DenseNet-121 and SqueezeNet for ImageNet.

For setup instructions, please refer to each of the components' readme.

Alternatively you can use the **setup_env_and_build.sh** script. It installs dependencies and builds each component. It also creates a virtual environment in a *mpc_venv* folder with all the required packages.

Please do ``source mpc_venv/bin/activate`` before using the toolchain.

We plan to release a docker version of the system as well which will make the system easier to setup.

## Wiki
Wiki section of this repository provides coding practices and examples to get started with EzPC.

## Issues/Bugs
For bugs and support, please create an issue on the issues page.
