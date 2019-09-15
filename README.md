# CrypTFlow: an end-to-end system for secure TensorFlow inference.
#### Nishant Kumar, Mayank Rathee, Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma

## Introduction
CrypTFlow consists of 3 components:
- Athos: an end-to-end compiler from TensorFlow to a variety of semi-honest MPC protocols. Athos leverages EzPC as a low-level intermediate language.
- Porthos: a semi-honest 3 party computation protocol which is geared towards TensorFlow-like applications.
- Aramis: a novel technique that uses hardware with integrity guarantees to convert any semi-honest MPC protocol into an MPC protocol that provides malicious security.

With these components in place, we are able to run for the first time secure inference on the [ImageNet dataset]([http://www.image-net.org) with the pre-trained models of the following deep neural nets: ResNet-50, DenseNet-121 and SqueezeNet for ImageNet.

In particular, this repository contains the code for the following components:
- Athos
- EzPC
- Porthos

Each one of the above is independent and usable in their own right and more information can be found in the readme of each of the components. But together these combine to make CrypTFlow a powerful system for end-to-end secure inference of deep neural networks written in TensorFlow.

For setup instructions, please refer to each of the components' readme. We plan to release a docker version of the system as well which will make the system easier to setup.

For bugs and support, please create an issue on the issues page.
