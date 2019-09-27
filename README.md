# CrypTFlow: An End-to-end System for Secure TensorFlow Inference.
#### Nishant Kumar, Mayank Rathee, Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma

**CrypTFlow** paper link: [eprint](https://eprint.iacr.org/2019/1049), [arXiv](https://arxiv.org/abs/1909.07814).

**EzPC** paper link: [eprint](https://eprint.iacr.org/2017/1109.pdf)

Project webpage: https://aka.ms/ezpc.

[![Build Status](https://travis-ci.org/mayank0403/EzPC.svg?branch=master)](https://github.com/mpc-msri/EzPC)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mpc-msri/EzPC/issues)

[![HitCount](http://hits.dwyl.io/mpc-msri/EzPC.svg)](http://hits.dwyl.io/mpc-msri/EzPC)
![GitHub contributors](https://img.shields.io/github/contributors/mpc-msri/EzPC)
![GitHub All Releases](https://img.shields.io/github/downloads/mpc-msri/EzPC/total)

![GitHub repo size](https://img.shields.io/github/repo-size/mpc-msri/EzPC)
![GitHub language count](https://img.shields.io/github/languages/count/mpc-msri/EzPC)
![Maintenance](https://img.shields.io/maintenance/yes/2019)


![GitHub search hit counter](https://img.shields.io/github/search/mpc-msri/EzPC/ezpc)
![GitHub search hit counter](https://img.shields.io/github/search/mpc-msri/EzPC/cryptflow)
![GitHub issues](https://img.shields.io/github/issues/mpc-msri/EzPC)


## Introduction
**CrypTFlow** consists of 3 components:
- **Athos**: an end-to-end compiler from TensorFlow to a variety of semi-honest MPC protocols. Athos leverages EzPC as a low-level intermediate language.
- **Porthos**: a semi-honest 3 party computation protocol which is geared towards TensorFlow-like applications.
- **Aramis**: a novel technique that uses hardware with integrity guarantees to convert any semi-honest MPC protocol into an MPC protocol that provides malicious security.

With these components in place, we are able to run for the first time secure inference on the [ImageNet dataset]([http://www.image-net.org) with the pre-trained models of the following deep neural nets: ResNet-50, DenseNet-121 and SqueezeNet for ImageNet.

In particular, this repository contains the code for the following components:
- **EzPC**
- **Athos** (part of **CrypTFlow**)
- **Porthos** (part of **CrypTFlow**)
- **Aramis** (part of **CrypTFlow**)

Each one of the above is independent and usable in their own right and more information can be found in the readme of each of the components. But together these combine to make **CrypTFlow** a powerful system for end-to-end secure inference of deep neural networks written in TensorFlow.

For setup instructions, please refer to each of the components' readme. We plan to release a docker version of the system as well which will make the system easier to setup.

## Wiki
Wiki section of this repository provides coding practices and examples to get started with EzPC.

## Issues/Bugs
For bugs and support, please create an issue on the issues page.
