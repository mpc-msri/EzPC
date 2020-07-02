This folder contains TensorFlow code for other benchmarks implemented for CrypTFlow. 
In particular, it contains the following networks:
- *MiniONN on CIFAR*: This is the neural network over the CIFAR-10 dataset used in the MiniONN paper. Implementation is our own, but network description is taken from fig. 13 of [MiniONN paper](https://eprint.iacr.org/2017/452.pdf).
- *ResNet32 on CIFAR100*: This neural network works for the ResNet32 architecture over the CIFAR100 dataset. Model implementation taken from here: https://github.com/mc2-project/delphi/blob/master/python/resnet/resnet32_model.py and adapted slightly for compilation by Athos.

## Setup
Run for example, MiniONN_CIFAR.py as `python3 MiniONN_CIFAR.py`. This will dump the TensorFlow metadata required by Athos for further compilation. Same for other networks.
