This folder contains code written in TensorFlow for the 4 neural networks that were used as benchmarks in [SecureNN](https://eprint.iacr.org/2018/442).
Implementations are our own, but the network descriptions are taken from elsewhere.

1. *Network A*: This network was first used in [SecureML](https://eprint.iacr.org/2017/396.pdf). The network description is taken from fig. 10 of [MiniONN paper](https://eprint.iacr.org/2017/452.pdf).
2. *Network B*: Network description taken from fig. 12 of [MiniONN paper](https://eprint.iacr.org/2017/452.pdf).
3. *Network C*: Network description taken from the [public implementation](https://github.com/snwagh/securenn-public/blob/master/src/main.cpp) of SecureNN.
4. *Network D*: Network description taken from fig. 3 of [Chameleon paper](https://eprint.iacr.org/2017/1164.pdf]).


## Setup
Run for example, network A as `python3 NetworkA.py`. This will dump the TensorFlow metadata required by Athos for further compilation. Similarly for other networks.
