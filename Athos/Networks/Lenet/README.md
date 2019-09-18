This folder contains code for the Lenet network. Original source code was taken from TensorFlow tutorial page ([here](https://web.archive.org/web/20170610053149/https://www.tensorflow.org/get_started/mnist/pros) is an archived version of the webpage) and modified for our purposes.

The folder contains 2 versions of Lenet - small and large. The architecture remains same in both the networks, but the sizes of the layers vary between the two.

## Setup
- To run training for example for the Lenet-Small network, execute the following: `python3 lenetSmall_mnist_train.py`.
- Subsquently to run inference, use this: `python3 lenetSmall_mnist_inference.py 1`, where `1` can be replaced by apt image number of MNIST. This command also dumps the TensorFlow metadata required for further compilation.
