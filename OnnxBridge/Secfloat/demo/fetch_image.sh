#!/bin/bash
wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" -O mnist/t10k-images-idx3-ubyte.gz
gunzip mnist/t10k-images-idx3-ubyte.gz

python mnist/create_image.py 