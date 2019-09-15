#!/bin/bash

cifar10DownloadLink="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
axel -a -n 3 -c --output CIFAR10 "$cifar10DownloadLink"
cd CIFAR10
tar -xvzf cifar-10-python.tar.gz --directory=.
