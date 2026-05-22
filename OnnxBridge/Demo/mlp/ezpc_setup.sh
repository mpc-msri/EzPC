#!/bin/bash

sudo apt update
sudo apt install -y --no-install-recommends libeigen3-dev cmake build-essential git

# upgrade cmake if version < 3.16
wget -O /tmp/cmake.sh wget -O /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh
sudo /bin/sh /tmp/cmake.sh --prefix=/usr/local --skip-license && rm /tmp/cmake.sh
echo $(cmake --version)

git clone https://github.com/mpc-msri/EzPC.git
# git checkout 8b07f73e187c5eb6ab98b0bf09b9bd276cd43949
cd EzPC/
#python3 -m venv venv
#source venv/bin/activate
pip install -r OnnxBridge/requirements.txt