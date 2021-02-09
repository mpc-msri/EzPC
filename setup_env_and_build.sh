#!/bin/bash

# Authors: Pratik Bhatu.

# Copyright:
# Copyright (c) 2021 Microsoft Research
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

sudo add-apt-repository ppa:deadsnakes/ppa
sudo add-apt-repository ppa:avsm/ppa
sudo apt update
sudo apt install -y build-essential make cmake libgmp-dev libglib2.0-dev libssl-dev libboost-all-dev m4 python3.7 opam


sudo apt install unzip bubblewrap
sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)
# environment setup
opam init
eval `opam env`
# install given version of the compiler
opam switch create 4.10.0
eval `opam env`
# check if we got what we wanted
which ocaml
ocaml -version
opam install -y Stdint
opam install -y menhir
opam install -y ocamlbuild 
opam install -y ocamlfind

#Virtual environment
sudo apt install -y python3.7-venv
python3.7 -m venv mpc_venv
source mpc_venv/bin/activate
pip install -U pip
pip install tensorflow==1.15.0 keras==2.3.0 scipy==1.1.0 matplotlib
pip install pytest pytest-cov 

# Now we build all the components.
ROOT="$(pwd)"
#Build Ezpc
cd EzPC/EzPC
eval `opam env`
make
#Build Porthos 
cd $ROOT/Porthos
./setup-eigen.sh
mkdir -p src/build
cd src/build
cmake ../
make -j
#Build SCI
cd $ROOT/SCI
mkdir -p build
cd build
cmake -DBUILD_NETWORKS=ON ../
make -j
