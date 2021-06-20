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

mode=$1

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo add-apt-repository ppa:avsm/ppa -y
sudo apt update
sudo apt install -y build-essential cmake libgmp-dev libglib2.0-dev libssl-dev libboost-all-dev m4 python3.7 opam
sudo apt install -y unzip bubblewrap

build_cmake () {
  echo "Building and installing cmake from source"
  wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4.tar.gz
  tar -zxvf cmake-3.13.4.tar.gz
  cd cmake-3.13.4
  sudo ./bootstrap
  sudo make
  sudo make install
  cd ..
  rm -rf cmake-3.13.4 cmake-3.13.4.tar.gz
}

if which cmake >/dev/null; then
  CMAKE_VERSION=$(cmake --version | grep -oE '[0-9]+.[0-9]+(\.)*[0-9]*')
  LATEST_VERSION=$(printf "$CMAKE_VERSION\n3.13\n" | sort | tail -n1)
  if [[ "$CMAKE_VERSION" == "$LATEST_VERSION" ]]; then
    echo "CMake already installed.."
  else
    sudo apt purge cmake
    build_cmake
  fi
else
  build_cmake
fi


wget "https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh"
if [ $? -ne 0 ]; then
  echo "Downloading of opam script failed"; exit
fi

chmod +x install.sh
if [[ "$mode" == "quick" ]]; then
	yes "" | ./install.sh
else
	./install.sh
fi
if [ $? -ne 0 ]; then
  rm install.sh
  echo "Opam installation failed"; exit
fi
rm install.sh

# environment setup
if [[ "$mode" == "quick" ]]; then
	yes "" | opam init
else
	opam init
fi
if [ $? -ne 0 ]; then
  echo "opam init failed"; exit
fi

# install given version of the compiler
eval `opam env`
if [[ "$mode" == "quick" ]]; then
	yes "" | opam switch create 4.10.0
else
	opam switch create 4.10.0
fi
opam switch list | grep "4.10.0" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "opam switch create 4.10.0 failed"; exit
fi
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
pip install onnx onnx-simplifier onnxruntime
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
cmake -DCMAKE_INSTALL_PREFIX=./install ../
cmake --build . --target install --parallel
