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

# If 2nd argument is provided, then SCI build will be modified. See SCI readme.
NO_REVEAL_OUTPUT=$2

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo add-apt-repository ppa:avsm/ppa -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install -y build-essential cmake libgmp-dev libglib2.0-dev libssl-dev \
                    libboost-all-dev m4 python3.7 opam unzip bubblewrap \
                    graphviz tmux bc time

#Install gcc 9
sudo apt install -y gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo update-alternatives --config gcc

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
	yes "" | opam init --disable-sandboxing
else 
	opam init --disable-sandboxing
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
pip install tensorflow==1.15.0 keras==2.3.0 scipy==1.1.0 matplotlib scikit-learn==0.24.2
pip install onnx onnx-simplifier onnxruntime black
pip install pytest pytest-cov 
python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com


build_boost () {
  sudo apt-get -y install python3.7-dev autotools-dev libicu-dev libbz2-dev
  echo "Building and installing boost from source"
  wget https://boostorg.jfrog.io/artifactory/main/release/1.67.0/source/boost_1_67_0.tar.gz
  tar -zxvf boost_1_67_0.tar.gz
  cd boost_1_67_0
  ./bootstrap.sh
  ./b2 -j $(nproc)
  sudo ./b2 install
  sudo ldconfig
  cd ..
  rm -rf boost_1_67_0.tar.gz boost_1_67_0
}

BOOST_REQUIRED_VERSION="1.66"
if dpkg -s libboost-dev >/dev/null; then
  BOOST_VERSION=$(dpkg -s libboost-dev | grep 'Version' | grep -oE '[0-9]+.[0-9]+(\.)*[0-9]*')
  LATEST_VERSION=$(printf "$BOOST_VERSION\n$BOOST_REQUIRED_VERSION\n" | sort | tail -n1)
  if [[ "$BOOST_VERSION" == "$LATEST_VERSION" ]]; then
    echo "Boost already installed.."
  else
    sudo apt purge libboost-all-dev -y
    build_boost
  fi
else
  build_boost
fi

# Now we build all the components.
ROOT="$(pwd)"
#Build Ezpc
cd EzPC/EzPC
eval `opam env`
make

#Build ABY
git clone --recursive https://github.com/encryptogroup/ABY.git
cd ABY/
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=./install -DABY_BUILD_EXE=On ..
if [ $? -ne 0 ]; then
  echo "ABY cmake command failed. Check error and refer to https://github.com/encryptogroup/ABY for help";
  exit
fi
cmake --build . --target install --parallel
if [ $? -ne 0 ]; then
  echo "ABY build failed. Check error and refer to https://github.com/encryptogroup/ABY for help";
  exit
fi

#Build EMP
wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py
python install.py --deps --tool --ot --sh2pc
rm -rf install.py

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

if [[ "$NO_REVEAL_OUTPUT" == "NO_REVEAL_OUTPUT" ]]; then
	cmake -DCMAKE_INSTALL_PREFIX=./install ../ -DNO_REVEAL_OUTPUT=ON
else
  cmake -DCMAKE_INSTALL_PREFIX=./install ../
fi

cmake --build . --target install --parallel

#Install pre-commit hook for formatting
cd $ROOT
cp Athos/HelperScripts/pre_commit_format_python.sh .git/hooks/pre-commit
