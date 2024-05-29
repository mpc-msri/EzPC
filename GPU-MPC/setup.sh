# Set environment variables
export NVCC_PATH="/usr/local/cuda-$CUDA_VERSION/bin/nvcc"

echo "Updating submodules"
git submodule update --init --recursive

# Install dependencies
echo "Installing g++-9"
sudo apt install -y gcc-9 g++-9;
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9;
sudo update-alternatives --config gcc;


#installing dependencies
sudo apt install libssl-dev cmake python3-pip libgmp-dev libmpfr-dev;


echo "Installing dependencies"
sudo apt install cmake make libeigen3-dev;

echo "Building CUTLASS"
# Build CUTLASS
cd ext/cutlass;
if [ -n "$1" ]
then 
git checkout $1;
fi
mkdir build && cd build;
cmake .. -DCUTLASS_NVCC_ARCHS=$GPU_ARCH -DCMAKE_CUDA_COMPILER_WORKS=1 -DCMAKE_CUDA_COMPILER=$NVCC_PATH;
make -j;
cd ../../..;

# Build sytorch
echo "Building Sytorch"
cd ext/sytorch;
mkdir build && cd build;
cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_BUILD_TYPE=Release ../ -DCUDAToolkit_ROOT="/usr/local/cuda-$CUDA_VERSION/bin/";
make sytorch -j;
cd ../../..;

# Download CIFAR-10
cd experiments/orca/datasets/cifar-10;
sh download-cifar10.sh;
cd ../../../..;


# Make shares of data
make share_data;
cd experiments/orca;
./share_data;
cd ../..;

# Build the orca codebase
# make orca; 

# Make output directories
# Orca
mkdir experiments/orca/output;
mkdir experiments/orca/output/P0;
mkdir experiments/orca/output/P1;
mkdir experiments/orca/output/P0/training;
mkdir experiments/orca/output/P1/training;
mkdir experiments/orca/output/P0/inference;
mkdir experiments/orca/output/P1/inference;

# Sigma
mkdir experiments/sigma/output;
mkdir experiments/sigma/output/P0;
mkdir experiments/sigma/output/P1;

# install matplotlib
pip3 install matplotlib
