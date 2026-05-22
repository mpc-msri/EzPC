
# GPU-MPC

Implementation of protocols from the papers [Orca](https://eprint.iacr.org/2023/206) and [SIGMA](https://eprint.iacr.org/2023/1269).

**Warning**: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.

## Build

This project requires NVIDIA GPUs, and assumes that GPU drivers and the [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/) are already installed. The following has been tested on Ubuntu 20.04 with CUDA 11.7, CMake 3.27.2 and g++-9. 

Please note that Sytorch requires CMake version >= 3.17 and the build will fail if this depency is not met. 

The code uses CUTLASS version 2.11 by default, so if you change the CUDA version, please make sure that the CUTLASS version being built is compatible with the new CUDA version.

The last line of `setup.sh` tries to install `matplotlib`, which is needed for generating Figures 5a and 5b. In our experience, the installation fails if the versions of Python and `pip` do not match. In case the installation fails, please install `matplotlib` manually before running `run_experiment.py`.

1. Export environment variables

```
export CUDA_VERSION=11.7
export GPU_ARCH=86
```

2. Set up the environment. 

```
sh setup.sh <CUTLASS branch>
```

To change the version of CUTLASS being built, optionally include the CUTLASS branch that should be built as

```
sh setup.sh <CUTLASS branch>
```
For example, to build the main branch, run

```
sh setup.sh main
```


3. Make Orca

```
make orca
```
4. Make sigma (this does not require making Orca)

```
make sigma
```

## Run Orca

Please see the [Orca README](experiments/orca/README.md).

## Run SIGMA

Please see the [SIGMA README](experiments/sigma/README.md)

## Docker Build

You can also build the docker image using the provided Dockerfile_Gen for building the Environment. 

### Install Nvidia Container Toolkit
- Configure the repository:
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update
```

- Install the NVIDIA Container Toolkit packages:
```
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
### Build the Docker Image / pull the image from Docker Hub
```
# Local Build
docker build -t gpu_mpc -f Dockerfile_Gen .

# Pull from Docker Hub (Cuda 11.8)
docker pull trajore/gpu_mpc
```
### Run the Docker Container
```
sudo docker run --gpus all --network host -v /home/$USER/path_to_GPU-MPC/:/home -it container_name /bin/bash

```
Then Run setup.sh to configure according to GPU_arch and make Orca/SIGMA as mentioned above.

