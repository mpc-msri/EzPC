## Single inference

Given an model onnx file, OnnxBridge can be used to generate an executable which can be run on two VMs, server and client (owning the model weights and input image respectively), to get the secure inference output. To do this, use the `ezpc-cli.sh` script by running the following command locally (not neccesarily on a VM):

```bash
./ezpc-cli.sh -m /absolute/path/to/model.onnx -preprocess /absolute/path/to/preprocess.py -s server-ip -i /absolute/path/to/image.jpg
```

In the above command, the paths are not local, but are the locations on the respective VMs. That is, `/absolute/path/to/model.onnx` is the path of model.onnx file on the server VM, `/absolute/path/to/preprocess.py` is the path of preprocessing script on the server VM, and `/absolute/path/to/image.jpg` is the path of image on the client VM. To write the preprocessing script for your use case, refer to the preprocessing file of the [chexpert demo](/Athos/demos/onnx/pre_process.py). If your preprocessing script uses some additional python packages, make sure they are installed on the server and client VMs. Also, ensure that the client can communicate with the server through the IP address provided on the ports between the range 42002-42100. Optionally, you can also pass the following arguments:

- `-b <backend>`: the MPC backend to use (default: `LLAMA`)
- `-scale <scale>`: the scaling factor for the model input (default: `15`)
- `-bl <bitlength>`: the bitlength to use for the MPC computation (default: `40`)

The script generates 4 scripts:

- `server-offline.sh` - Transfer this script to the server VM in any empty directory. Running this script (without any argument) reads the ONNX file, strips model weights out of it, dumps sytorch code, zips the code required to be sent to the client and waits for the client to download the zip. Once the zip is transfered, the script generates the preprocessing key material.
- `client-offline.sh` - Transfer this script to the client VM in any empty directory. Running this script fetches the stripped code from server and generates the preprocessing key material. This script must be run on client VM parallely while server VM is running it's server script. 
- `server-online.sh` - Transfer this script to the server VM in the same directory. Running this script waits for client and starts the inference once the client connects.
- `client-online.sh` - Transfer this script to the client VM in the same directory. Running this script preprocesses the input, connects with the server and starts the inference. After the secure inference is complete, inference output is printed and saved in `output.txt` file.

## Toy example - LeNet-MNIST inference

Using the above instructions, we now demonstrate LeNet inference on MNIST images. We assume that we start at the home path `/home/<user>` on both machines. The below instructions also work on three terminals opened on a single machine (each terminal representing client, server and local computer) by passing `127.0.0.1` as IP address. 

1. On both machines, install dependencies.

```
sudo apt update
sudo apt install libeigen3-dev cmake build-essential git
```

2. On both machines, install the python dependencies in a virtual environment.

```
python3 -m venv venv
source venv/bin/activate
wget https://raw.githubusercontent.com/mpc-msri/EzPC/master/OnnxBridge/requirements.txt
pip install -r requirements.txt
```
3. Download ONNX file and preprocessing script for LeNet on the server and make a temporary directory.

```
mkdir lenet-demo-server
cd lenet-demo-server
wget https://github.com/kanav99/models/raw/main/lenet.onnx
wget https://github.com/kanav99/models/raw/main/preprocess.py
mkdir tmp
cd tmp
```

4. Download the test image on the client and make a temporary directory.

```
mkdir lenet-demo-client
cd lenet-demo-client
wget https://github.com/kanav99/models/raw/main/input.jpg
mkdir tmp
cd tmp
```

5. On the local computer, clone EzPC repository, generate the scripts and transfer them to respective machines. If server and client are in same local network, then pass the local network IP in the `ezpc_cli.sh` command.

```
git clone https://github.com/mpc-msri/EzPC
cd EzPC
cd sytorch
./ezpc-cli.sh -m /home/<user>/lenet-demo-server/lenet.onnx -preprocess /home/<user>/lenet-demo-server/preprocess.py -s <SERVER-IP> -i /home/<user>/lenet-demo-client/input.jpg
scp server-offline.sh <SERVER-IP>:/home/<user>/lenet-demo-server/tmp/
scp server-online.sh  <SERVER-IP>:/home/<user>/lenet-demo-server/tmp/
scp client-offline.sh <CLIENT-IP>:/home/<user>/lenet-demo-client/tmp/
scp client-online.sh  <CLIENT-IP>:/home/<user>/lenet-demo-client/tmp/
```

6. On both machines, make the bash scripts executable and start the offline phase.

```
(on server)
chmod +x server-offline.sh server-online.sh
./server-offline.sh

(on client)
chmod +x client-offline.sh client-online.sh
./client-offline.sh
```

7. Once offline phase completes, start the online phase. The inference logits get printed on the client terminal.

```
(on server)
./server-online.sh

(on client)
./client-online.sh
```

In this particular example, you should get a score array of `[-2.71362 1.06747 4.43045 0.795044 -3.21173 -2.39871 -8.49094 10.3443 1.0567 -0.694458]`, which is maximum at index 7, which is indeed expected as the [input.jpg](https://github.com/kanav99/models/raw/main/input.jpg) file contains an image of handwritten 7.
