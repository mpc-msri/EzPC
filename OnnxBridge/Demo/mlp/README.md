# Inference App - MLP fMRI 

In this demo we will run the Inference App for rs-fMRI using MLP model for interactive but secure inferencing following secure MPC using EzPC.

## System Requirements

Check [inference-app Readme](/inference-app/README.md#system-requirements) for system requirements.<br>
<br>
To successfully execute this demo we will need three **Ubuntu** VMs [tested on Ubuntu 20.04.6 LTS]:
1. **Dealer** : Works to generate pre-computed randomness and sends it to Client and Server for each inference. 
2. **Server** : This party owns the model, and _does not share its model weights with Dealer/Client_, hence uses EzPC SMPC to achieve Secure Inference.
3. **Client** : This party acts as Client, but _does not hold any data by itself_, it gets Masked Image from the frontend, thus this party itself _can't see the image data in cleartext_. On receiving the Masked Image it starts the secure inference with Server and returns the result back to frontend.


Additionally we need a machine to run the frontend on, this is independent of OS, can be run on Client machine aswell if UI is available for Client VM, as the frontend runs in a browser.

## Install Dependencies

1. On all Ubuntu VM, install dependencies:
```bash
sudo apt update
sudo apt install libeigen3-dev cmake build-essential git zip
```

2. On all Ubuntu VM, install the python dependencies in a virtual environment.
``` bash
# Demo directory where we will install our dependencies and follow all the further steps.
mkdir MLP-DEMO
cd MLP-DEMO

sudo apt install python3.8-venv
python3 -m venv venv
source venv/bin/activate

wget https://raw.githubusercontent.com/mpc-msri/EzPC/master/OnnxBridge/requirements.txt
pip install --upgrade pip
sudo apt-get install python3-dev build-essential
pip install -r requirements.txt
pip install tqdm pyftpdlib flask
```

## Setup Server

```bash
# Run the below notebook to extract the mlp_model.onnx file from the Github Repo for fMRI-Classification
# https://github.com/AmmarPL/fMRI-Classification-JHU/
```
Run the [extract_mlp_model.ipynb](/OnnxBridge/Demo/mlp/extract_mlp_model.ipynb) to download the mlp onnx file.

```bash
# while inside MLP-DEMO
# copy the mlp_model.onnx file inside MLP-DEMO
cp /path/to/mlp_model.onnx .
mkdir play
cd play
```

## Setup Client
Make a temporary Directory.
```bash
# while inside MLP-DEMO
mkdir play
cd play
```

## Setup Dealer
Make a temporary Directory.
```bash
# while inside MLP-DEMO
mkdir play
cd play
```

## Setup Frontend

 On the system being used as the frontend, follow below instructions to setup Webapp
```bash
# clone repo
git clone https://github.com/mpc-msri/EzPC
cd EzPC

# create virtual environment and install dependencies 
sudo apt update
sudo apt install python3.8-venv
python3 -m venv mlinf
source mlinf/bin/activate
pip install --upgrade pip
sudo apt-get install python3-dev build-essential
pip install -r inference-app/requirements.txt
```
---
Generate the scripts and transfer them to respective machines. If server, client and dealer are in same virtual network, then pass the private network IP in the ezpc_cli-app.sh command.

```bash
cd inference-app
chmod +x ezpc-cli-app.sh
./ezpc-cli-app.sh -m /home/<user>/MLP-DEMO/MLP.onnx -s <SERVER-IP> -d <DEALER-IP> [ -nt <num_threads> ]
scp server.sh <SERVER-IP>:/home/<user>/MLP-DEMO/play/
scp dealer.sh  <DEALER-IP>:/home/<user>/MLP-DEMO/play/
scp client-offline.sh <CLIENT-IP>:/home/<user>/MLP-DEMO/play/
scp client-online.sh  <CLIENT-IP>:/home/<user>/MLP-DEMO/play/
```
In the above commands, the file paths and directories are absolute paths on the Ubuntu VMs used. To know more about the `ezpc-cli-app.sh` script see [link](/inference-app/Inference-App.md). <br/>

----

On all Ubuntu VMs, make the bash scripts executable and execute them.

```bash
# (on server)
chmod +x server.sh
./server.sh

# (on dealer)
chmod +x dealer.sh
./dealer.sh

# (on client)
chmod +x client-offline.sh client-online.sh
./client-offline.sh
```
-----
#### Create a .`env` file inside `EzPC/inference-app` directory to store the secrets as environment variables ( `_URL` is the IP address of Dealer ), the file should look as below:
    _URL = "X.X.X.X"
    _USER = "frontend"
    _PASSWORD = "frontend"
    _FILE_NAME = "masks.dat"
    _CLIENT_IP = "X.X.X.X"

----
#### Download the preprocessing file for image (specific to model) inside /inference-app directory:
```bash
# This file takes in image as <class 'PIL.Image.Image'>
# preprocess it and returns it as a numpy array of size required by Model.
wget "https://raw.githubusercontent.com/drunkenlegend/ezpc-warehouse/main/MLP_fMRI/preprocess.py" -O preprocess.py
```

#### 
```bash
# Next we download example image for the app.
cd Assets 
mkdir examples && cd examples 
wget "https://raw.githubusercontent.com/drunkenlegend/ezpc-warehouse/main/MLP_fMRI/1.png" -O 1.jpg
cd ../..
```
----
#### Replace the USER_INPUTS in constants.py file with below:

    # Description
    desc = "In this example app, we demonstrate how infer any fMRI Image with a MLP model trained by JHU in a secure manner using EzPC."

    # preprocess is a function that takes in an image and returns a numpy array
    preprocess = get_arr_from_image

    # The input shape of the model, batch size should be 1
    Input_Shape = (1, 1, 45, 54, 45)
    assert Input_Shape[0] == 1, "Batch size should be 1"
    dims = {
        "c": 1,
        "h": 45,
        "w": 54,
        "d": 45,
    }

    scale = 15

    # Labels till 54
    labels_map = {i: i for i in range(56)}

```bash
# while inside inference-app directory
python app_3d.py
```

Open the url received after running the last command on inference-app and play along:
1. Upload fMRI image.
2. Get Encryption Keys
3. Encrypt Image
4. Start Inference