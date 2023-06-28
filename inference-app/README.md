# Inference App
This Gradio App gives a frontend to [EzPC](https://github.com/mpc-msri/EzPC) and enables you to make secure inference for images with a pretrained Model and get results in a UI based setup. <br/>

https://github.com/drunkenlegend/ezpc-warehouse/assets/50737587/4ff01074-9970-434e-8be6-b53d2fd0c768


# System Requirements
Following are the system requirements and steps to run the Inference-App for doing secure inferencing on X-ray images with a Chexpert Model.
To successfully execute this demo we will need three **Ubuntu** VMs [tested on Ubuntu 20.04.6 LTS]:
1. **Dealer** : Works to generate pre-computed randomness and sends it to Client and Server for each inference. 
2. **Server** : This party owns the model, and _does not share its model weights with Dealer/Client_, hence uses EzPC SMPC to achieve Secure Inference.
3. **Client** : This party acts as Client, but _does not hold any data by itself_, it gets Masked Image from the frontend, thus this party itself _can't see the image data in cleartext_. On receiving the Masked Image it starts the secure inference with Server and returns the result back to frontend.


Additionally we need a machine to run the frontend on, this is independent of OS, can be run on Client machine aswell if UI is available for Client VM, as the frontend runs in a browser.

Notes:
- Frontend should be able to communicate with Dealer and Client over port 5000.
- Server should be able to communicate with Dealer and Client over port 8000.
- Dealer should be able to communicate with Server and Client over port 9000.
- Server and Client should be able to communicate over ports 42003-42005.


# Setup

1. On all Ubuntu VM, install dependencies:
```bash
sudo apt update
sudo apt install libeigen3-dev cmake build-essential git zip
```

2. On all Ubuntu VM, install the python dependencies in a virtual environment.
``` bash
# Demo directory where we will install our dependencies and follow all the further steps.
mkdir CHEXPERT-DEMO
cd CHEXPERT-DEMO

sudo apt install python3.8-venv
python3 -m venv venv
source venv/bin/activate

wget https://raw.githubusercontent.com/mpc-msri/EzPC/master/OnnxBridge/requirements.txt
pip install --upgrade pip
sudo apt-get install python3-dev build-essential
pip install -r requirements.txt
pip install tqdm pyftpdlib flask
```

3. **SERVER** : Download ONNX file for CheXpert model and make a temporary directory.
```bash
# while inside CHEXPERT-DEMO
wget "https://github.com/bhatuzdaname/models/raw/main/chexpert.onnx" -O chexpert.onnx
mkdir play
cd play
```

4. **CLIENT** : Make a temporary Directory.
```bash
# while inside CHEXPERT-DEMO
mkdir play
cd play
```

5. **DEALER** : Make a temporary Directory.
```bash
# while inside CHEXPERT-DEMO
mkdir play
cd play
```

6. **FRONTEND** : On the system being used as the frontend, follow below instructions to setup Webapp
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

7. **FRONTEND** : Generate the scripts and transfer them to respective machines. If server, client and dealer are in same virtual network, then pass the private network IP in the ezpc_cli-app.sh command.
```bash
cd inference-app
chmod +x ezpc-cli-app.sh
./ezpc-cli-app.sh -m /home/<user>/CHEXPERT-DEMO/chexpert.onnx -s <SERVER-IP> -d <DEALER-IP> [ -nt <num_threads> ]
scp server.sh <SERVER-IP>:/home/<user>/CHEXPERT-DEMO/play/
scp dealer.sh  <DEALER-IP>:/home/<user>/CHEXPERT-DEMO/play/
scp client-offline.sh <CLIENT-IP>:/home/<user>/CHEXPERT-DEMO/play/
scp client-online.sh  <CLIENT-IP>:/home/<user>/CHEXPERT-DEMO/play/
```
In the above commands in step 7, the file paths and directories are absolute paths on the Ubuntu VMs used. To know more about the `ezpc-cli-app.sh` script see [link](/inference-app/Inference-App.md). <br/><br/>
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

8. **FRONTEND** : setup & run the webapp:
#### Create a .`env` file inside `EzPC/inference-app` directory to store the secrets as environment variables ( `_URL` is the IP address of Dealer ), the file should look as below:
    _URL = "X.X.X.X"
    _USER = "frontend"
    _PASSWORD = "frontend"
    _FILE_NAME = "masks.dat"
    _CLIENT_IP = "X.X.X.X"

Download the preprocessing file for image (specific to model) inside `/inference-app` directory:
```bash
# This file takes in image as <class 'PIL.Image.Image'>
# preprocess it and returns it as a numpy array of size required by Model.
wget "https://raw.githubusercontent.com/mpc-msri/EzPC/master/inference-app/Assets/preprocess.py" -O preprocess.py
```

```bash
# Next we download example image for the app.
cd Assets 
mkdir examples && cd examples 
wget "https://raw.githubusercontent.com/drunkenlegend/ezpc-warehouse/main/Chexpert/cardiomegaly.jpg" -O 1.jpg
cd ../..
```

***Note:*** 

    Further in case of using some other model for demo and customising WebApp to fit your model,
    modify the USER_INPUTS in constants.py file in /inference-app directory.

```bash
# while inside inference-app directory
python app.py
```

Open the url received after running the last command on inference-app and play along:
1. Upload X-ray image.
2. Get Encryption Keys
3. Encrypt Image
4. Start Inference




