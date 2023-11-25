# HiNet-LLAMA Demo
In this demo we will see how we can use OnnxBridge to perform secure Inference on a Chexpert model using LLAMA backend.
## Setup Env
After having setup the environment using ./setup_env_and_build.sh quick, load the environment:
```bash
source ~/EzPC/mpc_venv/bin/activate
```
### OR
Setup the env manually using:
```bash
# OnnxBridge dependencies
cd ..
pip install -r requirements.txt

# LLAMA dependencies
sudo apt update
sudo apt install libeigen3-dev cmake build-essential 
```

# Server Side

### Download Model
```bash
./fetch_model.sh 
```
### Compile the Model
Run the following command to compile the model:
```bash

python ../../main.py --path "model.onnx" --generate "executable" --backend LLAMA --scale 15 --bitlength 40

```
This generates :
- A file with model weigths `~/EzPC/OnnxBridge/Demo/model_input_weights.dat` (Secret Server Data) 
- A model output binary : `~/EzPC/OnnxBridge/Demo/model_LLAMA_15.out` which will perform the secure mpc computation.
- A cpp file : `~/EzPC/OnnxBridge/Demo/model_LLAMA_15.cpp` which represents the model architecture needs to be passed to the DEALER and CLIENT.

# Dealer Side
### Compile Model
Compile the cpp model architecture file received from the server.
```bash
# compile secure code
~/EzPC/OnnxBridge/LLAMA/compile_llama.sh "model_LLAMA_15.cpp"
```

### Run Computation
Run the following command to start client side computation and connect with server:
```bash
./model_LLAMA_15 1  
```
Above command generated two files: 
- **server.dat** : pass it to server
- **client.dat** : pass it to client

# Server side
### Run Computation
Run the following command to start server side computation and wait for client connection:
```bash
./model_LLAMA_15 2 model_input_weights.dat
```

# Client Side

### Download Image
```bash
sudo ./fetch_image.sh 
```

Client needs to preprocess the image before computation starts:
```bash
./process_image.sh "input.jpg"
```
This generates two files:
- `input.npy` 
- `input_input.inp` for secure inference.

### Compile Model
Compile the cpp model architecture file received from the server.
```bash
# compile secure code
~/EzPC/OnnxBridge/LLAMA/compile_llama.sh "model_LLAMA_15.cpp"
```

### Run Computation
Run the following command to start client side computation and connect with server:
```bash
./model_LLAMA_15 3 127.0.0.1 < input_input.inp  > output.txt
```
Raw Output will be saved in `output.txt` , to get output as numpy array do : 
```bash
python ../../helper/make_np_arr.py "output.txt"
```
This dumps model output as a flattened numpy array(1-D) in output.npy .

## Verify Output
To verify if everything is working as expected, run the input image with the model itself using the onnx runtime:
```bash
python ../../helper/run_onnx.py model.onnx "input.npy"
```
It dumps the output in `onnx_output/input.npy` and also prints it on the screen. To compare the both outputs do:
```bash
python ../../helper/compare_np_arrs.py -i onnx_output/expected.npy output.npy
```
You should get output similar to:
```bash
Arrays matched upto 2 decimal points
```
