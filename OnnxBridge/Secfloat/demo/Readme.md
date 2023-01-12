
# Setup Env
After having setup the environment using ./setup_env_and_build.sh quick, load the environment:
```bash
source ~/EzPC/mpc_venv/bin/activate
```
### OR
Setup the env manually using:
```bash
cd ../../
pip install -r requirements.txt
```
Build SecFloat following [SCI](https://github.com/mpc-msri/EzPC/blob/onnx-fzpc/SCI/README.md).


# Server Side

## Download Model
```bash
./fetch_model.sh 
```
## Compile the Model
Run the following command to compile the model:
```bash

python ../../main.py --path "model.onnx" --generate "executable" --backend SECFLOAT

```
This generates :
- A file with model weigths `~/EzPC/OnnxBridge/Secfloat/demo/model_input_weights_.inp` (Secret Server Data) 
- A model output binary : `~/EzPC/OnnxBridge/Secfloat/demo/model_secfloat.out` which needs to be passed to client.

Run the following command to start server side computation and wait for client connection:
```bash
./model_secfloat r=2  [port=port] < model_input_weights_.inp
```

# Client Side

## Download Image
```bash
./fetch_image.sh 
```

Client needs to preprocess the image before computation starts:
```bash
./process_image.sh "input.jpg"
```
This generates two files:
- `input.npy` 
- `input_input.inp` for secure inference.

Run the following command to start client side computation and connect with server:
```bash
./model_secfloat r=1  [add=server_address] [port=port] < input_input.inp  > output.txt
```
Raw Output will be saved in `output.txt` , to get output as numpy array do : 
```bash
python ../../helper/make_np_arr.py "output.txt"
```
This dumps model output as a flattened numpy array(1-D) in output.npy .

## Verify Output
To verify if everything is working as expected, run the input image with the model itself using the onnx runtime:
```bash
python ../../helper/run_onnx.py "input.npy"
```
It dumps the output in `onnx_output/input.npy` and also prints it on the screen. To compare the both outputs do:
```bash
python ../../helper/compare_np_arrs.py -i onnx_output/input.npy output.npy
```
You should get output similar to:
```bash
Arrays matched upto 6 decimal points
```
