## Download the model:
```
./fetch_model.sh
```

## Compile the model:
After having setup the environment using ```./setup_env_and_build.sh quick```,
load the environment:
```
source ~/EzPC/mpc_venv/bin/activate
```
We call the party that owns the model, the server and the party that owns data to perform inference upon, the client. Both parties need to perform the above step.

### Server side

To compile the model do:
```
python ~/EzPC/Athos/CompileONNXGraph.py --config config.json --role server
```
This generates:
- `model_SCI_HE.out`: The binary with the MPC protocol.
- `model_input_weights_fixedpt_scale_15.inp`: Model weights to be fed as input to the binary.
- `client.zip`: This contains:
   - `config.json`: The compilation config file.
   - `optimised_model.onnx`: The model with the weights stripped off.

We will send this `client.zip` to the client. Since the model is pruned, it doesn't reveal any propreitary weights to the client and only the model structure i.e. the computation. The client will then compile this model using CrypTFlow and generate the MPC binary.


### Client side

The client, on receiving the client.zip file from the server needs to extract it and then compile it.
```
unzip client.zip
python ~/EzPC/Athos/CompileONNXGraph.py --config config.json --role client

```
This generates the same `model_SCI_HE.out` binary. Next the client needs to do the pre-processing of their input so that it would run with the model. This can include things like resizing, normalizing, cropping, etc.. This would depend on the server's model and the server will let the client know what all is required. After preprocessing we need to save the input as a numpy array. For the model in this directory, we do it according to `pre_process.py`.
```
python pre_process.py input.jpg
```
This will perform the transformations and dump the output in `input.npy` Next we convert this input to fixedpoint format so that it can run with the MPC protocol.
```
python ~/EzPC/Athos/CompilerScripts/convert_np_to_fixedpt.py --inp input.npy --config config.json
```
This dumps `input_fixedpt_scale_15.inp`. Now both parties are ready to run the computation.

## Running the model:
Since our protocols are interactive in nature, both machines need to be able to communicate with each other over the network. Ensure that you have a port open on the server machine and that the server is reachable via the client. Say the server's ip address is `123.231.231.123` and the open port is `12345`.

### Server side
```
./model_SCI_HE.out r=1 p=12345 < model_input_weights_fixedpt_scale_15.inp
```
### Client side
```
./model_SCI_HE.out r=2 ip=123.231.231.123 p=12345 < input_fixedpt_scale_15.inp > output.txt
```
`output.txt` contains the raw output from the computation. To get the output as a numpy array do:
```
python ~/EzPC/Athos/CompilerScripts/get_output.py output.txt config.json
```
This dumps model_output.npy as a flattened numpy array(1-D).

To verify if everything is working as expected, run the input image with the model itself using the onnx runtime:
```
python run_onnx.py input.npy
```
It dumps the output in onnx_output/input.npy and also prints it on the screen. To compare the both outputs do:
```
python ~/EzPC/Athos/CompilerScripts/comparison_scripts/compare_np_arrs.py onnx_output/input.npy model_output.npy
```
You should get output similar to:
```
Arrays matched upto 2 decimal points
```
