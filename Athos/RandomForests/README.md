# Requirements/Setup
If you used the setup_env_and_build.sh script the below would have already been installed in the `mpc_venv` environment. We require the below packages to run:
- python3.7
- scikit-learn=0.24.2
- numpy
- graphviz

Athos also makes use of the EzPC compiler internally (please check ../EzPC/README.md for corresponding dependencies).

# Compiling Random forests and Decision trees
Please source the virtual environment in mpc_venv if you used the setup_env_and_build.sh script to setup and build.

```source ~/EzPC/mpc_venv/bin/activate```

Use the `Athos/CompileRandomForests.py` script to compile random forests and decision trees. We support random forests and decision trees generated using scikit (v0.24). You need to save the trained model in pickle format (follow [this](https://github.com/mpc-msri/EzPC/blob/master/Athos/RandomForests/notebooks/RandomForestCaliforniaHousingPickle.ipynb) notebook as an example)
We will use the notebook example as an end-to-end demo.

 ```
 cd notebooks
 python RandomForestCaliforniaHousingPickle.py
 ```
 This dumps the model in `pickle_model.pickle`. Additionally it also dumps a sample input as a numpy array in `input.npy`.
 Note that the number of features in the input vector is 13. The script also prints the expected output for this example `Expected output:  133242.04011201978`
 ```
 python ~/EzPC/Athos/CompileRandomForests.py --pickle ~/EzPC/Athos/RandomForests/notebooks/pickle_model.pickle --task="reg" --model_type="forest" --no_features=13 --role="server"
 ```
The parameters are:
- `pickle` : Path to the pickle file
- `task` : "reg" for regression, "cla" for classification
- `model_type` : "tree" for decision tree, "forest" for random forest
- `no_features` : Number of features in the dataset
- `role` : "server" since we own the model. ("client" explained later)

Seeing `python CompileRandomForests.py --help` for additional flags.
On running the script, we see the following output:
```
Compiled binary: ~/EzPC/Athos/RandomForests/notebooks/ezpc_build_dir/random_forest
Model weights dumped in ~/EzPC/Athos/RandomForests/notebooks/ezpc_build_dir/weight_sf_10.inp
Send client.json to the client machine. Path: ~/EzPC/Athos/RandomForests/notebooks/ezpc_build_dir/client.json
```
The `random_forest` binary contains the 2 party secure protocol to execute the model.
The `weight_sf_10.inp` file contains the weights of the model.
The `client.json` file contains compilation parameters. This file needs to be sent to the client machine.
The client does not receive the model weights and only compute information about the model like number of trees, depth, no. of features.


On the client machine, you need to have EzPC setup too (for testing, we can do it on the same machine). Say you have put `client.json` in `~/testing`. To compile do:
```
python ~/EzPC/Athos/CompileRandomForests.py --role="client" --config ~/testing/client.json
```
The generated binary is in `~/testing/ezpc_build_dir/random_forest`

# Preprocessing input
On the client machine, we need to preprocess the input by converting it into fixed point. We will use the `input.py` generated using the notebook:
```
source ~/EzPC/mpc_venv/bin/activate
cp ~/EzPC/Athos/RandomForests/notebooks/input.npy ~/testing/ # Or place some other input dumped as a numpy array
python ~/EzPC/Athos/CompilerScripts/convert_np_to_fixedpt.py --inp ~/testing/input.npy --config ~/testing/client.json
```
This generates `input_fixedpt_scale_10.inp` in the `~/testing` directory.

The server already has its input ready in `~/EzPC/Athos/RandomForests/notebooks/ezpc_build_dir/weight_sf_10.inp`.

# Running
If you are testing on the same machine, open two terminals, one for client and one for server. Else open a terminal on each machine.

Server side:
```
~/EzPC/Athos/RandomForests/notebooks/ezpc_build_dir/random_forest -r 0 -p 32000 < ~/EzPC/Athos/RandomForests/notebooks/ezpc_build_dir/weight_sf_10.inp
```
Client side:
```
~/testing/ezpc_build_dir/random_forest -r 1 -p 32000 -a 127.0.0.1 < ~/testing/input_fixedpt_scale_10.inp
```
Here the `-p` parameter specifies the port number and the `-a` parameter needs to specify the ip address of the server.
We use 127.0.0.1 (localhost) as both are on the same machine.
You need to ensure that the port used is open on the server machine.
