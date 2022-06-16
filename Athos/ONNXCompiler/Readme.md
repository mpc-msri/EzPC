# Usage
Refer to [Compiling an ONNX model](https://github.com/mpc-msri/EzPC/blob/master/Athos/README.md#compiling-an-onnx-model) for instructions on compilation and running.
This part of the code compiles the onnx model to SeeDot AST. 

# Developer Debugging and Logging
Since debugging the code is an arduous task, several things are logged in the following files

`onnx_seedot_name_map.txt` It stores a map from onnx names to SeeDot names of variables

`seedot_ezpc_name_map.txt` It stores a map from SeeDot names to EzPC names of variables

`onnx_ezpc_name_map.txt` The above two maps are combined to create a map that shows the mapping from onnx names to ezpc/cpp names

# Dependency
Other than EzPC dependencies 
`onnx onnxruntime onnx-simplifier`

# Testing
python3 -m unittest
