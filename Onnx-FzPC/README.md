# Onnx-FzPC 
An end-to-end compiler for converting Onnx Models to Secure Floating Point backend(SecFloat).
- [Setup](#setup)
- [Usage](#usage)
- [Supported Nodes](#supported-nodes)
- [Add Support for Node](#add-support-for-nodes)

## Setup
If you used the `setup_env_and_build.sh` script the below would already have been installed in the `mpc_venv` environment. We require the below packages to run Onnx-FzPC.
- onnx==1.12.0
- onnxruntime==1.12.1
- onnxsim==0.4.8
- numpy==1.23.4
- protobuf==3.20.1

Above dependencies can be installed using the `requirements.txt` file as below:
```bash
pip3 install requirements.txt
```

#### Along with this SecFloat Backend also need to be build for Onnx-FzPC to work. Follow the steps from `../SCI/` to build SecFloat.

## Usage

### Generate Binaries:  
To compile an onnx file to SecFloat backend, use the below command:
```bash
cd Onnx-FzPC 
python3 main.py --path "/path/to/onnx-file" --generate "code"
```

To compile SecFloat Code generated from above step and get executable use:
```bash
./compile_secfloat.sh "/path/to/file.cpp"
```
---
### To directly generate executable from Onnx File use:
```bash
cd Onnx-FzPC 
python3 main.py --path "/path/to/onnx-file" --generate "executable"
```
---
### Run Inference:
To run secure inference on networks:

```bash
./<network> r=1 [port=port] < <model_weights_file> // Server
./<network> r=2 [ip=server_address] [port=port] < <image_file> // Client
```

## Supported Nodes
- Conv
- Relu
- Sigmoid
- Softmax
- MaxPool
- Global MaxPool
- AveragePool
- Concat
- BatchNormalization
- GlobalAveragePool
- Flatten
- Reshape
- Gemm

## Add Support for Nodes
Follow below steps to add support for any new node:

1. Implement the Node in `Onnx-FzPC/lib_secfloat/link_secfloat.cpp`

2. Add assertions for node in class `OnnxNode` inside in `Onnx-FzPC/utils/onnx_nodes.py` for various attributes.
```python
    @classmethod
    def Relu(cls, node):
        assert len(node.inputs) == 1
        logger.debug("Relu is OK!")
```
3. Add format for Function Prototype in class `Operator` in `Onnx-FzPC/utils/func_calls.py` for the node as implemented in step-1.
```python
    @classmethod
    def Relu(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        logger.debug("Inside Relu function call.")
        cmmnt = comment("Call  Relu(shape,input,output)\n", indent)
        return str(cmmnt +
                   f"{'   ' * indent}Relu("
                   f"{iterate_list(value_info[inputs[0]][1])}, "
                   f"{iterate_list([var_dict[x] for x in inputs])}, "
                   f"{iterate_list([var_dict[x] for x in outputs])}"
                   f");")
```


