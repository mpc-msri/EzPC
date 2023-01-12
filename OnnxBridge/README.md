# OnnxBridge 
An end-to-end compiler for converting Onnx Models to Secure Floating Point backend(SecFloat).
- [Setup](#setup)
- [Usage](#usage)
- [Supported Nodes](#supported-nodes)
- [Add Support for Node](#add-support-for-nodes)

## Setup
If you used the `setup_env_and_build.sh` script the below would already have been installed in the `mpc_venv` environment. We require the below packages to run OnnxBridge.
- onnx==1.12.0
- onnxruntime==1.12.1
- onnxsim==0.4.8
- numpy==1.21.0
- protobuf==3.20.1

Above dependencies can be installed using the `requirements.txt` file as below:
```bash
pip3 install -r requirements.txt
```

#### Along with this SecFloat Backend also need to be build for OnnxBridge to work. Follow the steps from `../SCI/` to build SecFloat.

## Usage

### Generate Binaries:  
To compile an onnx file to SecFloat backend, use the below command:
```bash
cd OnnxBridge 
python3 main.py --path "/path/to/onnx-file" --generate "code" --backend SECFLOAT
```

To compile SecFloat Code generated from above step and get executable use:
```bash
Secfloat/compile_secfloat.sh "/path/to/file.cpp"
```

---
### To directly generate executable from Onnx File use:
```bash
cd OnnxBridge 
python3 main.py --path "/path/to/onnx-file" --generate "executable" --backend SECFLOAT
```
---
### Run Inference:
To run secure inference on networks:

```bash
./<network> r=2 [port=port] [chunk=chunk] < <model_weights_file> // Server
./<network> r=1 [add=server_address] [port=port] [chunk=chunk] < <image_file> // Client
```

### To do the same using cleartext:
```bash
cd OnnxBridge 
python3 main.py --path "/path/to/onnx-file" --generate ["executable"/"code"] --backend [SECFLOAT/SECFLOAT_CLEARTEXT]
cat input_input.inp model_input_weights_.inp | ./model_secfloat_ct
```

## Supported Nodes
- Conv
- Relu
- Sigmoid
- Tanh
- Softmax
- MaxPool
- AveragePool
- Concat
- BatchNormalization
- GlobalAveragePool
- Flatten
- Reshape
- Gemm

## Add Support for Nodes
#### Follow below steps to add support for any new node:
#### For example we will consider `"Tanh"` node:
1. Implement the Node in  OnnxBridge/Secfloat/lib_secfloat/link_secfloat.cpp` using secfloat backend as follows:
```cpp
    // tanh(x) = 2 * sigmoid(2 * x) - 1 
    void Tanh(int32_t s1, vector<FPArray> &inArr, vector<FPArray> &outArr){

        const FPArray one = __public_float_to_baba(1.0, ALICE);
        const FPArray two = __public_float_to_baba(2.0, ALICE);

        // 2 * x
        auto twice_input = make_vector_float(ALICE, s1);
        for(int i=0; i<s1; i++){
            twice_input[i] = Mul(inArr[i],two);
        }

        // sigmoid(2 * x)
        auto sigmoid_twice_input = make_vector_float(ALICE, s1);
        Sigmoid(s1, twice_input, sigmoid_twice_input);


        // tanh(x) = 2 * sigmoid(2 * x) - 1
        for(int i=0; i<s1; i++){
            outArr[i] = Mul(two , sigmoid_twice_input[i]);
            outArr[i] = __fp_op->sub(outArr[i], one);
        }

    }// The operators used can be found in SCI/src/FloatingPoint/floating-point.cpp and SCI/src/library_float.h
```
Above implementation is for 1D FPArray, for multidimentional array its recommended to overload the Tanh function to receive multidimentional array which is then reshaped to 1D array and 1D Tanh implementation is called as below:
```cpp
    void Tanh(int32_t s1, int32_t s2, auto &inArr, auto &outArr)
    {
        int32_t size = (s1 * s2);

        auto reshapedInArr = make_vector_float(ALICE, size);

        auto reshapedOutArr = make_vector_float(ALICE, size);

        for (uint32_t i1 = 0; i1 < s1; i1++)
        {
            for (uint32_t i2 = 0; i2 < s2; i2++)
            {
                int32_t linIdx = ((i1 * s2) + i2);

                reshapedInArr[linIdx] = inArr[i1][i2];
            }
        }
        Tanh(size, reshapedInArr, reshapedOutArr);
        for (uint32_t i1 = 0; i1 < s1; i1++)
        {
            for (uint32_t i2 = 0; i2 < s2; i2++)
            {
                int32_t linIdx = ((i1 * s2) + i2);

                outArr[i1][i2] = reshapedOutArr[linIdx];
            }
        }
    }
```
This completes the node implementation in backend.


2. Add assertions for node in class `OnnxNode` inside in  OnnxBridge/utils/onnx_nodes.py` for various attributes.
```python
    @classmethod
    def Tanh(cls, node):
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        # we can print the node at this step and get info on all node parameters
        # additionaly based on your node implementation add assertions or modification on node attributes.
        logger.debug("Tanh is OK!")
```
3. Add format for Function Prototype in class `Operator` in  OnnxBridge/Secfloat/func_calls.py` for the node as implemented in step-1.
```python
    @classmethod
    def Tanh(cls, attributes, inputs, outputs, value_info, var_dict, indent):
        # :param var_dict: Variable Dictionary {actual variable name}->(used variable name).
        # :param value_info: Dictionary {var}->(data-type,shape).
        # :param indent: To give indentation to cpp code generated. 
        logger.debug("Inside Tanh function call.")
        cmmnt = comment("Call  Tanh(shape,input,output)\n", indent)
        return str(
            cmmnt + f"{'   ' * indent}Tanh("
            f"{iterate_list(value_info[inputs[0]][1])}, "
            f"{iterate_list([var_dict[x] for x in inputs])}, "
            f"{iterate_list([var_dict[x] for x in outputs])}"
            f");"
        )
```
4. Lastly add node name i.e "Tanh" to the `implemented` list in function `is_compatible` inside class `FzpcBackend` located in `Onnx-FzPC/backend.py`.
```python
    implemented = [
        "Relu",
        "Sigmoid",
        "Softmax",
        "Conv",
        "MaxPool",
        "Concat",
        "BatchNormalization",
        "AveragePool",
        "GlobalAveragePool",
        "Flatten",
        "Reshape",
        "Gemm",
        "Tanh"
    ]
```

## Demo
Follow [Demo](Secfloat/demo/Readme.md) for OnnxBridge demo with Secfloat.


