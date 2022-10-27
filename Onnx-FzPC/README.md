# Onnx-FzPC 
An end-to-end compiler for converting Onnx Models to Secure Floating Point backend(SecFloat).

## Setup
----
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

Along with this SecFloat Backend also need to be build for Onnx-FzPC to work. Follow the steps from `../SCI/` to build SecFloat.

## Usage
----

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

To directly generate executable from Onnx File use:
```bash
cd Onnx-FzPC 
python3 main.py --path "/path/to/onnx-file" --generate "executable"
```

### Run Inference:

