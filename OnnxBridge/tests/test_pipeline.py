import pytest
import os
from utils import (
    pre_process_input,
    run_onnx,
    compile_model,
    run_backend,
    compare_output,
)

# Get the directory path where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
ezpc_dir = os.path.join(script_directory, "..", "..")

lenet = {
    "model": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/lenet.onnx",
    "input": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/7.jpg",
    "preprocess": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/preprocess.py",
}
hinet = {
    "model": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/cnn3_cifar.onnx",
    "input": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/image_0.png",
    "preprocess": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/preprocess.py",
}


@pytest.mark.parametrize("model", ["lenet", "hinet"])
def test_model(test_dir, backend, model):
    os.chdir(test_dir)
    model = globals()[model]

    # download the model & data & preprocessing_file
    os.system(f"wget {model['model']} -O model.onnx")
    os.system(f"wget {model['input']} -O input.jpg")
    os.system(f"wget {model['preprocess']} -O preprocess.py")

    # preprocess the input
    pre_process_input()

    # run the model with OnnxRuntime
    run_onnx()

    # compile the model with backend
    compile_model(backend)

    # run the model with backend
    run_backend(backend)

    # compare the output
    compare_output()

    os.chdir("../..")
