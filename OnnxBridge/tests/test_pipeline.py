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


def test_lenet_mnist(test_dir, backend):

    # print("\n*************")
    # print(os.getcwd())
    # print("*************")
    os.chdir(test_dir)
    # print("*************")
    # print(os.getcwd())
    # print("*************")

    # download the model & data & preprocessing_file
    os.system(
        "wget https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/lenet.onnx -O model.onnx"
    )
    os.system(
        "wget https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/7.jpg -O input.jpg"
    )
    os.system(
        "wget https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/preprocess.py -O preprocess.py"
    )

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
