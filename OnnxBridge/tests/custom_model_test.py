import pytest
import os
from utils import (
    run_onnx,
    compile_model,
    run_backend,
    compare_output,
)

# Get the directory path where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
ezpc_dir = os.path.join(script_directory, "..", "..")


def test_custom_model(test_dir, backend, model, input_name):
    """
    Usage:
    pytest path/custom_model_test.py -s --backend CLEARTEXT_LLAMA --model /home/saksham/EzPC/OnnxBridge/nnUnet/optimized_fabiansPreActUnet.onnx --input_name /home/saksham/EzPC/OnnxBridge/nnUnet/inputs/2d_input
    """
    os.chdir(test_dir)

    # model is absolute path to model.onnx
    # input is absolute path to input1.j
    # download the model & data & preprocessing_file
    os.system(f"cp {model} model.onnx")
    os.system(f"cp {input_name}.inp input1.inp")
    os.system(f"cp {input_name}.npy input1.npy")

    # preprocessed input is directly copied

    # run the model with OnnxRuntime
    run_onnx("input1.npy")

    # compile the model with backend
    compile_model(backend)

    # run the model with backend
    run_backend(backend, "input1.inp")

    # compare the output
    compare_output()

    os.chdir("../..")
