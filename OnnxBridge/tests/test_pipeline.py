import pytest
import os
from utils import (
    pre_process_input,
    run_onnx,
    compile_model,
    run_backend,
    compare_output,
    append_np_arr,
)

# Get the directory path where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
ezpc_dir = os.path.join(script_directory, "..", "..")

lenet = {
    "model": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/lenet.onnx",
    "model_batch": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/lenet_batch.onnx",
    "input1": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/7.jpg",
    "input2": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/2.jpg",
    "preprocess": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Lenet_mnist/preprocess.py",
}
hinet = {
    "model": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/cnn3_cifar.onnx",
    "model_batch": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/cnn3_cifar_batch.onnx",
    "input1": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/image_0.png",
    "input2": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/image_1.png",
    "input3": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/image_2.png",
    "input4": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/image_3.png",
    "input5": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/image_4.png",
    "preprocess": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/HiNet_cifar10/preprocess.py",
}
chexpert = {
    "model": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Chexpert/chexpert.onnx",
    "input1": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Chexpert/cardiomegaly.jpg",
    "preprocess": "https://github.com/drunkenlegend/ezpc-warehouse/raw/main/Chexpert/preprocess.py",
}


@pytest.mark.parametrize("model", ["lenet", "hinet", "chexpert"])
def test_model(test_dir, backend, model):
    os.chdir(test_dir)
    model = globals()[model]

    # download the model & data & preprocessing_file
    os.system(f"wget {model['model']} -O model.onnx")
    os.system(f"wget {model['input1']} -O input1.jpg")
    os.system(f"wget {model['preprocess']} -O preprocess.py")

    # preprocess the input
    pre_process_input(1)

    # run the model with OnnxRuntime
    run_onnx("input1.npy")

    # compile the model with backend
    compile_model(backend)

    # run the model with backend
    run_backend(backend, "input1.inp")

    # compare the output
    compare_output()

    os.chdir("../..")


@pytest.mark.parametrize("model", ["lenet", "hinet"])
def test_model_with_batch(test_dir, backend, model, batch_size):
    os.chdir(test_dir)
    model = globals()[model]

    # download the model & data & preprocessing_file
    os.system(f"wget {model['model_batch']} -O model.onnx")
    for i in range(batch_size):
        os.system(f"wget {model[f'input{i+1}']} -O input{i+1}.jpg")
    os.system(f"wget {model['preprocess']} -O preprocess.py")

    # preprocess the input
    for i in range(batch_size):
        pre_process_input(i + 1)

    # append the input
    append_np_arr(model, batch_size)

    # run the model with OnnxRuntime
    run_onnx("batch_input.npy")

    # compile the model with backend
    compile_model(backend)

    # run the model with backend
    run_backend(backend, "batch_input.inp")

    # compare the output
    compare_output()

    os.chdir("../..")
