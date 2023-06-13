import os
import numpy as np

# Get the directory path where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
ezpc_dir = os.path.join(script_directory, "..", "..")


def pre_process_input():
    # check if input.jpg and preprocess.py exists
    assert os.path.exists("input.jpg")
    assert os.path.exists("preprocess.py")

    # convert jpg -> npy
    os.system("python3 preprocess.py input.jpg")
    assert os.path.exists("input.npy")

    # convert npy -> inp
    os.system(
        f"python3 {ezpc_dir}/OnnxBridge/helper/convert_np_to_float_inp.py --inp input.npy --out input.inp"
    )
    assert os.path.exists("input.inp")


def run_onnx():
    # check if model.onnx and input.npy exists
    assert os.path.exists("model.onnx")
    assert os.path.exists("input.npy")

    # run the model with OnnxRuntime
    os.system(f"python3 {ezpc_dir}/OnnxBridge/helper/run_onnx.py model.onnx input.npy")
    assert os.path.exists("onnx_output/expected.npy")


def compile_model(backend):
    # check if model.onnx exists
    assert os.path.exists("model.onnx")

    # compile the model
    if backend == "LLAMA" or backend == "CLEARTEXT_LLAMA":
        os.system(
            f"python3 {ezpc_dir}/OnnxBridge/main.py --path model.onnx --generate executable --backend {backend} --scale 15 --bitlength 40 "
        )
    elif backend == "SECFLOAT" or backend == "SECFLOAT_CLEARTEXT":
        os.system(
            f"python3 {ezpc_dir}/OnnxBridge/main.py --path model.onnx --generate executable --backend {backend} "
        )


def run_backend(backend):
    # check if model.onnx and input.inp exists
    assert os.path.exists("model.onnx")
    assert os.path.exists("input.inp")

    raw_output = os.path.join("raw_output.txt")
    # run the model with backend
    if backend == "CLEARTEXT_LLAMA":
        # check if model compiled
        assert os.path.exists("model_CLEARTEXT_LLAMA_15")
        assert os.path.exists("model_input_weights.dat")

        os.system(
            f"./model_CLEARTEXT_LLAMA_15 0 model_input_weights.dat < input.inp > {raw_output}"
        )
    elif backend == "LLAMA":
        # check if model compiled
        assert os.path.exists("model_LLAMA_15")
        assert os.path.exists("model_input_weights.dat")

        # running dealer
        os.system(f"./model_LLAMA_15 1")

        # running server
        os.system(f"./model_LLAMA_15 2 model_input_weights.dat &")

        # running client
        os.system(f"./model_LLAMA_15 3 127.0.0.1 < input.inp > {raw_output}")

    elif backend == "SECFLOAT_CLEARTEXT":
        # check if model compiled
        assert os.path.exists("model_secfloat_ct")
        assert os.path.exists("model_input_weights.inp")

        os.system(
            f"cat input.inp model_input_weights.inp | ./model_secfloat_ct > {raw_output}"
        )

    # save the raw output as npy
    os.system(f"python3 {ezpc_dir}/OnnxBridge/helper/make_np_arr.py {raw_output}")
    assert os.path.exists("output.npy")


def compare_output():
    # check if output.npy and expected.npy exists
    assert os.path.exists("output.npy")
    assert os.path.exists("onnx_output/expected.npy")

    # compare the output
    arr1 = np.load("output.npy", allow_pickle=True).flatten()
    arr2 = np.load("onnx_output/expected.npy", allow_pickle=True).flatten()

    matching_prec = -1
    for prec in range(1, 10):
        try:
            np.testing.assert_almost_equal(arr1, arr2, decimal=prec)
        except AssertionError:
            break
        matching_prec = prec

    print("Secure Output: " + str(arr1))
    print("Expected Output: " + str(arr2))

    if matching_prec == -1:
        print("Output mismatch")
    else:
        print("Arrays matched upto {} decimal points".format(matching_prec))

    assert matching_prec != -1
