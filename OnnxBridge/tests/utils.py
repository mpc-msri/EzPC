import os
import numpy as np

# Get the directory path where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
ezpc_dir = os.path.join(script_directory, "..", "..")


def pre_process_input(i):
    # check if input.jpg and preprocess.py exists
    assert os.path.exists(f"input{i}.jpg")
    assert os.path.exists("preprocess.py")

    # convert jpg -> npy
    os.system(f"python3 preprocess.py input{i}.jpg")
    assert os.path.exists(f"input{i}.npy")

    # convert npy -> inp
    os.system(
        f"python3 {ezpc_dir}/OnnxBridge/helper/convert_np_to_float_inp.py --inp input{i}.npy --out input{i}.inp"
    )
    assert os.path.exists(f"input{i}.inp")


def run_onnx(input):
    # check if model.onnx and input.npy exists
    assert os.path.exists("model.onnx")
    assert os.path.exists(input)

    # run the model with OnnxRuntime
    os.system(f"python3 {ezpc_dir}/OnnxBridge/helper/run_onnx.py model.onnx {input}")
    assert os.path.exists("onnx_output/expected.npy")


def compile_model(backend):
    # check if model.onnx exists
    assert os.path.exists("model.onnx")

    # compile the model
    if backend == "LLAMA" or backend == "CLEARTEXT_LLAMA":
        os.system(
            f"python3 {ezpc_dir}/OnnxBridge/main.py --path model.onnx --generate executable --backend {backend} --scale 15 --bitlength 40 "
        )
    elif backend == "CLEARTEXT_fp":
        os.system(
            f"python3 {ezpc_dir}/OnnxBridge/main.py --path model.onnx --generate executable --backend {backend} "
        )
    elif backend == "SECFLOAT" or backend == "SECFLOAT_CLEARTEXT":
        os.system(
            f"python3 {ezpc_dir}/OnnxBridge/main.py --path model.onnx --generate executable --backend {backend} "
        )


def run_backend(backend, input):
    # check if model.onnx and input.inp exists
    assert os.path.exists("model.onnx")
    assert os.path.exists(input)

    raw_output = os.path.join("raw_output.txt")
    # run the model with backend
    if backend == "CLEARTEXT_LLAMA":
        # check if model compiled
        assert os.path.exists("model_CLEARTEXT_LLAMA_15")
        assert os.path.exists("model_input_weights.dat")

        os.system(
            f"./model_CLEARTEXT_LLAMA_15 0 model_input_weights.dat < {input} > {raw_output}"
        )
    elif backend == "CLEARTEXT_fp":
        # check if model compiled
        assert os.path.exists("model_CLEARTEXT_fp_0")
        assert os.path.exists("model_input_weights.dat")

        os.system(
            f"./model_CLEARTEXT_fp_0 0 model_input_weights.dat < {input} > {raw_output}"
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
        os.system(f"./model_LLAMA_15 3 127.0.0.1 < {input} > {raw_output}")

    elif backend == "SECFLOAT_CLEARTEXT":
        # check if model compiled
        assert os.path.exists("model_secfloat_ct")
        assert os.path.exists("model_input_weights.inp")

        os.system(
            f"cat {input} model_input_weights.inp | ./model_secfloat_ct > {raw_output}"
        )

    elif backend == "SECFLOAT":
        # check if model compiled
        assert os.path.exists("model_secfloat")
        assert os.path.exists("model_input_weights.inp")

        # running server
        os.system(f"./model_secfloat r=2 < model_input_weights.inp &")

        # running client
        os.system(f"./model_secfloat r=1 < {input} > {raw_output}")

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

    print("Secure Output Shape: " + str(arr1.shape))
    print("Expected Output Shape: " + str(arr2.shape))

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


# function to append n numpy array as a single numpy array
def append_np_arr(model, n):
    # assert model dictionary has n fields starting with 'input'
    for i in range(n):
        assert f"input{i+1}" in model
    # assert all the input files exist
    for i in range(n):
        assert os.path.exists(f"input{i+1}.npy")

    # append all the input files on 1st dimension i.e batch dimension
    for i in range(n):
        out = np.load(f"input{i+1}.npy")
        if i == 0:
            final_out = out
        else:
            final_out = np.append(final_out, out, axis=0)

    # save the final output
    np.save("batch_input.npy", final_out)

    # convert npy -> inp
    os.system(
        f"python3 {ezpc_dir}/OnnxBridge/helper/convert_np_to_float_inp.py --inp batch_input.npy --out batch_input.inp"
    )
    assert os.path.exists(f"batch_input.inp")
