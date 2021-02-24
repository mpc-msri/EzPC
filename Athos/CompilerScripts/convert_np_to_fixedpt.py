import argparse
from argparse import RawTextHelpFormatter

import parse_config
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--inp",
        required=True,
        type=str,
        help="Path to numpy array dumped using np.save  (.npy file)",
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Path to the config json file"
    )
    args = parser.parse_args()
    return args


def convert_np_to_fixedpt(path_to_numpy_arr, scaling_factor):
    if not os.path.exists(path_to_numpy_arr):
        sys.exit("Numpy arr {} specified does not exist".format(path_to_numpy_arr))
    input_name = os.path.splitext(path_to_numpy_arr)[0]
    output_path = input_name + "_fixedpt_scale_" + str(scaling_factor) + ".inp"

    np_inp = np.load(path_to_numpy_arr, allow_pickle=True)
    with open(output_path, "w") as ff:
        for xx in np.nditer(np_inp, order="C"):
            ff.write(str(int(xx * (1 << scaling_factor))) + " ")
        ff.write("\n")
    return output_path


if __name__ == "__main__":
    args = parse_args()
    params = parse_config.get_params(args.config)
    scale = 12 if params["scale"] is None else params["scale"]
    output_path = convert_np_to_fixedpt(args.inp, scale)
    print("Fixed point output saved in ", output_path)
