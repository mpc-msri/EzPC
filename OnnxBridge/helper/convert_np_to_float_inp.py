"""
Authors: Pratik Bhatu.
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
from argparse import RawTextHelpFormatter
import sys

import os
import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--inp",
        required=True,
        type=str,
        help="Path to numpy array dumped using np.save  (.npy file)",
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        help="Path where the output floatpoint numbers will be stored.",
    )
    args = parser.parse_args()
    return args


def convert_np_to_floatpt(path_to_numpy_arr):
    if not os.path.exists(path_to_numpy_arr):
        sys.exit("Numpy arr {} specified does not exist".format(path_to_numpy_arr))
    input_name = os.path.splitext(path_to_numpy_arr)[0]
    output_path = (
        (input_name + "_floatpt_scale" + ".inp") if args.output is None else args.output
    )
    np_inp = np.load(path_to_numpy_arr, allow_pickle=True)
    with open(output_path, "w") as ff:
        for xx in np.nditer(np_inp, order="C"):
            # print(xx)
            ff.write(str((xx)) + " ")
            ff.write("\n")
    return output_path


if __name__ == "__main__":
    args = parse_args()

    output_path = convert_np_to_floatpt(args.inp)
    print("FLoat point output saved in ", output_path)