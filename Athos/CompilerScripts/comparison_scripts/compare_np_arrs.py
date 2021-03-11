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
import numpy as np
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--inputs",
        help="Paths of the two numpy arrays to compare. -i arr1.npy arr2.npy ",
        required=True,
        type=str,
        nargs=2,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose mode. Prints arrays.",
        action="store_true",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    arr1 = np.load(args.inputs[0], allow_pickle=True).flatten()
    arr2 = np.load(args.inputs[1], allow_pickle=True).flatten()

    matching_prec = -1
    for prec in range(1, 10):
        try:
            np.testing.assert_almost_equal(arr1, arr2, decimal=prec)
        except AssertionError:
            break
        matching_prec = prec

    if matching_prec == -1:
        print("Output mismatch")
    else:
        print("Arrays matched upto {} decimal points".format(matching_prec))

    if args.verbose:
        print(args.inputs[0], ": ", arr1)
        print(args.inputs[1], ": ", arr2)
