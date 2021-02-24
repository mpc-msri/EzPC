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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: compare_np_arrs.py arr1.npy arr2.npy")
    arr1 = np.load(sys.argv[1], allow_pickle=True).flatten()
    arr2 = np.load(sys.argv[2], allow_pickle=True).flatten()

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
