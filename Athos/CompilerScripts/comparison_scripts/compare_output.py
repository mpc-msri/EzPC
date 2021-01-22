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


def extract_txt_to_numpy_array(file, sf):
    f = open(file, "r")
    op = [float(int(line.rstrip())) / (2 ** sf) for line in f]
    f.close()
    return np.array(op, dtype=np.float32)


def extract_float_txt_to_numpy_array(file):
    f = open(file, "r")
    op = [float(line.rstrip()) for line in f]
    f.close()
    return np.array(op, dtype=np.float32)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: compare_output.py floating_point.txt fixed_point.txt SCALING_FACTOR PRECISION"
        )
    assert len(sys.argv) == 5
    sf = int(sys.argv[3])
    inp1 = extract_float_txt_to_numpy_array(sys.argv[1])
    inp2 = extract_txt_to_numpy_array(sys.argv[2], sf)
    prec = int(sys.argv[4])
    np.testing.assert_almost_equal(inp1, inp2, decimal=prec)
