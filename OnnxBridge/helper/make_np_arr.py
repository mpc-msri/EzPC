"""
Authors: Saksham Gupta.
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
import os
import re


def convert_raw_output_to_np(filename):
    matcher = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
    array = []
    with open(filename, "r") as f:
        for line in f:
            numbers = line.split(" ")
            for number in numbers:
                match = matcher.fullmatch(number.rstrip())
                if match:
                    number = match.group(0)
                    array.append(float(number))
    return np.array(array)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python make_np.py output.txt")
    output_fname = sys.argv[1]
    path = os.path.dirname(output_fname)
    np_arr = convert_raw_output_to_np(output_fname)
    np.save(path + "output.npy", np.array(np_arr))
    print(f"Saved at {path}/output.npy")
