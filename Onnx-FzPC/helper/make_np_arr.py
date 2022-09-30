import numpy as np
import sys
import os
import re


def convert_raw_output_to_np(filename):
    matcher = re.compile(r"[-]?[0-9]+")
    array = []
    with open(filename, "r") as f:
        for line in f:
            match = matcher.fullmatch(line.rstrip())
            if match:
                number = int(match.group(0))

                array.append(float(number))
    return np.array(array)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python make_np.py output.txt")
    output_fname = sys.argv[1]
    path = os.path.dirname(output_fname)
    np_arr = convert_raw_output_to_np(output_fname)
    np.save(path + "/output.npy", np.array(np_arr))
    print(f"Saved at {path}/output.npy")
