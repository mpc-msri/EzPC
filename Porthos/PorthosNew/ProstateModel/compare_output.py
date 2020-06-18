import numpy as np
import sys

def extract_txt_to_numpy_array(file, sf):
    f = open(file, 'r')
    op = [int(line.rstrip())/(2**sf) for line in f]
    f.close()
    return np.array(op, dtype=np.float32)

if __name__ == "__main__":
    if len(sys.argv) > 3 :
        sf = int(sys.argv[3])
    else:
        sf = 24
    inp1 = extract_txt_to_numpy_array(sys.argv[1], sf)
    inp2 = extract_txt_to_numpy_array(sys.argv[2], sf)
    np.testing.assert_almost_equal(inp1, inp2, decimal=4)
