import numpy as np
import sys

def extract_txt_to_numpy_array(file, sf):
    f = open(file, 'r')
    op = [float(int(line.rstrip()))/(2**sf) for line in f]
    f.close()
    return np.array(op, dtype=np.float32)

def extract_float_txt_to_numpy_array(file):
    f = open(file, 'r')
    op = [float(line.rstrip()) for line in f]
    f.close()
    return np.array(op, dtype=np.float32)

if __name__ == "__main__":
    if (len(sys.argv) != 5):
    	print("Usage: compare_output.py floating_point.txt fixed_point.txt SCALING_FACTOR PRECISION")
    assert(len(sys.argv) == 5)
    sf = int(sys.argv[3])
    inp1 = extract_float_txt_to_numpy_array(sys.argv[1])
    inp2 = extract_txt_to_numpy_array(sys.argv[2], sf)
    prec = int(sys.argv[4])
    np.testing.assert_almost_equal(inp1, inp2, decimal=prec)
