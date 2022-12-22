import numpy as np
import os

arr = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
inp = np.random.rand(1, 1, 1, 3).astype(np.float32)
print(inp.shape)
output_dir = "onnx_input"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
np.save("onnx_input/custom_input.npy", inp)
print("\n".join(map(str, inp.flatten())))
