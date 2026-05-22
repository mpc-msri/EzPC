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

import onnx_graphsurgeon as gs
import numpy as np
import onnx

# W_val = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# bias = [1, 2, 3]

# X = gs.Variable(name="X", dtype=np.float32, shape=(1, 1, 1, 3))
# # Since W is a Constant, it will automatically be exported as an initializer
# W = gs.Constant(name="W", values=np.array(W_val, dtype=np.float32))
# bias = gs.Constant(name="bias", values=np.array(bias, dtype=np.float32))

# Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 3))
# Z = gs.Variable(name="Z", dtype=np.float32, shape=(1, 3))

# node = gs.Node(op="Flatten", inputs=[X], outputs=[Y])
# node2 = gs.Node(op="Gemm", inputs=[Y, W, bias], outputs=[Z])

# # Note that initializers do not necessarily have to be graph inputs
# graph = gs.Graph(nodes=[node, node2], inputs=[X], outputs=[Z])
# onnx.save(gs.export_onnx(graph), "onnx_new_files/fc_bias.onnx")

# X = gs.Variable(name="X", dtype=np.float32, shape=(1, 1, 3, 3))
# # Since W is a Constant, it will automatically be exported as an initializer
# W = gs.Constant(name="W", values=np.ones(shape=(1, 1, 2, 2), dtype=np.float32))

# Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 1, 2, 2))

# node = gs.Node(op="Conv", inputs=[X, W], attrs={"strides":[1,1],"kernel_shape":[2,2],"pads":[0,0,0,0]}, outputs=[Y])

# # Note that initializers do not necessarily have to be graph inputs
# graph = gs.Graph(nodes=[node], inputs=[X], outputs=[Y])
# onnx.save(gs.export_onnx(graph), "onnx_files/conv.onnx")

# writing chexpert model node by node with initializers value as random

# input = gs.Variable(name="input", dtype=np.float32, shape=(1, 3, 320, 320))
# # Since W is a Constant, it will automatically be exported as an initializer
# W = gs.Constant(name="W", values=np.random.rand(64, 3, 7, 7).astype(np.float32))
# bias = gs.Constant(name="bias", values=np.random.rand(64).astype(np.float32))
# conv_out = gs.Variable(name="conv_out", dtype=np.float32, shape=(1, 64, 160, 160))

# node = gs.Node(op="Conv", inputs=[input, W, bias], attrs={"strides": [2, 2], "kernel_shape": [7, 7], "pads": [3, 3, 3, 3]}, outputs=[conv_out])

# relu_out = gs.Variable(name="relu_out", dtype=np.float32, shape=(1, 64, 160, 160))
# node2 = gs.Node(op="Relu", inputs=[conv_out], outputs=[relu_out])

# pool_out = gs.Variable(name="pool_out", dtype=np.float32, shape=(1, 64, 80, 80))
# node3 = gs.Node(op="AveragePool", inputs=[relu_out], attrs={"strides": [2, 2], "kernel_shape": [2, 2], "pads": [0, 0, 0, 0]}, outputs=[pool_out])

# concat_out = gs.Variable(name="concat_out", dtype=np.float32, shape=(1, 64, 80, 80))
# node4 = gs.Node(op="Concat", inputs=[pool_out], attrs={"axis": 1}, outputs=[concat_out])

# batchnorm_out = gs.Variable(name="batchnorm_out", dtype=np.float32, shape=(1, 64, 80, 80))
# scale = gs.Constant(name="scale", values=np.random.rand(64).astype(np.float32))
# B = gs.Constant(name="bias1", values=np.random.rand(64).astype(np.float32))
# mean = gs.Constant(name="mean", values=np.random.rand(64).astype(np.float32))
# var = gs.Constant(name="var", values=np.random.rand(64).astype(np.float32))
# node5 = gs.Node(op="BatchNormalization", inputs=[concat_out, scale, B, mean, var], outputs=[batchnorm_out])

# graph = gs.Graph(nodes=[node, node2, node3,], inputs=[input], outputs=[pool_out])
# onnx.save(gs.export_onnx(graph), "mod_chexpert.onnx")


# writing a model with just one maxpool layer
# input = gs.Variable(name="input", dtype=np.float32, shape=(1, 3, 320, 320))
# pool_out = gs.Variable(name="pool_out", dtype=np.float32, shape=(1, 3, 160, 160))
# node = gs.Node(op="MaxPool", inputs=[input], attrs={"strides": [2, 2], "kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}, outputs=[pool_out])
# graph = gs.Graph(nodes=[node], inputs=[input], outputs=[pool_out])
# onnx.save(gs.export_onnx(graph), "maxpool.onnx")


# #writing a model with just one transpose layer of shape 2,3,3
# input = gs.Variable(name="input", dtype=np.float32, shape=(1, 2,3, 3))
# transpose_out = gs.Variable(name="transpose_out", dtype=np.float32, shape=(3,1,2, 3))
# node = gs.Node(op="Transpose", inputs=[input], attrs={"perm": [3, 0,1, 2]}, outputs=[transpose_out])
# graph = gs.Graph(nodes=[node], inputs=[input], outputs=[transpose_out])
# onnx.save(gs.export_onnx(graph), "transpose.onnx")

# writing a model with just one relu layer of shape 3,3
# input = gs.Variable(name="input", dtype=np.float32, shape=(3, 3))
# transpose_out = gs.Variable(name="relu_out", dtype=np.float32, shape=(3, 3))
# node = gs.Node(op="LeakyRelu", inputs=[input], outputs=[transpose_out])
# graph = gs.Graph(nodes=[node], inputs=[input], outputs=[transpose_out])
# onnx.save(gs.export_onnx(graph), "leakyrelu.onnx")


# #writing a model with just one convtranspose of shape 1,8,8,8
# input = gs.Variable(name="input", dtype=np.float32, shape=(1, 1, 2, 2))
# transpose_out = gs.Variable(name="transpose_out", dtype=np.float32, shape=(1, 1, 3, 3))
# W = gs.Constant(name="W", values=np.random.rand(1, 1, 2, 2).astype(np.float32))
# node = gs.Node(op="ConvTranspose", inputs=[input, W], attrs={"strides": [1, 1], "kernel_shape": [2,2], "pads": [0, 0, 0, 0], "group": 1, "dilations": [1, 1]}, outputs=[transpose_out])
# graph = gs.Graph(nodes=[node], inputs=[input], outputs=[transpose_out])
# onnx.save(gs.export_onnx(graph), "convtranspose.onnx")


# #writing a model with just one conv3d of shape 1,1,64,64,64
input = gs.Variable(name="input", dtype=np.float32, shape=(1, 1, 6, 6))
output = gs.Variable(name="output", dtype=np.float32, shape=(1, 2, 6, 6))
W = gs.Constant(name="W", values=np.random.rand(2, 1, 3, 3).astype(np.float32))
node = gs.Node(
    op="Conv",
    inputs=[input, W],
    attrs={
        "strides": [1, 1],
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "group": 1,
        "dilations": [1, 1],
    },
    outputs=[output],
)
graph = gs.Graph(nodes=[node], inputs=[input], outputs=[output])
onnx.save(gs.export_onnx(graph), "conv2d.onnx")
