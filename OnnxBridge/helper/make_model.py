import onnx_graphsurgeon as gs
import numpy as np
import onnx

W_val = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
bias = [1, 2, 3]

X = gs.Variable(name="X", dtype=np.float32, shape=(1, 1, 1, 3))
# Since W is a Constant, it will automatically be exported as an initializer
W = gs.Constant(name="W", values=np.array(W_val, dtype=np.float32))
bias = gs.Constant(name="bias", values=np.array(bias, dtype=np.float32))

Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 3))
Z = gs.Variable(name="Z", dtype=np.float32, shape=(1, 3))

node = gs.Node(op="Flatten", inputs=[X], outputs=[Y])
node2 = gs.Node(op="Gemm", inputs=[Y, W, bias], outputs=[Z])

# Note that initializers do not necessarily have to be graph inputs
graph = gs.Graph(nodes=[node, node2], inputs=[X], outputs=[Z])
onnx.save(gs.export_onnx(graph), "onnx_new_files/fc_bias.onnx")

# X = gs.Variable(name="X", dtype=np.float32, shape=(1, 1, 3, 3))
# # Since W is a Constant, it will automatically be exported as an initializer
# W = gs.Constant(name="W", values=np.ones(shape=(1, 1, 2, 2), dtype=np.float32))

# Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 1, 2, 2))

# node = gs.Node(op="Conv", inputs=[X, W], attrs={"strides":[1,1],"kernel_shape":[2,2],"pads":[0,0,0,0]}, outputs=[Y])

# # Note that initializers do not necessarily have to be graph inputs
# graph = gs.Graph(nodes=[node], inputs=[X], outputs=[Y])
# onnx.save(gs.export_onnx(graph), "onnx_files/conv.onnx")
