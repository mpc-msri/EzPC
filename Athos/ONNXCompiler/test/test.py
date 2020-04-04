'''

Authors: Shubham Ugare.

Copyright:
Copyright (c) 2018 Microsoft Research
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

'''


# start with testing convTranspose
from onnx import helper
import unittest
from onnx import TensorProto
import numpy as np

class TestNode(unittest.TestCase):

	def _get_rnd_float32(self, low=-1.0, high=1.0, shape=None):
		output = np.random.uniform(low, high, shape)
		if shape == None:
			return np.float32(output)
		else:
			return output.astype(np.float32)

	def test_conv_transpose(self):
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1])
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1])
		node_def = helper.make_node("ConvTranspose", ["X", "weights"], ["Y"],
		                                pads=[1, 1])
		X_shape = [1, 5, 4]
		X_val = self._get_rnd_float32(shape=X_shape)
		weight_shape = [5, 3, 2]
		weights_val = self._get_rnd_float32(shape=weight_shape)
		
		X = helper.make_tensor('X', TensorProto.FLOAT, (), (X_val, ))
		weights = helper.make_tensor('weights', TensorProto.FLOAT, (), (weights_val, ))

		graph = helper.make_graph(
		        [state_in, state_out, node_def],
		        "convTranspose",
		        [state_in],
		        [state_out],
		        [X, weights]
		    )

		model = onnx.helper.make_model(graph, producer_name='backend-test')
		onnx.save(model, '../models/test.onnx')

if __name__ == '__main__':
	unittest.main()