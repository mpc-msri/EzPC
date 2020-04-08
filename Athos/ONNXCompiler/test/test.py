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


import onnx
from onnx import helper
import unittest
from onnx import TensorProto
import numpy as np
import subprocess
import common
from datetime import date
import time

class TestNode(unittest.TestCase):

	def _get_rnd_float32(self, low=-1.0, high=1.0, shape=None):
		output = np.random.uniform(low, high, shape)
		cnt = 1
		for val in shape: cnt*=val
		if shape == None:
			return np.float32(output)
		else:
			return output.astype(np.float32).reshape(cnt).tolist()

	def check_result(self, graph, name):
		current_milli_time = lambda: str(int(round(time.time() * 1000)))
		name = name + "_" + current_milli_time()
		model = onnx.helper.make_model(graph, producer_name='onnx-compiler-test')
		onnx.save(model, 'models/' + name + '.onnx')

		bashCommand = './compile.sh ' + name
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()

		# print(output)
		# print(error)

		res_onnx = common.extract_txt_to_numpy_array('debug/onnx_output.txt')	
		res_cpp = common.extract_txt_to_numpy_array('debug/cpp_output.txt')

		np.testing.assert_almost_equal(res_cpp, res_onnx, decimal=4)		


	def test_conv2d(self):
		name = "conv2d"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 3, 10, 10])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 5, 10, 10])
		node_def = helper.make_node("Conv", ['state_in', 'weight'], ['state_out'],
		                                pads=[1, 1, 1, 1], strides=[1, 1], kernel_shape=[3, 3])

		weight_shape = [5, 3, 3, 3]
		weight_val = self._get_rnd_float32(shape=weight_shape)

		weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_val)

		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        [weight]
		    )
		self.check_result(graph, name)


	def test_conv3d(self):
		name = "conv3d"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 32, 64, 256, 256])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 32, 64, 256, 256])
		node_def = helper.make_node("Conv", ['state_in', 'weight'], ['state_out'],
		                                pads=[1, 1, 1, 1, 1, 1], strides=[1, 1, 1], kernel_shape=[3, 3, 3])

		weight_shape = [32, 32, 3, 3, 3]
		weight_val = self._get_rnd_float32(shape=weight_shape)

		weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_val)

		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        [weight]
		    )
		self.check_result(graph, name)	

	def test_conv_transpose(self):
		name = "conv_transpose"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 3, 10, 10])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 5, 19, 19])
		node_def = helper.make_node("ConvTranspose", ['state_in', 'weight'], ['state_out'],
		                                pads=[1, 1, 1, 1], strides=[2, 2], kernel_shape=[3, 3])

		weight_shape = [3, 5, 3, 3]
		weight_val = self._get_rnd_float32(shape=weight_shape)

		weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_val)

		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        [weight]
		    )

		self.check_result(graph, name)

	# For this to run onnx_run_tf.py should be used in the compile script
	# since onnxruntime does not support convtranspose3d	
	def test_conv_transpose3d(self):
		name = "conv3dTranspose"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 3, 10, 10, 10])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 5, 10, 10, 10])
		node_def = helper.make_node("ConvTranspose", ['state_in', 'weight'], ['state_out'],
										# check with pads which are not 1
		                                pads=[1, 1, 1, 1, 1, 1], strides=[1, 1, 1], kernel_shape=[3, 3, 3])

		weight_shape = [3, 5, 3, 3, 3]
		weight_val = self._get_rnd_float32(shape=weight_shape)

		weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_val)

		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        [weight]
		    )
		self.check_result(graph, name)	

	def test_relu(self):
		name = "relu"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 3, 10, 10])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 3, 10, 10])
		node_def = helper.make_node("Relu", ['state_in'], ['state_out'])
		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        []
		    )
		self.check_result(graph, name)

if __name__ == '__main__':
	unittest.main()