'''

Authors: Shubham Ugare.

Copyright:
Copyright (c) 2020 Microsoft Research
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
from onnx import helper, numpy_helper
import unittest
from onnx import TensorProto
import numpy as np
import subprocess
import common
from datetime import date
import time
import hashlib

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

		old_hash = hashlib.md5(open('debug/cpp_output.txt','rb').read()).hexdigest()

		bashCommand = './compile.sh ' + name
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()

		print(output)
		print(error)
		new_hash = hashlib.md5(open('debug/cpp_output.txt','rb').read()).hexdigest()

		self.assertNotEqual(old_hash, new_hash, 'the compilation did not terminate')	

		res_onnx = common.extract_txt_to_numpy_array('debug/onnx_output.txt')	
		res_cpp = common.extract_txt_to_numpy_array('debug/cpp_output.txt')

		np.save('res_onnx', res_onnx)
		np.save('res_cpp', res_cpp)

		self.assertIsNone(error, 'error is non None')
		np.testing.assert_almost_equal(res_cpp, res_onnx, decimal=4)		


	def test_conv2d(self):
		name = "conv2d"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 3, 10, 10])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 6, 5, 5])
		node_def = helper.make_node("Conv", ['state_in', 'weight'], ['state_out'],
		                                pads=[1, 1, 1, 1], strides=[2, 2], kernel_shape=[3, 3], group=3)

		weight_shape = [6, 1, 3, 3]
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
		state_in = helper.make_tensor_value_info('state_in',TensorProto.FLOAT, [1, 2, 4, 16, 16])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 2, 4, 16, 16])
		node_def = helper.make_node("Conv", ['state_in', 'weight'], ['state_out'],
		                                pads=[1, 1, 1, 1, 1, 1], strides=[1, 1, 1], kernel_shape=[3, 3, 3])

		weight_shape = [2, 2, 3, 3, 3]
		weight_val = self._get_rnd_float32(shape=weight_shape)
		np.save('weight', weight_val)

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
		                                               TensorProto.FLOAT, [1, 5, 19, 19, 19])
		node_def = helper.make_node("ConvTranspose", ['state_in', 'weight', 'bias'], ['state_out'],
										# check with pads which are not 1
		                                pads=[1, 1, 1, 1, 1, 1], strides=[2, 2, 2], kernel_shape=[3, 3, 3])

		weight_shape = [3, 5, 3, 3, 3]
		weight_val = self._get_rnd_float32(shape=weight_shape)
		bias_shape = [5]
		bias_val = self._get_rnd_float32(shape=bias_shape)

		weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_val)
		bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_val)

		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        [weight, bias]
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

	def test_pad(self):
		name = "pad"
		state_in = helper.make_tensor_value_info('state_in', TensorProto.FLOAT, [1, 3, 10, 10])
		pads  = helper.make_tensor_value_info('pads', TensorProto.INT64, [8])
		pad_init = numpy_helper.from_array(np.array([0,0,1,1,0,0,1,1], dtype=int), name='pads')
		const_val  = helper.make_tensor_value_info('const_val', TensorProto.FLOAT, [1])
		const_val_init = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name='const_val')
		state_out  = helper.make_tensor_value_info('state_out', TensorProto.FLOAT, [1,3,12,12])
		node_def = helper.make_node("Pad", ['state_in', 'pads', 'const_val'], ['state_out'], mode="constant")
		graph = helper.make_graph([node_def],name,[state_in, pads, const_val],[state_out],initializer=[pad_init, const_val_init])
		self.check_result(graph, name)


	def test_relu3d(self):
		name = "relu3d"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 3, 7, 7, 7])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 3, 7, 7, 7])
		node_def = helper.make_node("Relu", ['state_in'], ['state_out'])
		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        []
		    )
		self.check_result(graph, name)	

	def test_reducemean(self):
		name = "reducemean"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 1024, 7, 7])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 1024])
		node_def = helper.make_node("ReduceMean", ['state_in'], ['state_out'], axes=[2,3], keepdims=0)
		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        []
		    )
		self.check_result(graph, name)

	def test_batchnormalization(self):
		name = "batchnormalization"
		state_in = helper.make_tensor_value_info('state_in',
		                                             TensorProto.FLOAT, [1, 24, 10, 10])
		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, [1, 24, 10, 10])
		node_def = helper.make_node("BatchNormalization", ['state_in', 'weight', 'bias','mean','var'], ['state_out'],
		                                momentum=0.8999999761581421)

		weight_shape = [24]
		weight_val = self._get_rnd_float32(shape=weight_shape)
		weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_val)

		bias_shape = [24]
		bias_val = self._get_rnd_float32(shape=weight_shape)
		bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_val)

		mean_shape = [24]
		mean_val = self._get_rnd_float32(shape=weight_shape)
		mean = helper.make_tensor('mean', TensorProto.FLOAT, mean_shape, mean_val)


		var_shape = [24]
		var_val = self._get_rnd_float32(shape=weight_shape, low=0, high=1)
		var = helper.make_tensor('var', TensorProto.FLOAT, var_shape, var_val)

		graph = helper.make_graph(
		        [node_def],
		        name,
		        [state_in],
		        [state_out],
		        [weight, bias, mean, var]
		    )
		self.check_result(graph, name)	

if __name__ == '__main__':
	unittest.main()
