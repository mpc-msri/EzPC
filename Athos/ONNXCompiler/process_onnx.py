
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

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SeeDot')) #Add SeeDot directory to path
import _pickle as pickle
import onnx
import onnx.shape_inference
import AST.AST as AST
from ONNXNodesAST import ONNXNodesAST
from onnx.helper import make_tensor_value_info
from onnx import TensorProto
from AST.PrintAST import PrintAST 
from AST.MtdAST import MtdAST
import math
from onnx import numpy_helper
import common

import numpy as np
np.set_printoptions(threshold=np.inf)

DEBUG = False
out_var_prefix = "J"

def main():
	sys.setrecursionlimit(10000)
	# First read the ONNX file
	if (len(sys.argv) < 2):
		print("TF python file unspecified.", file=sys.stderr)
		exit(1)
	file_path = sys.argv[1]

	# load the model and extract the graph		
	model = onnx.load(file_path)
	graph_def = model.graph

	model_name_to_val_dict = {}
	# Before shape inference (model.graph.value_info) should have shapes of all the variables and constants 
	model.graph.value_info.append(make_tensor_value_info(model.graph.input[0].name, TensorProto.FLOAT, common.proto_val_to_dimension_tuple(model.graph.input[0])))
	model.graph.value_info.append(make_tensor_value_info(model.graph.output[0].name, TensorProto.FLOAT, common.proto_val_to_dimension_tuple(model.graph.output[0])))
	
	input_array = numpy.load(model_name+'_input.npy')
	(chunk, cnt) = common.numpy_float_array_to_fixed_point_val_str(input_array, scaling_factor)

	for init_vals in model.graph.initializer:
		# TODO: Remove float_data. Change this to appropriate data type. 
		model_name_to_val_dict[init_vals.name] = init_vals.float_data
		model.graph.value_info.append(make_tensor_value_info(init_vals.name, TensorProto.FLOAT, tuple(init_vals.dims)))	
		chunk += init_vals.name + ' = ' + str(numpy_helper.to_array(init_vals)) + '\n'
		(chunk_1, cnt_1) = common.numpy_float_array_to_fixed_point_val_str(numpy_helper.to_array(init_vals), scaling_factor)
		chunk += chunk_1
		cnt += cnt_1

	f = open(model_name + '_input.h', 'w') 
	f.write(chunk)
	f.close()

	print('Total ' + str(cnt) + ' integers were written in ' + model_name + '_input.h')
	preprocess_batch_normalization(graph_def, model_name_to_val_dict)

	if(DEBUG):	
		print("Shape inference *****************")
		print(model.graph.value_info)

	inferred_model = onnx.shape_inference.infer_shapes(model)
	
	if(DEBUG):	
		print("Printing shape ******************")
		print(inferred_model.graph.value_info)
		print("Done ******************")

	# value_info: dictionary of name -> (type, dimension tuple)
	value_info = {}
	for val in inferred_model.graph.value_info:
		value_info[val.name] = (val.type.tensor_type.elem_type, common.proto_val_to_dimension_tuple(val))

	# Iterate through the ONNX graph nodes and translate them to SeeDot AST nodes	
	program = None
	innermost_let_ast_node = None
	node_name_to_out_var_dict = {}
	out_var_count = 0
	mtdAST = MtdAST()

	(program, innermost_let_ast_node, out_var_count) = process_input_variables(program, innermost_let_ast_node, node_name_to_out_var_dict, out_var_count, mtdAST, graph_def, value_info)

	process_onnx_nodes(innermost_let_ast_node, node_name_to_out_var_dict, out_var_count, mtdAST, graph_def, value_info)

	PrintAST().visit(program)	
	
	common.write_debug_info(node_name_to_out_var_dict)

	with open('astOutput.pkl', 'wb') as f:
		pickle.dump(program, f)

def process_input_variables(program, innermost_let_ast_node, node_name_to_out_var_dict, out_var_count, mtdAST, graph_def, value_info):
	node = graph_def.input[0]
	curAst = ONNXNodesAST.Input(node, value_info, node_name_to_out_var_dict)
	mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName : 'Input',
						AST.ASTNode.mtdKeyTFNodeName : node.name}
	cur_out_var_ast_node = AST.ID(node.name)	

	if program:
		assert(type(innermost_let_ast_node) is AST.Let)
		newNode = AST.Let(cur_out_var_ast_node, curAst, cur_out_var_ast_node)
		mtdAST.visit(newNode, mtdForCurAST)
		# Updating the innermost Let AST node and the expression for previous Let Node 
		innermost_let_ast_node.expr = newNode
		innermost_let_ast_node = newNode
	else:
		innermost_let_ast_node = AST.Let(cur_out_var_ast_node, curAst, cur_out_var_ast_node)
		mtdAST.visit(innermost_let_ast_node, mtdForCurAST)
		innermost_let_ast_node.depth = 0
		program = innermost_let_ast_node

	node_name_to_out_var_dict[node.name] = node.name

	for node in graph_def.initializer:
		if(DEBUG):
			print("Node information")
			print(node)	
	
		curAst = ONNXNodesAST.Input(node, value_info, node_name_to_out_var_dict)
		mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName : 'Input',
							AST.ASTNode.mtdKeyTFNodeName : node.name}
		if (curAst is None):
			continue		
	
		cur_out_var_ast_node = AST.ID(node.name)	

		if program:
			assert(type(innermost_let_ast_node) is AST.Let)
			newNode = AST.Let(cur_out_var_ast_node, curAst, cur_out_var_ast_node)
			mtdAST.visit(newNode, mtdForCurAST)
			# Updating the innermost Let AST node and the expression for previous Let Node 
			innermost_let_ast_node.expr = newNode
			innermost_let_ast_node = newNode
		else:
			innermost_let_ast_node = AST.Let(cur_out_var_ast_node, curAst, cur_out_var_ast_node)
			mtdAST.visit(innermost_let_ast_node, mtdForCurAST)
			innermost_let_ast_node.depth = 0
			program = innermost_let_ast_node
	
		node_name_to_out_var_dict[node.name] = node.name
	return (program, innermost_let_ast_node, out_var_count)	

def process_onnx_nodes(innermost_let_ast_node, node_name_to_out_var_dict, out_var_count, mtdAST, graph_def, value_info):	
	for node in graph_def.node:
		if(DEBUG):
			print("Node information")
			print(node)	

		

		mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName : node.op_type,
							AST.ASTNode.mtdKeyTFNodeName : node.name}

		func = getattr(ONNXNodesAST, node.op_type) 
		(innermost_let_ast_node, out_var_count) = func(node, value_info, node_name_to_out_var_dict, innermost_let_ast_node, out_var_count, mtdAST)					


		assert(type(innermost_let_ast_node) is AST.Let)

def get_seedot_shape_order(old_shape):
	if(len(old_shape) == 4):
		# Case when spatial dimension is 2
		return ([old_shape[0], old_shape[2], old_shape[3], old_shape[1]], [1, 3, 4, 2])	
	else:
		# Casr when spatial dimension is 3 	
		return ([old_shape[0], old_shape[2], old_shape[3], old_shape[4], old_shape[1]], [1, 3, 4, 5, 2])

def preprocess_batch_normalization(graph_def, model_name_to_val_dict):
	# set names to graph nodes if not present
	for node in graph_def.node: 
		node.name = node.output[0]
		# Update the batch normalization scale and B
		# so that mean and var are not required
		if(node.op_type == 'BatchNormalization'):
			# scale
			gamma = model_name_to_val_dict[node.input[1]]
			# B
			beta = model_name_to_val_dict[node.input[2]]
			mean = model_name_to_val_dict[node.input[3]]
			var = model_name_to_val_dict[node.input[4]]
			for i in range(len(gamma)):
				rsigma = 1/math.sqrt(var[i]+1e-5)
				gamma[i] = gamma[i]*rsigma
				beta[i] = beta[i]-gamma[i]*mean[i]	
				mean[i] = 0
				var[i] = 1-1e-5

	# Just testing if the correct values are put			
	model_name_to_val_dict2 = {}
	for init_vals in graph_def.initializer:
		# TODO: Remove float_data
		model_name_to_val_dict2[init_vals.name] = init_vals.float_data		
	for node in graph_def.node: 
		node.name = node.output[0]
		if(node.op_type == 'BatchNormalization'):
			mean = model_name_to_val_dict[node.input[3]]
			for val in mean:
				assert(val == 0)

if __name__ == "__main__":
	main()				