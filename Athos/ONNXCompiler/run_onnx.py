
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

import onnx
import onnx.shape_inference
import AST.AST as AST
from ONNXNodesAST import ONNXNodesAST
from onnx.helper import make_tensor_value_info
from onnx import TensorProto
from AST.PrintAST import PrintAST 
from AST.MtdAST import MtdAST

DEBUG = False

def proto_val_to_dimension_tuple(proto_val):
	return tuple([dim.dim_value for dim in proto_val.type.tensor_type.shape.dim])

def main():
	# First read the ONNX file
	if (len(sys.argv) < 2):
		print("TF python file unspecified.", file=sys.stderr)
		exit(1)
	filePath = sys.argv[1]

	# load the model and extract the graph		
	model = onnx.load(filePath)
	graph_def = model.graph

	# Before shape inference (model.graph.value_info) should have shapes of all the variables and constants 
	model.graph.value_info.append(make_tensor_value_info(model.graph.input[0].name, TensorProto.FLOAT, proto_val_to_dimension_tuple(model.graph.input[0])))
	model.graph.value_info.append(make_tensor_value_info(model.graph.output[0].name, TensorProto.FLOAT, proto_val_to_dimension_tuple(model.graph.output[0])))
	for init_vals in model.graph.initializer:
		model.graph.value_info.append(make_tensor_value_info(init_vals.name, TensorProto.FLOAT, tuple(init_vals.dims)))

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
		value_info[val.name] = (val.type.tensor_type.elem_type ,proto_val_to_dimension_tuple(val))

	# Iterate through the ONNX graph nodes and translate them to SeeDot AST nodes	
	program = None
	innerMostLetASTNode = None
	outVarPrefix = "J"
	mtdAST = MtdAST()

	for node in graph_def.node:
		if(DEBUG):
			print("Node information")
			print(node)	
		func = getattr(ONNXNodesAST, node.op_type) 
		curAst = func(node, value_info)

		mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName : node.op_type,
							AST.ASTNode.mtdKeyTFNodeName : node.name}

		if (curAst is None):
			continue
		curOutVarStr = outVarPrefix + str(node.output)
		curOutVarAstNode = AST.ID(curOutVarStr)
		if program:
			assert(type(innerMostLetASTNode) is AST.Let)
			newNode = AST.Let(curOutVarAstNode, curAst, curOutVarAstNode)
			mtdAST.visit(newNode, mtdForCurAST)
			# Updating the innermost Let AST node and the expression for previous Let Node 
			innerMostLetASTNode.expr = newNode
			innerMostLetASTNode = newNode
		else:
			print(curAst)
			innerMostLetASTNode = AST.Let(AST.ID(curOutVarStr), curAst, curOutVarAstNode)
			mtdAST.visit(innerMostLetASTNode, mtdForCurAST)
			innerMostLetASTNode.depth = 0
			program = innerMostLetASTNode

	PrintAST().visit(program)

if __name__ == "__main__":
	main()	