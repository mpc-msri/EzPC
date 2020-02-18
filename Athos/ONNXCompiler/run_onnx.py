
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SeeDot')) #Add SeeDot directory to path

import onnx
import onnx.shape_inference
import AST.AST as AST
from ONNXNodesAST import ONNXNodesAST
# from AST.WriteSeeDot import WriteSeeDot
from onnx.helper import make_tensor_value_info
from onnx import TensorProto

# import torch
# import onnxruntime
# Load the ONNX model
model = onnx.load("resnet50v1.onnx")

# Check that the IR is well formed
print(onnx.checker.check_model(model))

graph_def = model.graph

print(dir(model.graph.input[0]))
print(model.graph.input[0].name)
print(model.graph.input[0].type.tensor_type.elem_type)
print(model.graph.input[0].type.tensor_type.shape.dim)
# print(tuple(model.graph.input[0].type.tensor_type.shape.dim.dim_value))

def proto_val_to_dimension_tuple(proto_val):
	return tuple([dim.dim_value for dim in proto_val.type.tensor_type.shape.dim])

model.graph.value_info.append(make_tensor_value_info(model.graph.input[0].name, TensorProto.FLOAT, proto_val_to_dimension_tuple(model.graph.input[0])))

init_names = [init_vals.name for init_vals in model.graph.initializer]

for init_vals in model.graph.initializer:
	model.graph.value_info.append(make_tensor_value_info(init_vals.name, TensorProto.FLOAT, tuple(init_vals.dims)))
	print(tuple(init_vals.dims))

print("Shape inference *****************")
print(model.graph.value_info)

print("Printing shape ******************")
inferred_model = onnx.shape_inference.infer_shapes(model)
print(inferred_model.graph.value_info)

print("Done ******************")
# for value_info in graph_def.input:
# 	print(value_info)

# for node in model.graph.node:
# 	print(node)


# value_info: dictionary of name -> (type, dimension tuple)
value_info = {}
for val in inferred_model.graph.value_info:
	# print("lol", val.name)
	value_info[val.name] = (val.type.tensor_type.elem_type ,proto_val_to_dimension_tuple(val))


program = None
innerMostLetASTNode = None
outVarPrefix = "J"

for node in graph_def.node:
	print("Node information")
	print(node)	
	func = getattr(ONNXNodesAST, node.op_type) 
	curAst = func(node, value_info)

	if (curAst is None):
		continue
	curOutVarStr = outVarPrefix + str(node.output)
	curOutVarAstNode = AST.ID(curOutVarStr)
	if program:
		assert(type(innerMostLetASTNode) is AST.Let)
		newNode = AST.Let(curOutVarAstNode, curAst, curOutVarAstNode)
		# Updating the innermost Let AST node and the expression for previous Let Node 
		innerMostLetASTNode.expr = newNode
		innerMostLetASTNode = newNode
	else:
		print(curAst)
		innerMostLetASTNode = AST.Let(AST.ID(curOutVarStr), curAst, curOutVarAstNode)
		innerMostLetASTNode.depth = 0
		program = innerMostLetASTNode

WriteSeeDot().visit(program)


# Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

# session = onnxruntime.InferenceSession("prostate.onnx")

# dummy_input = torch.randn(1, 1, 64, 256, 256).cuda()

# pred = session.run(None, dummy_input)


