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

import AST.AST as AST
from onnx import mapping
from onnx import TensorProto
from numbers import Number

DEBUG = False

class OnnxNode(object):
  """
  Reimplementation of NodeProto from ONNX, but in a form
  more convenient to work with from Python.
  """

  def __init__(self, node):
    self.name = str(node.name)
    self.op_type = str(node.op_type)
    self.domain = str(node.domain)
    self.attrs = dict([(attr.name,
                       translate_onnx(attr.name, convert_onnx(attr)))
                       for attr in node.attribute])
    self.inputs = list(node.input)
    self.outputs = list(node.output)
    self.node_proto = node

__onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx2seedot(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx2seedot(x),
}


def convert_onnx(attr):
  return __convert_onnx_attribute_proto(attr)


def __convert_onnx_attribute_proto(attr_proto):
  """
  Convert an ONNX AttributeProto into an appropriate Python object
  for the type.
  NB: Tensor attribute gets returned as the straight proto.
  """
  if attr_proto.HasField('f'):
    return attr_proto.f
  elif attr_proto.HasField('i'):
    return attr_proto.i
  elif attr_proto.HasField('s'):
    return str(attr_proto.s, 'utf-8') if IS_PYTHON3 else attr_proto.s
  elif attr_proto.HasField('t'):
    return attr_proto.t  # this is a proto!
  elif attr_proto.HasField('g'):
    return attr_proto.g
  elif attr_proto.floats:
    return list(attr_proto.floats)
  elif attr_proto.ints:
    return list(attr_proto.ints)
  elif attr_proto.strings:
    str_list = list(attr_proto.strings)
    if IS_PYTHON3:
      str_list = list(map(lambda x: str(x, 'utf-8'), str_list))
    return str_list
  elif attr_proto.HasField('sparse_tensor'):
    return attr_proto.sparse_tensor
  else:
    raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))

def translate_onnx(key, val):
  return __onnx_attr_translator.get(key, lambda x: x)(val)

def onnx2seedot(dtype):
  return TENSOR_TYPE_TO_SEEDOT_TYPE[_onnx_dtype(dtype)] 	

def _onnx_dtype(dtype):
  if isinstance(dtype, Number):
    onnx_dype = dtype
  elif isinstance(dtype, str):
    onnx_dype = TensorProto.DataType.Value(dtype)
  else:
    raise RuntimeError("dtype should be number or str.")
  return onnx_dype  

TENSOR_TYPE_TO_SEEDOT_TYPE = {
    int(TensorProto.FLOAT): 'float32',
    int(TensorProto.UINT8): 'uint8',
    int(TensorProto.INT8): 'int8',
    int(TensorProto.UINT16): 'uint16',
    int(TensorProto.INT16): 'int16',
    int(TensorProto.INT32): 'int32',
    int(TensorProto.INT64): 'int64',
    int(TensorProto.BOOL): 'bool',
    int(TensorProto.FLOAT16): 'float16',
    int(TensorProto.DOUBLE): 'float64',
    int(TensorProto.COMPLEX64): 'complex64',
    int(TensorProto.COMPLEX128): 'complex128',
    int(TensorProto.UINT32): 'uint32',
    int(TensorProto.UINT64): 'uint64',
    int(TensorProto.STRING): 'string'
}

def getOperatorsIdx(token):
		#TODO : remove usage of this
		return AST.Operators.convSymbolToEnumValue(token)

class ONNXNodesAST:

	# value_info: dictionary of name -> (type, dimension tuple)
	def Input(node, value_info, dictNodeNameToOutVarStr):
		if(DEBUG):
			print(node)
		return AST.Input(node.dims, onnx2seedot(node.data_type))


	def Cast(node, value_info, dictNodeNameToOutVarStr):
		node = OnnxNode(node) 
		if(DEBUG):
			print(node)
		inputsRef = node.inputs
		assert(len(inputsRef) == 1)
		destType = node.attrs['to']
		return AST.UninterpFuncCall(value_info[node.outputs[0]][1],
											'Cast', 
											[AST.ID(inputsRef[0]), 
											AST.ID(inputsRef[0]),
											AST.ID(destType)
											])

	def Relu(node, value_info, dictNodeNameToOutVarStr):
		node = OnnxNode(node) 
		if(DEBUG):
			print(node)
		inputsRef = node.inputs
		assert(len(inputsRef)==1)
		return AST.Func(getOperatorsIdx('relu'), AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]))

	def Add(node, value_info, dictNodeNameToOutVarStr):
		node = OnnxNode(node) 
		if(DEBUG):
			print(node)
		inputsRef = node.inputs
		assert(len(inputsRef) == 2)
		return AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
							getOperatorsIdx('+'),
							AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
							)

	# currently supports only 2D convolution	
	# TODO: 3D conv
	def Conv(node, value_info, dictNodeNameToOutVarStr):
		node = OnnxNode(node) 
		if(DEBUG):
			print(node)
		inputsRef = node.inputs
		assert(len(inputsRef)==2)
		
		stridesUsed = node.attrs['strides']
		assert(len(stridesUsed)==2)
		strideH = stridesUsed[0]
		strideW = stridesUsed[1]

		inputShape = value_info[inputsRef[0]][1]
		# print(inputShape)
		imgH = inputShape[2]
		imgW = inputShape[3]

		filterShape = value_info[inputsRef[1]][1]
		FH = filterShape[2]
		FW = filterShape[3]

		assert(value_info[node.inputs[1]][1][2:] == tuple(node.attrs['kernel_shape']))

		# verify this order
		[zPadHLeft, zPadHRight, zPadWLeft, zPadWRight] = paddingUsedStr = node.attrs['pads']

		options = {}
		options[AST.PaddingKeysDict.FH] = FH
		options[AST.PaddingKeysDict.FW] = FW
		options[AST.PaddingKeysDict.zPadHLeft] = zPadHLeft
		options[AST.PaddingKeysDict.zPadHRight] = zPadHRight
		options[AST.PaddingKeysDict.zPadWLeft] = zPadWLeft
		options[AST.PaddingKeysDict.zPadWRight] = zPadWRight
		options[AST.PaddingKeysDict.strideH] = strideH
		options[AST.PaddingKeysDict.strideW] = strideW	  	
		return AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]), 
								getOperatorsIdx('#'),
								AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]), 
								options)

	def Reshape(node, value_info, dictNodeNameToOutVarStr):
		node = OnnxNode(node) 
		if(DEBUG):
			print(node)
		inputsRef = node.inputs
		assert(len(inputsRef)==2)
		print("OOO", list(value_info[node.outputs[0]][1]))
		return AST.Reshape(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]), list(value_info[node.outputs[0]][1]), 0)

	def BatchNormalization(node, value_info, dictNodeNameToOutVarStr):
		node = OnnxNode(node) 
		if(DEBUG):
			print(node)
		inputsRef = node.inputs
		# Are running mean and var used for something?
		assert(len(inputsRef)==5)
		return AST.FusedBatchNorm(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
										 AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
										 AST.ID(dictNodeNameToOutVarStr[inputsRef[2]]),
										)	

	def MaxPool(node, value_info, dictNodeNameToOutVarStr):
		return ONNXNodesAST.helper_processPool(node, value_info, 'MAXPOOL')

	def AvgPool(node, value_info):
		return ONNXNodesAST.helper_processPool(node, value_info, 'AVGPOOL')

	def GlobalAveragePool(node, value_info, dictNodeNameToOutVarStr):
		node = OnnxNode(node) 
		if(DEBUG):
			print(node)
		inputsRef = node.inputs
		assert(len(inputsRef)==1)

		return AST.Pool(AST.Pool.PoolType.AvgPool,
							  AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
							  {
							  	AST.PaddingKeysDict.FH: value_info[inputsRef[0]][1][0],
							  	AST.PaddingKeysDict.FW: value_info[inputsRef[0]][1][1],
							  	AST.PaddingKeysDict.zPadHLeft: 0,
							  	AST.PaddingKeysDict.zPadHRight: 0,
							  	AST.PaddingKeysDict.zPadWLeft: 0,
							  	AST.PaddingKeysDict.zPadWRight: 0,
							  	AST.PaddingKeysDict.strideH: 1,
							  	AST.PaddingKeysDict.strideW: 1
							  }
							)	

	def helper_processPool(node, value_info, typeOfPool):
		node = OnnxNode(node) 
		if(DEBUG):
			print(node)
		inputsRef = node.inputs
		assert(len(inputsRef)==1)
				
		stridesUsed = node.attrs['strides']
		strideH = stridesUsed[0]
		strideW = stridesUsed[1]

		kSizeUsed = node.attrs['kernel_shape']
		# assert((kSizeUsed[0] == 1) and (kSizeUsed[3] == 1))
		kSizeH = kSizeUsed[0]
		kSizeW = kSizeUsed[1]
		
		inputShape = value_info[inputsRef[0]][1]
		print(inputShape)
		imgH = inputShape[2]
		imgW = inputShape[3]

		# verify order
		[zPadHLeft, zPadHRight, zPadWLeft, zPadWRight] = node.attrs['pads']

		poolType = None
		if typeOfPool=='MAXPOOL': poolType = AST.Pool.PoolType.MaxPool
		elif typeOfPool=='AVGPOOL': poolType = AST.Pool.PoolType.AvgPool
		else: 
			print("Unknown type of pooling layer.", file=sys.stderr)
			assert(False)
		return AST.Pool(poolType,
							  AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
							  {
							  	AST.PaddingKeysDict.FH: kSizeH,
							  	AST.PaddingKeysDict.FW: kSizeW,
							  	AST.PaddingKeysDict.zPadHLeft: zPadHLeft,
							  	AST.PaddingKeysDict.zPadHRight: zPadHRight,
							  	AST.PaddingKeysDict.zPadWLeft: zPadWLeft,
							  	AST.PaddingKeysDict.zPadWRight: zPadWRight,
							  	AST.PaddingKeysDict.strideH: strideH,
							  	AST.PaddingKeysDict.strideW: strideW
							  }
							)
		