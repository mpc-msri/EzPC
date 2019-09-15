from enum import Enum, auto

OperatorsSymbolDict = {
		"ADD": '+',
		"SUB": '-',
		"MUL": '*',
		"CONV": '#',
		"RELU": 'relu',
		"Equal": '==',
		"ElemWiseMul":'.*',
		"ElemWiseDiv": './',
		"Max": 'max',
		"Floor": 'floor',
		"Shape": 'shape',
		"Mean": 'mean',
		"ClearMemSecret": 'clearmemsecret',
		"ClearMemPublic": 'clearmempublic'
}

class Operators(Enum):
	ADD = auto()
	SUB = auto()
	MUL = auto()
	CONV = auto()
	RELU = auto()
	Equal = auto()
	ElemWiseMul = auto()
	ElemWiseDiv = auto()
	Max = auto()
	Floor = auto()
	Shape = auto()
	Mean = auto()
	ClearMemSecret = auto()
	ClearMemPublic = auto()

	def convSymbolToEnumValue(symbolStr):
		enumStr = None
		for k,v in OperatorsSymbolDict.items():
			if v==symbolStr:
				enumStr = k
		assert(enumStr is not None)
		return Operators[enumStr]

class PaddingKeysDict:
	FH = "FH"
	FW = "FW"
	zPadHLeft = "zPadHLeft"
	zPadHRight = "zPadHRight"
	zPadWLeft = "zPadWLeft"
	zPadWRight = "zPadWRight"
	strideH = "strideH"
	strideW = "strideW"

# If this is marked true, each astNode checks the types of its inputs to confirm it satisfies the assumption
# Turn this off to get speedup in compilation
assertInputTypes = True

# Represents expression. All other nodes are specific types of expr.
class ASTNode:
	mtdKeyTFOpName = "TFOpName"
	mtdKeyTFNodeName = "TFNodeName"
	def __init__(self):
		self.gamma = {}
		self.metadata = {}
		self.decls = {}
		self.depth = 0
		self.optidict = {}

class Int(ASTNode):
	def __init__(self, value: int, bitLen=None, isSecret=True):
		if assertInputTypes: 
			assert isinstance(value, int)
			if bitLen: 
				assert isinstance(bitLen, int)
				assert ((bitLen==32) or (bitLen==64))
			assert isinstance(isSecret, bool)
		super().__init__()
		self.value = value
		self.bitLen = bitLen
		self.isSecret = isSecret

class Float(ASTNode):
	def __init__(self, value: float, isSecret=True):
		if assertInputTypes: 
			assert isinstance(value, float)
			assert isinstance(isSecret, bool)
		super().__init__()
		self.value = value
		self.isSecret = isSecret

class ID(ASTNode):
	def __init__(self, name: str):
		if assertInputTypes: 
			assert isinstance(name, str)
		super().__init__()
		self.name = name

# shape : list of int, valueList : list of int/float AST Nodes
class Decl(ASTNode):
	def __init__(self, shape: list, dataType: str, valueList: list, isSecret=True):
		if assertInputTypes:
			for elem in shape: assert isinstance(elem, int)
			if dataType:
				assert isinstance(dataType, str)
			if valueList: 
				for elem in valueList: assert isinstance(elem ,(Int,Float))
			assert(isinstance(isSecret, bool))
		super().__init__()
		self.shape = shape
		self.dataType = dataType
		self.valueList = valueList
		self.isSecret = isSecret

# expr : ASTNode
class Transp(ASTNode):
	def __init__(self, expr: ASTNode):
		if assertInputTypes:
			assert isinstance(expr, ASTNode)
		super().__init__()
		self.expr = expr

# expr : ASTNode, shape : list of int, order : int : optional
class Reshape(ASTNode):
	def __init__(self, expr: ASTNode, shape: list, order: int):
		if assertInputTypes:
			assert isinstance(expr, ASTNode)
			for elem in shape: assert isinstance(elem, int)
			assert isinstance(order, (int,type(None)))
		super().__init__()
		self.expr = expr
		self.shape = shape
		self.order = order

# expr : ASTNode
# options : Other options required by maxpool
#			Order: [FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW]
class Pool(ASTNode):
	class PoolType:
		MaxPool = "MaxPool"
		AvgPool = "AvgPool"

	def __init__(self, poolType:str, expr:ASTNode, options:dict):
		if assertInputTypes:
			assert (poolType==Pool.PoolType.MaxPool or poolType==Pool.PoolType.AvgPool)
			assert isinstance(expr, ASTNode)
			assert isinstance(options, dict)
			assert (PaddingKeysDict.FH in options)
			assert (PaddingKeysDict.FW in options)
			assert (PaddingKeysDict.zPadHLeft in options)
			assert (PaddingKeysDict.zPadHRight in options)
			assert (PaddingKeysDict.zPadWLeft in options)
			assert (PaddingKeysDict.zPadWRight in options)
			assert (PaddingKeysDict.strideH in options)
			assert (PaddingKeysDict.strideW in options)

		super().__init__()
		self.poolType = poolType
		self.expr = expr
		self.options = options

# expr:ID, index : list of int
class Index(ASTNode):
	def __init__(self, expr: ID, index: list):
		if assertInputTypes:
			assert isinstance(expr, ID)
			for elem in index: assert isinstance(elem, int)
		super().__init__()
		self.expr = expr
		self.index = index

class UOp(ASTNode):
	def __init__(self, op: Operators, expr:ASTNode):
		if assertInputTypes:
			assert isinstance(op, Operators)
			assert isinstance(expr, ASTNode)
		super().__init__()
		self.op = op
		self.expr = expr

class BOp(ASTNode):
	# Options is used to convey extra info if the operator needs so
	# For example, it will be useful for convolution to convey strides etc.
	def __init__(self, expr1: ASTNode, op: Operators, expr2: ASTNode, options=None):
		if assertInputTypes:
			assert isinstance(expr1, ASTNode)
			assert isinstance(op, Operators)
			assert isinstance(expr2, ASTNode)
			if options: assert isinstance(options, dict)
			if op == Operators.CONV:
				assert (PaddingKeysDict.FH in options)
				assert (PaddingKeysDict.FW in options)
				assert (PaddingKeysDict.zPadHLeft in options)
				assert (PaddingKeysDict.zPadHRight in options)
				assert (PaddingKeysDict.zPadWLeft in options)
				assert (PaddingKeysDict.zPadWRight in options)
				assert (PaddingKeysDict.strideH in options)
				assert (PaddingKeysDict.strideW in options)
		super().__init__()
		self.expr1 = expr1
		self.op = op
		self.expr2 = expr2
		self.options = options

class Func(ASTNode):
	def __init__(self, op: Operators, expr: ASTNode):
		if assertInputTypes:
			assert isinstance(op, Operators)
			assert isinstance(expr, ASTNode)
		super().__init__()
		self.op = op
		self.expr = expr

class Let(ASTNode):
	def __init__(self, name: ID, decl: ASTNode, expr: ASTNode):
		if assertInputTypes:
			assert isinstance(name, ID)
			assert isinstance(decl, ASTNode)
			assert isinstance(expr, ASTNode)
		super().__init__()
		self.name = name
		self.decl = decl
		self.expr = expr

# Assumption is that the output of this is always a tensor
# outputShape : list of int, funcName : str, argsList : list of ASTNodes
# isSecret : whether the output of this node is public or secret
# outputDiffInpDims = 0 => output only different input dims
#					= 1 => always output input dims
#					= 2 => never output input dims
#					: NOTE this doesn't apply for function names 
class UninterpFuncCall(ASTNode):
	def __init__(self, outputShape: list, funcName: str, argsList: list, isSecret=True, outputDiffInpDims=0):
		if assertInputTypes:
			for elem in outputShape: assert isinstance(elem, int)
			assert isinstance(funcName, str)
			for arg in argsList: assert isinstance(arg, ASTNode)
			assert isinstance(isSecret, bool)
			assert isinstance(outputDiffInpDims, int)
		super().__init__()
		self.outputShape = outputShape
		self.funcName = funcName
		self.argsList = argsList
		self.isSecret = isSecret
		self.outputDiffInpDims = outputDiffInpDims

class ArgMax(ASTNode):
	def __init__(self, outputShape: list, expr: ID, dim: ASTNode, inShape: list):
		if assertInputTypes:
			for elem in outputShape: assert isinstance(elem, int)
			assert isinstance(expr, ID)
			assert isinstance(dim, ASTNode)
			for elem in inShape: assert isinstance(elem, int)
		super().__init__()
		self.outputShape = outputShape
		self.expr = expr
		self.dim = dim
		self.inShape = inShape

class Reduce(ASTNode):
	def __init__(self, expr:ID, dim:ID, keepdims:Int, outShape:list, op: Operators):
		# keepdims is unused for now
		if assertInputTypes:
			assert isinstance(expr, ID)
			assert isinstance(dim, ID)
			assert isinstance(keepdims, Int)
			assert isinstance(outShape, list)
			for elem in outShape: assert isinstance(elem, int)
			assert isinstance(op, Operators)
		super().__init__()
		self.expr = expr
		self.dim = dim
		self.keepdims = keepdims
		self.outShape = outShape
		self.op = op

# shape : list of int, dataType : ID
# NOTE: Though datatype is being passed to this function, the output code eventually only has 
#		int in the apt bitlen for which the whole compilation is done
class Input(ASTNode):
	def __init__(self, shape:list, dataType:str, isSecret=True): 
		if assertInputTypes:
			for elem in shape: assert isinstance(elem, int)
			assert isinstance(dataType, str)
		super().__init__()
		self.shape = shape
		self.dataType = dataType
		self.isSecret = isSecret

# Since some optimizations are possible around batchnorm, keep this as an interpreted node
class FusedBatchNorm(ASTNode):
	def __init__(self, expr:ID, multExpr:ID, addExpr:ID):
		if assertInputTypes:
			assert isinstance(expr, ID)
			assert isinstance(multExpr, ID)
			assert isinstance(addExpr, ID)
		super().__init__()
		self.expr = expr
		self.multExpr = multExpr
		self.addExpr = addExpr
