'''

Authors: Sridhar Gopinath, Nishant Kumar.

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

import Util
import operator
from functools import reduce
import AST.AST as AST
from AST.ASTVisitor import ASTVisitor

class Type:
	pass

class Int(Type):
	def __init__(self, bitlen=-1, isSecret=False):
		if bitlen==-1:
			self.bitlen = Util.Config.wordLength
		else:
			self.bitlen = bitlen
		self.isSecret = isSecret

class Unit(Type):
	pass

class Tensor(Type):
	def __init__(self, shape:list, bitlen=-1, isSecret=True):
		self.shape = shape
		self.dim = len(shape)
		if bitlen==-1:
			self.bitlen = Util.Config.wordLength
		else:
			self.bitlen = bitlen
		self.isSecret = isSecret

	def size(self):
		return reduce(operator.mul, self.shape, 1)

	# Tensor without any dimension (float) or a tensor with all dimensions equal to 1
	def isShapeOne(self):
		return self.dim == 0 or self.size() == 1

def isInt(type:Type):
	return isinstance(type, Int)

def isTensor(type:Type):
	return isinstance(type, Tensor)

def isUnit(type:Type):
	return isinstance(type, Unit)

def isEqual(type1:Type, type2:Type):
	if isInt(type1) and isInt(type2):
		return True
	elif isTensor(type1) and isTensor(type2):
		if type1.dim != type2.dim:
			return False
		return type1.shape == type2.shape
	else:
		assert False

class InferType(ASTVisitor):
	def visitInt(self, node:AST.Int, args=None):
		bitlen = Util.Config.wordLength
		if node.bitLen:
			bitlen = node.bitLen
		node.type = Int(bitlen, node.isSecret)
		return node.type

	def visitFloat(self, node:AST.Float, args=None):
		# Float is represented as a tensor with 0 dimension
		node.type = Tensor([], -1, node.isSecret)
		return node.type

	def visitId(self, node:AST.ID, args=None):
		if node.name not in node.gamma:
			print("Error in type checking: Found id which is not contained in gamma.", file=sys.stderr)
			assert(False)
		else:
			node.type = node.gamma[node.name]
		return node.type

	def visitDecl(self, node:AST.Decl, args=None):
		#TODO -- fill in bitlen properly
		node.type = Tensor(node.shape, -1, node.isSecret)
		return node.type

	def visitTranspose(self, node:AST.Transpose, args=None):
		node.expr.gamma = dict(node.gamma)
		exprType = self.visit(node.expr)

		assert isTensor(exprType)

		perm = node.perm
		shape = exprType.shape
		if (perm is None):
			perm = [i for i in reversed(range(len(shape)))]
		new_shape = []
		for i in perm:
			new_shape.append(shape[i])
		node.type = Tensor(new_shape)
		return node.type

	def visitReshape(self, node:AST.Reshape, args=None):
		node.expr.gamma = dict(node.gamma)
		exprType = self.visit(node.expr)
		
		assert isTensor(exprType) and exprType.dim >= 1
		
		# Reshape is valid if the total number of elements remain same after reshape
		assert reduce(operator.mul, exprType.shape, 1) == reduce(operator.mul, node.shape, 1)
		node.type = Tensor(node.shape)

		return node.type

	def visitPool(self, node:AST.Pool, args=None):
		node.expr.gamma = dict(node.gamma)
		exprType = self.visit(node.expr)

		# Implementation only performs maxpool over a 4D input
		assert isTensor(exprType) and exprType.dim == 4
		[N, H, W, CI] = exprType.shape
		FH = node.options[AST.PaddingKeysDict.FH]
		FW = node.options[AST.PaddingKeysDict.FW]
		zPadHLeft = node.options[AST.PaddingKeysDict.zPadHLeft]
		zPadHRight = node.options[AST.PaddingKeysDict.zPadHRight]
		zPadWLeft = node.options[AST.PaddingKeysDict.zPadWLeft]
		zPadWRight = node.options[AST.PaddingKeysDict.zPadWRight]
		strideH = node.options[AST.PaddingKeysDict.strideH]
		strideW = node.options[AST.PaddingKeysDict.strideW]

		newH = ((H + zPadHLeft + zPadHRight - FH)//strideH) + 1
		newW = ((W + zPadWLeft + zPadWRight - FW)//strideW) + 1

		node.type = Tensor([N, newH, newW, CI])

		return node.type

	def visitUOp(self, node:AST.UOp, args=None):
		node.expr.gamma = dict(node.gamma)
		node.type = self.visit(node.expr)
		return node.type

	def visitBOp(self, node:AST.BOp, args=None):
		node.expr1.gamma = dict(node.gamma)
		eType = self.visit(node.expr1)

		node.expr2.gamma = dict(node.gamma)
		fType = self.visit(node.expr2)

		if node.op in [AST.Operators.ADD, AST.Operators.ElemWiseMul, AST.Operators.ElemWiseDiv]:
			# Ops supporting broadcasting
			return self.typeCheckBroadcastOps(node, eType, fType)
		elif node.op in [AST.Operators.SUB, AST.Operators.Equal]:
			return self.visitBopAddLike(node, eType, fType)
		elif node.op == AST.Operators.MUL:
			return self.visitBopMul(node, eType, fType)
		elif node.op == AST.Operators.CONV:
			return self.visitBopConv(node, eType, fType)
		elif node.op == AST.Operators.CONVTRANSPOSE:
			return self.visitBopConvTranspose(node, eType, fType)
		else:
			assert False

	def typeCheckBroadcastOps(self, node:AST.BOp, eType:Type, fType:Type):
		# Ops which support broadcasting have different type checking
		# If adding a new op here which supports broadcasting, then be careful!
		# Currently, its assumed the op is commutative. If that is not true, following will be wrong ! 

		assert node.op in [AST.Operators.ADD, AST.Operators.ElemWiseMul, AST.Operators.ElemWiseDiv]
		if (len(eType.shape) < len(fType.shape)):
			# swap expr1 and expr2 -- this is valid for commutative ops
			# be careful for ops which are not commutative
			temp = node.expr1
			node.expr1 = node.expr2
			node.expr2 = temp

			temp = eType
			eType = fType
			fType = temp
		
		# Now true that dim(eType) >= dim(fTYpe)
		assert len(eType.shape) >= len(fType.shape)

		if isInt(eType) and isInt(fType):
			node.type = Int(eType.bitlen, eType.isSecret)
		elif isTensor(eType) and isTensor(fType):
			revETypeShape = eType.shape[::-1]
			revFTypeShape = fType.shape[::-1]
			for i, fTypeCurDim in enumerate(revFTypeShape):
				eTypeCurDim = revETypeShape[i]
				if not(eTypeCurDim==1 or fTypeCurDim==1 or eTypeCurDim==fTypeCurDim):
					# broadcast not possible - raise error
					print("Broadcast not possible for current node.", eType.shape, fType.shape)
					assert False

			# Broadcast possible
			node.type = eType
		else:
			print(eType, fType)
			assert False
		return node.type

	def visitBopMul(self, node:AST.BOp, eType:Type, fType:Type, args=None):
		if isInt(eType) and isInt(fType):
			node.type = Int(eType.bitlen, eType.isSecret)
		elif isTensor(eType) and isTensor(fType):
			if eType.dim == 0:
				node.type = fType
			elif fType.dim == 0:
				node.type = eType
			else:
				assert eType.dim == 2 and fType.dim == 2
				[n1, n2] = eType.shape
				[n3, n4] = fType.shape
				assert n2 == n3
				node.type = Tensor([n1, n4])
		else:
			print("Error: Unknown condition in type checking.", file=sys.stderr)
			assert(False)

		return node.type

	def visitBopConv(self, node:AST.BOp, eType:Type, fType:Type, args=None):
		assert isTensor(eType) and isTensor(fType)
		convDim = 2
		group = 1
		if AST.PaddingKeysDict.ConvDim in node.options:
			convDim = node.options[AST.PaddingKeysDict.ConvDim]

		if convDim==2:
			assert eType.dim == 4 and fType.dim == 4
		elif convDim==3:
			assert eType.dim == 5 and fType.dim == 5
		else:
			assert(False)
		
		N = D = H = W = CI = FD = FH = FW = CI1 = CO = -1
		newD = -1
		if (convDim == 2):
			[N, H, W, CI] = eType.shape
			[FH, FW, CI1, CO] = fType.shape
		elif (convDim == 3):
			[N, D, H, W, CI] = eType.shape
			[FD, FH, FW, CI1, CO] = fType.shape
			assert(FD == node.options[AST.PaddingKeysDict.FD])
			zPadDLeft = node.options[AST.PaddingKeysDict.zPadDLeft]
			zPadDRight = node.options[AST.PaddingKeysDict.zPadDRight]
			strideD = node.options[AST.PaddingKeysDict.strideD]

			newD = ((D + zPadDLeft + zPadDRight - FD)//strideD) + 1
		else:
			assert(False)

		if AST.PaddingKeysDict.group in node.options:	
				group = node.options[AST.PaddingKeysDict.group]

		assert(FH == node.options[AST.PaddingKeysDict.FH])
		assert(FW == node.options[AST.PaddingKeysDict.FW])
		assert(CI1*group == CI)
		zPadHLeft = node.options[AST.PaddingKeysDict.zPadHLeft]
		zPadHRight = node.options[AST.PaddingKeysDict.zPadHRight]
		zPadWLeft = node.options[AST.PaddingKeysDict.zPadWLeft]
		zPadWRight = node.options[AST.PaddingKeysDict.zPadWRight]
		strideH = node.options[AST.PaddingKeysDict.strideH]
		strideW = node.options[AST.PaddingKeysDict.strideW]

		newH = ((H + zPadHLeft + zPadHRight - FH)//strideH) + 1
		newW = ((W + zPadWLeft + zPadWRight - FW)//strideW) + 1

		if convDim == 2:
			shape = [N, newH, newW, CO]
		elif convDim == 3:
			shape = [N, newD, newH, newW, CO]
		node.type = Tensor(shape)
		return node.type

	def visitBopConvTranspose(self, node:AST.BOp, eType:Type, fType:Type, args=None):
		assert isTensor(eType) and isTensor(fType)
		
		convDim = 2
		if AST.PaddingKeysDict.ConvDim in node.options:
			convDim = node.options[AST.PaddingKeysDict.ConvDim]

		if convDim==2:
			[N, HP, WP, CI1] = eType.shape
			[FH, FW, CO, CI] = fType.shape
		elif convDim==3:
			[N, DP, HP, WP, CI1] = eType.shape
			[FD, FH, FW, CO, CI] = fType.shape
		else:
			assert(False)
		assert(CI1 == CI)
		if convDim==3:
			outputImgD = node.options[AST.PaddingKeysDict.outputImgD]
		outputImgH = node.options[AST.PaddingKeysDict.outputImgH]
		outputImgW = node.options[AST.PaddingKeysDict.outputImgW]

		if convDim==2:
			shape = [N, outputImgH, outputImgW, CO]
		else:
			shape = [N, outputImgD, outputImgH, outputImgW, CO]

		# Logic explanation:
		#	ConvTranpose can be thought of as the inverse of some convolution for which it is doing the upsampling.
		#	For calculation of padding in the convTranspose operation, the output image size is required.
		#	This is why TF also mandates the operator to be specified with output size.
		#	This conv transpose operation can be thought of as conv between output
		#		of size shape = [N, outputImgH, outputImgW, CI], and filter of size [FH, FW, CI, CO].
		#		Hence, the input for this convTranspose would be [N, HP, WP, CO]

		node.type = Tensor(shape)
		return node.type

	def visitBopAddLike(self, node:AST.BOp, eType: Type, fType: Type, args=None):
		if isInt(eType) and isInt(fType):
			pass
		elif isTensor(eType) and isTensor(fType):
			assert eType.shape == fType.shape
		else:
			assert False
		
		node.type = eType
		return node.type

	def visitFunc(self, node:AST.Func, args=None):
		node.expr.gamma = dict(node.gamma)
		eType = self.visit(node.expr)

		if node.op == AST.Operators.RELU:
			assert isTensor(eType) and eType.dim >= 1
			node.type = eType
		elif node.op == AST.Operators.Floor:
			node.type = eType
		elif node.op == AST.Operators.Shape:
			assert isTensor(eType)
			node.type = Tensor([len(eType.shape)])
		elif node.op  == AST.Operators.ClearMemSecret:
			node.type = Unit()
		elif node.op  == AST.Operators.ClearMemPublic:
			node.type = Unit()
		else:
			print(node.op)
			assert False

		return node.type

	def visitLet(self, node:AST.Let, args=None):
		node.decl.gamma = dict(node.gamma)
		eType = self.visit(node.decl)

		node.expr.gamma = dict(node.gamma)
		node.expr.gamma[node.name.name] = eType
		fType = self.visit(node.expr)

		node.type = fType
		return node.type

	def visitUninterpFuncCall(self, node:AST.UninterpFuncCall, args=None):
		# Assert that outputShape and inputDims are lists of int astNode.
		assert(len(node.argsList) > 0)
		for curArg in node.argsList:
			curArg.gamma = dict(node.gamma)
			self.visit(curArg) #This should set the type of each of the input nodes
		outputShape = node.outputShape
		node.type = Tensor(outputShape)
		return node.type

	def visitArgMax(self, node:AST.ArgMax, args=None):
		node.expr.gamma = dict(node.gamma)
		inpTensorType = self.visit(node.expr)
		
		node.dim.gamma = dict(node.gamma)
		dimType = self.visit(node.dim)
		assert(isInt(dimType) or (isTensor(dimType) and (len(dimType.shape)==0)))

		node.type = Tensor(node.outputShape)
		return node.type
	
	def visitReduce(self, node:AST.Reduce, args=None):
		cur_gamma = dict(node.gamma)
		node.expr.gamma = cur_gamma
		node.dim.gamma = cur_gamma
		node.keepdims.gamma = cur_gamma

		self.visit(node.expr)
		self.visit(node.dim)
		self.visit(node.keepdims)

		node.type = Tensor(node.outShape)
		return node.type

	def visitInput(self, node:AST.Input, args=None):
		node.type = Tensor(node.shape)
		return node.type

	def visitFusedBatchNorm(self, node:AST.FusedBatchNorm, args=None):
		cur_gamma = dict(node.gamma)
		node.expr.gamma = cur_gamma
		node.multExpr.gamma = cur_gamma
		node.addExpr.gamma = cur_gamma
		
		exprType = self.visit(node.expr)
		multExprType = self.visit(node.multExpr)
		addExprType = self.visit(node.addExpr)

		assert(len(multExprType.shape)==1)
		assert(len(addExprType.shape)==1)

		[C1] = multExprType.shape
		[C2] = addExprType.shape

		assert(exprType.shape[-1]==C1 and C1==C2)

		node.type = exprType
		return node.type

