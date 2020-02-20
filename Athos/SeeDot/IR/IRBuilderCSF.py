'''

Authors: Nishant Kumar.

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

import os, math
import operator
import numpy as np
from collections import OrderedDict

import Util
import IR.IR as IR
import Type as Type
import AST.AST as AST
import IR.IRUtil as IRUtil
from AST.ASTVisitor import ASTVisitor

class IRBuilderCSF(ASTVisitor):
	varNameDelim = ''
	def __init__(self):
		# For tracking temp variables
		self._var_cnt = 0
		self._iter_cnt = 0

		# Global variables
		self.decls = {} #Mapping of (identifier name (string) -> list of [type, secret/public variable, bitlen of decl])
						#	The 2nd arg can be either 'secret' or 'public'.
						#	If public/secret unspecified, default to 'secret'.
						#	The 3rd arg is used to specify the bitlen of the decl.

	def getConsSF(self):
		return Util.Config.consSF

	# Variable and iterators creation
	def getTempVars(self, n:int):
		return [self.getTempVar() for i in range(n)]

	def getTempVar(self):
		var = IR.Var('tmp' + str(self._var_cnt))
		self._var_cnt += 1
		return var

	def getTempIterators(self, n:int):
		return [self.getTempIterator() for i in range(n)]

	def getTempIterator(self):
		var = IR.Var('i' + str(self._iter_cnt))
		self._iter_cnt += 1
		return var

	# Computing exponent and intervals
	def get_expnt(self, maxabs:float): # -> int
		return self.getConsSF()

	#=================
	# Visit Functions
	#=================

	def visitInt(self, node:AST.Int, args=None):
		n = node.value
		prog = IR.Prog([IR.Comment('Int node, isSecret = {0}.'.format(node.isSecret))])
		expr = None
		if not(node.isSecret):
			if node.bitLen:
				bitLen = node.bitLen
				expr = IR.Int(n, bitLen)
			else:
				expr = IR.Int(n)
		else:
			expr = self.getTempVar()
			self.decls[expr.idf] = [node.type]
			prog = IRUtil.prog_merge(IR.Prog([IR.Decl(expr.idf, node.type)]), prog)
		return (prog, expr)

	def visitFloat(self, node:AST.Float, args=None):
		r = node.value
		p = self.get_expnt(abs(r))
		k = IR.DataType.getInt(np.ldexp(r, p))
		expr = None
		prog = IR.Prog([IR.Comment('Float to int : {0} to {1}, isSecret = {2}.'.format(str(r), str(k), node.isSecret))])
		if not(node.isSecret):
			expr = IR.Int(k)
		else:
			expr = self.getTempVar()
			self.decls[expr.idf] = [node.type]
			prog = IRUtil.prog_merge(IR.Prog([IR.Decl(expr.idf, node.type)]), prog)
		return (prog, expr)

	def visitId(self, node:AST.ID, args=None):
		idf = node.name
		prog = IR.Prog([])
		expr = IR.Var(idf)
		return (prog, expr)

	def visitDecl(self, node:AST.Decl, args=None):
		def helperAssignGen(l1, l2, allComb):
			if l2 == []:
				allComb.append(l1)
			else:
				for cur in range(l2[0]):
					helperAssignGen(l1 + [cur], l2[1:], allComb)

		prog = IR.Prog([])
		expr = self.getTempVar()
		expr.inputVar = True
		
		# If there is a valueList, then add assignment commands
		specialBitLen = -1
		if node.valueList:
			# Add assignment statements for each element of the tensor in a different array
			comment = IR.Comment(str(node.metadata))
			prog = IRUtil.prog_merge(prog, IR.Prog([comment, IR.Comment('Element assignments for ' + expr.idf)]))
			allComb = []
			helperAssignGen([], node.shape, allComb)
			for i,curComb in enumerate(allComb):
				curVal = node.valueList[i]
				finalVal = None
				if isinstance(curVal, AST.Int):
					finalVal = IR.Int(curVal.value, curVal.bitLen)
					if (specialBitLen == -1 and curVal.bitLen != Util.Config.wordLength):
						specialBitLen = curVal.bitLen
				elif isinstance(curVal, AST.Float):
					finalVal = IR.DataType.getInt(np.ldexp(curVal.value, Util.Config.consSF))
				else:
					# Assuming the elements can only be either int or floats
					assert False
				prog = IRUtil.prog_merge(prog, IR.Prog([IR.Assn(IRUtil.addIndex(expr, 
																				list(map(lambda x: IR.Int(x), curComb))), 
																finalVal)]))

		self.decls[expr.idf] = [node.type, 
								"public" if node.isSecret is False else "secret",
								Util.Config.wordLength if specialBitLen == -1 else specialBitLen
								]
		prog = IRUtil.prog_merge(IR.Prog([IR.Decl(expr.idf, 
												node.type, 
												Util.Config.wordLength if specialBitLen == -1 else specialBitLen, 
												"public" if node.isSecret is False else "secret"
												)]), prog)
		
		return (prog, expr)

	def visitTransp(self, node:AST.Transp, args=None):
		(prog_1, expr_1) = self.visit(node.expr)
		
		# decl fresh vars
		expr_2 = self.getTempVar()
		
		# cmdl_for
		typ_2 = node.type
		[I, J] = typ_2.shape
		cmd0 = IR.Comment(expr_1.idf + "^T")
		comment = IR.Comment(str(node.metadata))
		argsList = OrderedDict()
		argsList[IR.Int(I, 32)] = "I"
		argsList[IR.Int(J, 32)] = "J" 
		argsList[expr_1] = "A"
		argsList[expr_2] = "B"
		funcCall = IR.FuncCall("Transpose" + self.varNameDelim + str(len(typ_2.shape)), argsList)

		prog_for = IR.Prog([cmd0, comment, funcCall])
		prog_2 = IRUtil.prog_merge(prog_1, prog_for)
		
		self.decls[expr_2.idf] = [typ_2]
		prog = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_2.idf, typ_2)]), prog)
		return (prog_2, expr_2)

	def visitReshape(self, node:AST.Reshape, args=None):
		(prog_1, expr_1) = self.visit(node.expr)

		'''
		reshape(A, n, h, w)

		cmd1:  t1 = t2 = t3 = 0;
		loop2: for n in 0:N:
				 for h in 0:H:
				   for w in 0:W:
		cmd3:        B[n][h][w] = A[t1][t2][t3]
		cmd4:        t3++;
		cmd5:        if (t3 == WW)
					   t3 = 0;
					   t2++;
					   if (t2 == HH)
						 t2 = 0;
						 t1++;
		'''

		typ_1 = node.expr.type
		typ_2 = node.type

		# Declare variables
		expr_2 = self.getTempVar()
		iters_1 = self.getTempIterators(typ_1.dim)
		iters_2 = self.getTempIterators(typ_2.dim)

		# Initialize to 0
		cmd1 = [IR.Assn(var, IRUtil.zero) for var in iters_1]

		# Incrementing the first index
		first_iter = iters_1[0]
		cmd4 = IRUtil.incCmd(first_iter)

		# Incrementing other indices using a loop
		cmd5 = [cmd4]
		for i in range(1, typ_1.dim):
			curr_iter = iters_1[i]
			curr_size = IR.Int(typ_1.shape[i])
			cmd5 = [IRUtil.incCmd(curr_iter), IR.If(IRUtil.eq(curr_iter, curr_size), [IRUtil.initVarToZero(curr_iter)] + cmd5)]
		
		# Outer loop
		loopShape = []
		loopIters = []
		if node.order:
			for order in node.order:
				order = order - 1
				loopShape.append(typ_2.shape[order])
				loopIters.append(iters_2[order])
		else:
			loopShape = typ_2.shape
			loopIters = iters_2

		loop2 = IRUtil.loop(loopShape, loopIters, [IR.Assn(IRUtil.addIndex(expr_2, iters_2), IRUtil.addIndex(expr_1, iters_1))] + cmd5)

		# Finalize
		comment1 = IR.Comment(str(node.metadata))
		comment2 = IR.Comment("reshape(" + expr_1.idf + ", " + ', '.join(str(e) for e in typ_2.shape) + ")")
		reshape_prog = IR.Prog([comment1, comment2] + cmd1 + loop2)
		prog_2 = IRUtil.prog_merge(prog_1, reshape_prog)

		# Update context
		self.decls[expr_2.idf] = [typ_2]
		
		# Update declarations
		self.decls.update(dict((var.idf, [Type.Int(), 'public']) for var in iters_1))
		self.decls.update(dict((var.idf, [Type.Int(), 'public']) for var in iters_2))

		for var in iters_1:
			prog_2 = IRUtil.prog_merge(IR.Prog([IR.Decl(var.idf, Type.Int(), isSecret="public")]), prog_2)
		for var in iters_2:
			prog_2 = IRUtil.prog_merge(IR.Prog([IR.Decl(var.idf, Type.Int(), isSecret="public")]), prog_2)
		prog_2 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_2.idf, typ_2)]), prog_2)

		return (prog_2, expr_2)
	
	def visitPool(self, node:AST.Pool, args=None):
		(prog_1, expr_1) = self.visit(node.expr)

		[N, H, W, CI] = node.expr.type.shape
		[N1, outH, outW, CI1] = node.type.shape
		assert(N==N1 and CI==CI1)
		[expr_2] = self.getTempVars(1)

		comment = IR.Comment(str(node.metadata))
		funcCallArgsDict = OrderedDict()
		funcCallArgsDict[IR.Int(N1, 32)] = "N1"
		funcCallArgsDict[IR.Int(outH, 32)] = "outH"
		funcCallArgsDict[IR.Int(outW, 32)] = "outW"
		funcCallArgsDict[IR.Int(CI1, 32)] = "CI1"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.FH], 32)] = "FH"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.FW], 32)] = "FW"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadHLeft], 32)] = "zPadHLeft"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadHRight], 32)] = "zPadHRight"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadWLeft], 32)] = "zPadWLeft"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadWRight], 32)] = "zPadWRight"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.strideH], 32)] = "strideH"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.strideW], 32)] = "strideW"
		funcCallArgsDict[IR.Int(N, 32)] = "N"
		funcCallArgsDict[IR.Int(H, 32)] = "H"
		funcCallArgsDict[IR.Int(W, 32)] = "W"
		funcCallArgsDict[IR.Int(CI, 32)] = "CI"

		funcCallArgsDict[expr_1] = "input"
		funcCallArgsDict[expr_2] = "output"

		funcCall = IR.FuncCall(node.poolType, funcCallArgsDict)

		prog_pool = IR.Prog([comment, funcCall])
		prog_2 = IRUtil.prog_merge(prog_1, prog_pool)
		
		self.decls[expr_2.idf] = [node.type]
		prog_2 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_2.idf, node.type)]), prog_2)

		return (prog_2, expr_2)
	
	def visitIndex(self, node:AST.Index, args=None):
		(prog_1, expr_1) = self.visit(node.expr)
		prog_idx = expr_idx = []
		for curIdx in node.index:
			(prog_cur, expr_cur) = self.visit(curIdx)
			prog_idx.append(prog_cur)
			expr_idx.append(expr_cur)
		prog_3 = IRUtil.prog_merge(prog_1, [curCmd for curProg in prog_idx for curCmd in curProg])
		expr_3 = IRUtil.addIndex(expr_1, expr_idx)

		return (prog_3, expr_3)

	def visitUOp(self, node:AST.UOp, args=None):
		(prog_1, expr_1) = self.visit(node.expr)
		op = node.op
		if op == AST.Operators.ADD:
			return (prog_1, expr_1)
		assert op == AST.Operators.SUB
		
		typ_2 = node.type
		
		# e : Int
		if Type.isInt(typ_2):
			prog_2 = prog_1
			expr_2 = IRUtil.negate(expr_1)

		# e: Tensor(), or Tensor(..)
		else:
			# decl fresh vars
			expr_2 = self.getTempVar()
			iters = self.getTempIterators(typ_2.dim)

			# cmdl_assn
			expr_1_elt = IRUtil.addIndex(expr_1, iters)
			expr_2_elt = IRUtil.addIndex(expr_2, iters)
			cmdl_assn = IRUtil.loop(typ_2.shape, iters, [IR.Assn(expr_2_elt, IRUtil.negate(expr_1_elt))])
			comment = IR.Comment(str(node.metadata))
			prog_2 = IRUtil.prog_merge(prog_1, IR.Prog([comment] + cmdl_assn))

			self.decls[expr_2.idf] = [typ_2]
			prog_2 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_2.idf, node.type)]), prog_2)

		return (prog_2, expr_2)

	def visitBOp(self, node:AST.BOp, args=None):
		op = node.op
		if (op in [AST.Operators.ADD, AST.Operators.SUB, AST.Operators.Equal, AST.Operators.Max]): return self.visitBopAddOrSubLike(node)
		elif (op in [AST.Operators.ElemWiseMul, AST.Operators.ElemWiseDiv]): return self.visitBopElemWiseOp(node)
		elif op == AST.Operators.MUL: return self.visitBopMul(node)
		elif op == AST.Operators.CONV: return self.visitBopConv(node)
		elif op == AST.Operators.CONVTRANSPOSE: return self.visitBopConvTranspose(node)
		else: assert False

	def visitBopAddOrSubLike(self, node:AST.BOp, args=None):
		(prog_1, expr_1) = self.visit(node.expr1)
		(prog_2, expr_2) = self.visit(node.expr2)

		# op_ir, typ_3
		op = node.op
		if   (op == AST.Operators.ADD):
			(op_ir, op_fn) = (IR.Op.Op['+'], operator.add)
			funcName = "MatAdd"
		elif (op == AST.Operators.SUB):
			(op_ir, op_fn) = (IR.Op.Op['-'], operator.sub)
			funcName = "MatSub"
		elif (op == AST.Operators.Equal):
			(op_ir, op_fn) = (IR.Op.Op['=='], operator.eq)
			funcName = "MatEqual"
		elif (op == AST.Operators.Max):
			(op_ir, op_fn) = (IR.Op.Op['max'], None) #TODO : the operator for max is not needed right now -- add this later
			funcName = "MatMax"
		else:
			assert False

		typ_3 = node.type

		# e : Int
		if Type.isInt(typ_3):
			prog_3 = IRUtil.prog_merge(prog_1, prog_2)
			expr_3 = IR.IntBop(expr_1, op_ir, expr_2)

		# e : Tensor() -- float, or Tensor(..)
		else:
			## TODO : Hack for techfest
			if (node.type.dim != node.expr1.type.dim):
				# This needs broadcast of expr1
				assert False # For now this shouldn't occur
			if (node.type.dim != node.expr2.type.dim):
				# This needs broadcast of expr2
				funcName += 'BroadCast'

			# decl fresh vars
			expr_3 = self.getTempVar()

			cmd0 = IR.Comment(expr_1.idf + ' ' + op_ir.name + ' ' + expr_2.idf)
			outputShape = typ_3.shape
			argsDict = OrderedDict()
			for ii,curDimSize in enumerate(outputShape):
				argsDict[IR.Int(curDimSize, 32)] = "size_" + str(ii)
			argsDict[expr_1] = "A"
			argsDict[expr_2] = "B"
			argsDict[expr_3] = "C"
			funcCall = IR.FuncCall(funcName + self.varNameDelim + str(len(outputShape)), 
									argsDict
									)
			comment = IR.Comment(str(node.metadata))
			prog_3 = IRUtil.prog_merge(prog_1, prog_2, IR.Prog([comment, cmd0, funcCall]))
			self.decls[expr_3.idf] = [typ_3]
			prog_3 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_3.idf, node.type)]), prog_3)

		return (prog_3, expr_3)

	def visitBopElemWiseOp(self, node:AST.BOp, args=None):
		(prog_1, expr_1) = self.visit(node.expr1)
		(prog_2, expr_2) = self.visit(node.expr2)

		shr_n1, shr_n2 = IR.Int(0), IR.Int(0)
		if (node.op == AST.Operators.ElemWiseMul):
			op_ir = IR.Op.Op['.*']
			funcName = "ElemWiseMul"
			shr_n3 = IR.Int(Util.Config.consSF)
		elif (node.op == AST.Operators.ElemWiseDiv):
			op_ir = IR.Op.Op['./']
			funcName = "ElemWiseDiv"
			shr_n3 = IR.Int(Util.Config.consSF) # TODO : rem this, passing +ve -- ?

		typ_3 = node.type
		expr_3 = self.getTempVar()
		cmd0 = IR.Comment(expr_1.idf + ' ' + op_ir.name + ' ' + expr_2.idf)
		outputShape = typ_3.shape
		argsDict = OrderedDict()
		for ii,curDimSize in enumerate(outputShape):
			argsDict[IR.Int(curDimSize, 32)] = "size_" + str(ii)
		argsDict[expr_1] = "A"
		argsDict[expr_2] = "B"
		argsDict[expr_3] = "C"
		argsDict[shr_n3] = "shrC"
		# TODO : for consistency, add shrA and shrB ? 
		funcCall = IR.FuncCall(funcName + self.varNameDelim + str(len(outputShape)), argsDict)
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, IR.Prog([cmd0, funcCall]))
		self.decls[expr_3.idf] = [typ_3]
		prog_3 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_3.idf, node.type)]), prog_3)

		return (prog_3, expr_3)

	def visitBopMul(self, node:AST.BOp, args=None):
		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type
		# Int mul
		if (Type.isInt(typ_3)):  return self.visitBopMulInt(node)
		# Scalar mul
		elif (typ_1.dim == 0 or Type.isInt(typ_1) or typ_2.dim == 0 or Type.isInt(typ_2)): return self.visitBopMulScalar1DTensor(node)
		# Mat mul
		else: return self.visitBopMul2DTensor(node)

	def visitBopMulInt(self, node:AST.BOp, args=None):
		(prog_1, expr_1) = self.visit(node.expr1)
		(prog_2, expr_2) = self.visit(node.expr2)

		prog_3 = IRUtil.prog_merge(prog_1, prog_2)
		expr_3 = IRUtil.mul(expr_1, expr_2)

		return (prog_3, expr_3)

	def visitBopMulScalar1DTensor(self, node:AST.BOp, args=None):
		(prog_1, expr_1) = self.visit(node.expr1)
		(prog_2, expr_2) = self.visit(node.expr2)

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		isIntMult = False
		if typ_1.dim == 0 or Type.isInt(typ_1):
			a, b = expr_1, expr_2
			outputShape = typ_2.shape
			isIntMult = (Type.isInt(typ_1))
		else:
			a, b = expr_2, expr_1
			outputShape = typ_1.shape
			isIntMult = (Type.isInt(typ_2))

		# a represents the scalar and b the tensor
		shr3 = IR.Int(Util.Config.consSF)
		if isIntMult: shr3 = 0 # If multiplying with an int, then sf = 0

		# decl fresh vars
		expr_3 = self.getTempVar()
		cmd0 = IR.Comment(expr_1.idf + ' * ' + expr_2.idf)
		funcCallArgsDict = OrderedDict()
		for ii,curDimSize in enumerate(outputShape):
				funcCallArgsDict[IR.Int(curDimSize, 32)] = "size_" + str(ii)
		funcCallArgsDict[a] = "A"
		funcCallArgsDict[b] = "B"
		funcCallArgsDict[expr_3] = "C"
		funcCallArgsDict[shr3] = "shr3"

		funcCall = IR.FuncCall('ScalarMul' + self.varNameDelim + str(len(outputShape)), funcCallArgsDict)
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, IR.Prog([cmd0, funcCall]))

		self.decls[expr_3.idf] = [typ_3]
		prog_3 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_3.idf, node.type)]), prog_3)

		return (prog_3, expr_3)

	def visitBopMul2DTensor(self, node:AST.BOp, args=None):
		(prog_1, expr_1) = self.visit(node.expr1)
		(prog_2, expr_2) = self.visit(node.expr2)

		# decl fresh vars
		expr_3 = self.getTempVar()

		typ_1 = node.expr1.type
		typ_2 = node.expr2.type
		typ_3 = node.type

		[I, J] = typ_1.shape
		[J, K] = typ_2.shape
		typ_mul = Type.Tensor([J])

		shrT = Util.Config.consSF

		cmd0 = IR.Comment(expr_1.idf + ' * ' + expr_2.idf)
		funcCallArgsDict = OrderedDict()
		funcCallArgsDict[IR.Int(I, 32)] = "I"
		funcCallArgsDict[IR.Int(J, 32)] = "J"
		funcCallArgsDict[IR.Int(K, 32)] = "K"
		funcCallArgsDict[expr_1] = "A"
		funcCallArgsDict[expr_2] = "B"
		funcCallArgsDict[expr_3] = "C"
		funcCallArgsDict[IR.Int(shrT)] = "shrT"
		
		
		funcCall = IR.FuncCall("MatMulCSF2D", funcCallArgsDict)
		comment = IR.Comment(str(node.metadata))
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, IR.Prog([comment, cmd0, funcCall]))		
		self.decls[expr_3.idf] = [typ_3]
		prog_3 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_3.idf, node.type)]), prog_3)

		return (prog_3, expr_3)

	def visitBopConv(self, node:AST.BOp, args=None):
		(prog1, expr1) = self.visit(node.expr1)
		(prog2, expr2) = self.visit(node.expr2)
		
		convDim = 2
		if (AST.PaddingKeysDict.ConvDim in node.options):
			convDim = node.options[AST.PaddingKeysDict.ConvDim]

		if convDim == 2:
			[N, H, W, CI] = node.expr1.type.shape
			[FH, FW, CI, CO] = node.expr2.type.shape
		elif convDim == 3:
			[N, D, H, W, CI] = node.expr1.type.shape
			[FD, FH, FW, CI, CO] = node.expr2.type.shape
		else:
			assert(False)

		returnExpr = self.getTempVar()
		comment = IR.Comment(expr1.idf + ' # ' + expr2.idf + ', convDim = ' + str(convDim))
		funcCallArgsDict = OrderedDict()
		funcCallArgsDict[IR.Int(N, 32)] = "N"
		if convDim == 3:
			funcCallArgsDict[IR.Int(D, 32)] = "D"	
		funcCallArgsDict[IR.Int(H, 32)] = "H"
		funcCallArgsDict[IR.Int(W, 32)] = "W"
		funcCallArgsDict[IR.Int(CI, 32)] = "CI"
		if convDim == 3:
			funcCallArgsDict[IR.Int(FD, 32)] = "FD"
		funcCallArgsDict[IR.Int(FH, 32)] = "FH"
		funcCallArgsDict[IR.Int(FW, 32)] = "FW"
		funcCallArgsDict[IR.Int(CO, 32)] = "CO"
		if convDim == 3:
			funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadDLeft], 32)] = "zPadDLeft"
			funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadDRight], 32)] = "zPadDRight"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadHLeft], 32)] = "zPadHLeft"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadHRight], 32)] = "zPadHRight"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadWLeft], 32)] = "zPadWLeft"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.zPadWRight], 32)] = "zPadWRight"
		if convDim == 3:
			funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.strideD], 32)] = "strideD"	
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.strideH], 32)] = "strideH"
		funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.strideW], 32)] = "strideW"

		funcCallArgsDict[expr1] = "input"
		funcCallArgsDict[expr2] = "filter"
		funcCallArgsDict[IR.Int(Util.Config.consSF, 32)] = "consSF"
		funcCallArgsDict[returnExpr] = "output"

		if convDim == 2:
			funcCallName = "Conv2DCSF"
		else:
			funcCallName = "Conv3DCSF"
		funcCall = IR.FuncCall(funcCallName, funcCallArgsDict)

		progConv = IR.Prog([comment, funcCall])
		returnProg = IRUtil.prog_merge(prog1, prog2, progConv)
		
		self.decls[returnExpr.idf] = [node.type]
		returnProg = IRUtil.prog_merge(IR.Prog([IR.Decl(returnExpr.idf, node.type)]), returnProg)
		return (returnProg, returnExpr)

	def visitBopConvTranspose(self, node:AST.BOp, args=None):
		(prog1, expr1) = self.visit(node.expr1)
		(prog2, expr2) = self.visit(node.expr2)
		
		[N, H_prime, W_prime, CO1] = node.expr1.type.shape
		[FH, FW, CI, CO] = node.expr2.type.shape
		assert(CO1 == CO)
		
		H = node.options[AST.PaddingKeysDict.outputImgH] #outputH
		W = node.options[AST.PaddingKeysDict.outputImgW] #outputW
		pad_h_total = node.options[AST.PaddingKeysDict.zPadHLeft] + node.options[AST.PaddingKeysDict.zPadHRight]
		pad_w_total = node.options[AST.PaddingKeysDict.zPadWLeft] + node.options[AST.PaddingKeysDict.zPadWRight]
		strideH = node.options[AST.PaddingKeysDict.strideH]
		strideW = node.options[AST.PaddingKeysDict.strideW]
		[pad_h_tr_total, stride_h_tr, h_prime_tilde] = AST.Operators.findConvTransposePadding(H, H_prime, FH, pad_h_total, strideH)
		[pad_w_tr_total, stride_w_tr, w_prime_tilde] = AST.Operators.findConvTransposePadding(W, W_prime, FW, pad_w_total, strideW)

		[pad_h_tr_left, pad_h_tr_right] = AST.Operators.findLeftRightPaddingFromTotalPadding(pad_h_tr_total)
		[pad_w_tr_left, pad_w_tr_right] = AST.Operators.findLeftRightPaddingFromTotalPadding(pad_w_tr_total)

		convDim = 2
		returnExpr = self.getTempVar()
		comment = IR.Comment(expr1.idf + ' #T ' + expr2.idf + ', convDim = ' + str(convDim))
		funcCallArgsDict = OrderedDict()
		funcCallArgsDict[IR.Int(N, 32)] = "N"
		funcCallArgsDict[IR.Int(H_prime, 32)] = "H_prime"
		funcCallArgsDict[IR.Int(W_prime, 32)] = "W_prime"
		funcCallArgsDict[IR.Int(CO, 32)] = "CO"
		funcCallArgsDict[IR.Int(FH, 32)] = "FH"
		funcCallArgsDict[IR.Int(FW, 32)] = "FW"
		funcCallArgsDict[IR.Int(CI, 32)] = "CI"
		funcCallArgsDict[IR.Int(pad_h_tr_left, 32)] = "pad_h_tr_left"
		funcCallArgsDict[IR.Int(pad_h_tr_right, 32)] = "pad_h_tr_right"
		funcCallArgsDict[IR.Int(pad_w_tr_left, 32)] = "pad_w_tr_left"
		funcCallArgsDict[IR.Int(pad_w_tr_right, 32)] = "pad_w_tr_right"
		funcCallArgsDict[IR.Int(stride_h_tr, 32)] = "stride_h_tr"
		funcCallArgsDict[IR.Int(stride_w_tr, 32)] = "stride_w_tr"		

		funcCallArgsDict[expr1] = "input"
		funcCallArgsDict[expr2] = "filter"
		funcCallArgsDict[IR.Int(Util.Config.consSF, 32)] = "consSF"
		funcCallArgsDict[returnExpr] = "output"

		if convDim == 2:
			funcCallName = "ConvTranspose2DCSF"
		else:
			funcCallName = "ConvTranspose3DCSF"
		funcCall = IR.FuncCall(funcCallName, funcCallArgsDict)

		progConv = IR.Prog([comment, funcCall])
		returnProg = IRUtil.prog_merge(prog1, prog2, progConv)
		
		self.decls[returnExpr.idf] = [node.type]
		returnProg = IRUtil.prog_merge(IR.Prog([IR.Decl(returnExpr.idf, node.type)]), returnProg)
		return (returnProg, returnExpr)

	def visitFunc(self, node:AST.Func, args=None):
		op = node.op
		
		if  op in [AST.Operators.Floor, AST.Operators.Shape, AST.Operators.RELU, AST.Operators.ClearMemSecret, AST.Operators.ClearMemPublic]:
			return self.visitFloorLike(node)
		else: assert False

	def visitFloorLike(self, node:AST.Func, args=None):
		(prog1, expr1) = self.visit(node.expr)
		tmpExpr = self.getTempVar()

		if node.op == AST.Operators.Floor:
			funcName = "Floor"
		elif node.op == AST.Operators.Shape:
			funcName = "Shape"
		elif node.op == AST.Operators.RELU:
			funcName = "Relu"
		elif node.op == AST.Operators.ClearMemSecret:
			funcName = "ClearMemSecret"
		elif node.op == AST.Operators.ClearMemPublic:
			funcName = "ClearMemPublic"
		else:
			assert False

		argsList = OrderedDict()
		
		inputType = node.expr.type
		if Type.isTensor(inputType):
			for ii, curDim in enumerate(inputType.shape):
				argsList[IR.Int(curDim, 32)] = "inShape_" + str(ii)
		argsList[expr1] = "inArr"

		if Type.isTensor(node.type):
			argsList[tmpExpr] = "outArr"
			self.decls[tmpExpr.idf] = [node.type]

		if node.op == AST.Operators.Floor:
			argsList[IR.Int(Util.Config.consSF)] = "curScale"
		comment = IR.Comment(str(node.metadata))

		funcNameSuffix = ""
		if Type.isTensor(inputType):
			funcNameSuffix = str(len(inputType.shape))

		progFinal = IRUtil.prog_merge(prog1 , IR.Prog([comment, IR.FuncCall(funcName + self.varNameDelim + funcNameSuffix, argsList)]))
		if Type.isTensor(node.type):
			progFinal = IRUtil.prog_merge(IR.Prog([IR.Decl(tmpExpr.idf, node.type)]), progFinal)

		return (progFinal, tmpExpr)

	def visitLet(self, node:AST.Let, args=None):
		(prog_1, expr_1) = self.visit(node.decl)
		typ_1 = node.decl.type
		idf = node.name.name
		(prog_2, expr_2) = self.visit(node.expr)
		prog_2 = prog_2.subst(idf, expr_1)
		expr_2 = expr_2.subst(idf, expr_1)
		prog_3 = IRUtil.prog_merge(prog_1, prog_2)
		return (prog_3, expr_2)

	def visitUninterpFuncCall(self, node:AST.UninterpFuncCall, args=None):
		progList = []
		exprList = []
		for ii, curArg in enumerate(node.argsList):
			(progN, exprN) = self.visit(curArg)
			progList.append(progN)
			exprList.append(exprN)
		
		returnExpr = self.getTempVar()

		funcName = node.funcName
		funcName += self.varNameDelim + str(len(node.outputShape))
		for ii, curArg in enumerate(node.argsList):
			if Type.isTensor(curArg.type):
				curShape = curArg.type.shape

				# If len(shape) == 0 : that means its a float - no need to qualify
				#	the function name with 0 in that case, since its essentially
				#	become an int.
				if (len(curShape) > 0):
					funcName += self.varNameDelim + str(len(curShape))
			### TODO : WRONG -- TEMP FIX -- right now if random strings like int are passed, its being set as datatype int -- int datatype in
			#		   unintrepreted func call is being used in a hacky way right now -- fix this later

		# Policy : 
		#	First output tensor sizes are inserted in args.
		#	Then for each input tensor, its shape is inserted in args, followed by the input tensor itself.
		#	If the current input tensor has the same shape as any of the previous tensors, then its shape is not inserted.
		funcArgsList = OrderedDict()
		tensorShapesFound = {}
		outputShape = node.type.shape
		for ii, curDim in enumerate(outputShape):
			funcArgsList[IR.Int(curDim, 32)] = "OutputShape_" + str(ii)
		tensorShapesFound[tuple(outputShape)] = True
		for ii in range(0, len(node.argsList)):
			if node.outputDiffInpDims < 2 and Type.isTensor(node.argsList[ii].type):
				curInpShape = node.argsList[ii].type.shape
				if ((node.outputDiffInpDims == 1) or (node.outputDiffInpDims == 0 and tuple(curInpShape) not in tensorShapesFound)):
					for jj, curDim in enumerate(curInpShape):
						funcArgsList[IR.Int(curDim, 32)] = "Input_" + str(ii) + self.varNameDelim + str(jj)
					tensorShapesFound[tuple(curInpShape)] = True
			funcArgsList[exprList[ii]] = "inpExpr_" + str(ii)
		funcArgsList[returnExpr] = "output"
		
		comment = IR.Comment(str(node.metadata))
		progFinal = progList[0]
		if len(progList) > 1:
			for ii in range(1, len(progList)):
				progFinal = IRUtil.prog_merge(progFinal, progList[ii])
		progFinal = IRUtil.prog_merge(progFinal, IR.Prog([comment, IR.FuncCall(funcName, funcArgsList)]))

		self.decls[returnExpr.idf] = [node.type, "public" if node.isSecret is False else "secret"]
		progFinal = IRUtil.prog_merge(IR.Prog([IR.Decl(returnExpr.idf, 
														node.type, 
														isSecret="public" if node.isSecret is False else "secret")]), 
										progFinal)
		return (progFinal, returnExpr)

	def visitArgMax(self, node:AST.ArgMax, args=None):
		(prog_1, expr1) = self.visit(node.expr)
		(prog_2, expr2) = self.visit(node.dim)
		
		tmpExpr = self.getTempVar()
		outputShape = node.type.shape

		funcArgsList = OrderedDict()
		outputShape = node.type.shape
		for ii, curDim in enumerate(outputShape):
			funcArgsList[IR.Int(curDim, 32)] = "OutputShape_" + str(ii)
		for ii, curDim in enumerate(node.inShape):
			funcArgsList[IR.Int(curDim, 32)] = "OutputShape_" + str(ii)
		funcArgsList[expr1] = "inArr"
		funcArgsList[expr2] = "dim"
		funcArgsList[tmpExpr] = "outArr"

		funcCall = IR.FuncCall("ArgMax" + self.varNameDelim + str(len(outputShape)), funcArgsList)
		comment = IR.Comment(str(node.metadata))
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, IR.Prog([comment, funcCall]))
		self.decls[tmpExpr.idf] = [node.type]
		prog_3 = IRUtil.prog_merge(IR.Prog([IR.Decl(tmpExpr.idf, node.type)]), prog_3)
		return (prog_3, tmpExpr)

	def visitInput(self, node:AST.Input, args=None):
		returnExpr = self.getTempVar()
		returnExpr.inputVar = True
		comment = IR.Comment(str(node.metadata))
		return (IR.Prog([comment, IR.Input(returnExpr, node.shape, node.dataType, node.isSecret, node.inputByParty)]), returnExpr)

	def visitReduce(self, node:AST.Reduce, args=None):
		(prog_1, expr1) = self.visit(node.expr)
		(prog_2, expr2) = self.visit(node.dim)

		returnExpr = self.getTempVar()

		assert(node.op in [AST.Operators.ADD, AST.Operators.MUL, AST.Operators.Mean])
		scalingFac = None
		if (node.op == AST.Operators.ADD):
			funcName = "ReduceSum"
		elif (node.op == AST.Operators.MUL):
			funcName = "ReduceMul"
			scalingFac = Util.Config.consSF
		elif (node.op == AST.Operators.Mean):
			funcName = "ReduceMean"
		else:
			print("Unknown node.op in AST.Reduce.", file=sys.stderr)
			assert False

		funcArgsList = OrderedDict()
		outputShape = node.type.shape
		for ii, curDim in enumerate(outputShape):
			funcArgsList[IR.Int(curDim, 32)] = "OutputShape_" + str(ii)

		inputShape = node.expr.type.shape
		for ii, curDim in enumerate(inputShape):
			funcArgsList[IR.Int(curDim, 32)] = "InputShape_" + str(ii)

		funcArgsList[expr1] = "inputArr"
		funcArgsList[expr2] = "dimension"
		funcArgsList[returnExpr] = "outArr"
		if scalingFac:
			funcArgsList[IR.Int(scalingFac)] = "ScalingFactor"

		funcCall = IR.FuncCall(funcName + self.varNameDelim + str(len(outputShape)) + self.varNameDelim + str(len(inputShape)), funcArgsList)
		comment = IR.Comment(str(node.metadata))
		prog_3 = IRUtil.prog_merge(prog_1, prog_2, IR.Prog([comment, funcCall]))

		self.decls[returnExpr.idf] = [node.type]
		prog_3 = IRUtil.prog_merge(IR.Prog([IR.Decl(returnExpr.idf, node.type)]), prog_3)
		return (prog_3, returnExpr)

	def visitFusedBatchNorm(self, node:AST.FusedBatchNorm, args=None):
		(prog1, expr1) = self.visit(node.expr)
		(prog2, expr2) = self.visit(node.multExpr)
		(prog3, expr3) = self.visit(node.addExpr)

		returnExpr = self.getTempVar()

		funcArgsList = OrderedDict()
		for ii, elem in enumerate(node.type.shape):
			funcArgsList[IR.Int(elem, 32)] = "elem_"+str(ii)
		funcArgsList[expr1] = "expr"
		funcArgsList[expr2] = "multExpr"
		funcArgsList[expr3] = "addExpr"
		funcArgsList[IR.Int(Util.Config.consSF, 32)] = "consSF"
		funcArgsList[returnExpr] = "returnExpr"
		funcCallIR = IR.FuncCall("FusedBatchNorm" + self.varNameDelim 
							  + str(len(node.type.shape)) + self.varNameDelim #one for output
							  + str(len(node.type.shape)) + self.varNameDelim #one for input
							  + str(len(node.multExpr.type.shape)) + self.varNameDelim
							  + str(len(node.addExpr.type.shape)), 
							  funcArgsList)

		comment = IR.Comment(str(node.metadata))
		returnProg = IRUtil.prog_merge(prog1, prog2, prog3, IR.Prog([comment, funcCallIR]))

		self.decls[returnExpr.idf] = [node.type]
		returnProg = IRUtil.prog_merge(IR.Prog([IR.Decl(returnExpr.idf, node.type)]), returnProg)
		return (returnProg, returnExpr)
