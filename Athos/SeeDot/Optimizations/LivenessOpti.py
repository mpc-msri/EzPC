'''

Authors: Nishant Kumar.

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

import AST.AST as AST
from AST.ASTVisitor import ASTVisitor
from AST.MtdAST import MtdAST

#In the below analysis, each node saves what all unbound variables 
#	are used in its sub-tree. If the set is empty, nothing is saved.
#	A subsequent pass then finds the variables
#	wnich can be cleared.
class LivenessAnalysis(ASTVisitor):
	optidictKey = "LivenessAnalysis" #This key will be used to store in optidict of the ASTNode 
									 #	list of all variables which are unbound in that sub-tree.
	def visitInt(self, node:AST.Int, args):
		return []

	def visitFloat(self, node:AST.Float, args):
		return []

	def visitId(self, node:AST.ID, args):
		unboundVars = [node.name]
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitDecl(self, node:AST.Decl, args):
		return []

	def visitTranspose(self, node:AST.Transpose, args):
		unboundVars = self.visit(node.expr, args)
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitReshape(self, node:AST.Reshape, args):
		unboundVars = self.visit(node.expr, args)
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars
	
	def visitPool(self, node:AST.Pool, args):
		unboundVars = self.visit(node.expr, args)
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitUOp(self, node:AST.UOp, args):
		unboundVars = self.visit(node.expr, args)
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitBOp(self, node:AST.BOp, args):
		unboundVars = list(set(self.visit(node.expr1, args) + self.visit(node.expr2, args)))
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitFunc(self, node:AST.Func, args):
		unboundVars = self.visit(node.expr, args)
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitLet(self, node:AST.Let, args):
		declVars = self.visit(node.decl, args)
		exprVars = self.visit(node.expr, args)
		unboundVars = list((set(declVars)|set(exprVars))-set([node.name.name]))
		if isinstance(node.decl, AST.ID):
			#This is of the type let J1 = J2 in J1. 
			#	Since J1 and J2 refer to the same variable, J2 should remain bounded.
			unboundVars = list(set(unboundVars) - set([node.decl.name]))
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars		

	def visitUninterpFuncCall(self, node:AST.UninterpFuncCall, args):
		unboundVarsSet = set([])
		for elem in node.argsList:
			unboundVarsSet |= set(self.visit(elem, args))
		unboundVars = list(unboundVarsSet)
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitArgMax(self, node:AST.ArgMax, args):
		unboundVars = list(set(self.visit(node.expr, args) + self.visit(node.dim, args)))
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitReduce(self, node:AST.Reduce, args):
		unboundVars = list(set(self.visit(node.expr, args) + self.visit(node.dim, args) + self.visit(node.keepdims, args)))
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

	def visitInput(self, node:AST.Input, args):
		return []

	def visitFusedBatchNorm(self, node:AST.FusedBatchNorm, args):
		unboundVars = list(set(self.visit(node.expr, args) + self.visit(node.multExpr, args) + self.visit(node.addExpr, args)))
		node.optidict[self.optidictKey] = unboundVars
		return unboundVars

class LivenessOpti(ASTVisitor):
	def visitLet(self, node:AST.Let, args):
		assert(isinstance(args, list))
		assert(isinstance(args[0], MtdAST))
		assert(isinstance(args[1], int))
		assert(isinstance(args[2], dict)) #dict {variable name string -> isSecretVariable bool}
		curUnboundVars = []
		exprUnboundVars = []
		if LivenessAnalysis.optidictKey in node.optidict:
			curUnboundVars = node.optidict[LivenessAnalysis.optidictKey]
		if LivenessAnalysis.optidictKey in node.expr.optidict:
			exprUnboundVars = node.expr.optidict[LivenessAnalysis.optidictKey]
		varsToDeAllocate = list(set(curUnboundVars)-set(exprUnboundVars))
		origNodeExpr = node.expr
		astSubTree = node.expr
		mtdForNewASTNodes = {AST.ASTNode.mtdKeyTFOpName : "No-op: ClearMem",
							 AST.ASTNode.mtdKeyTFNodeName : ""}
		for ii, curVarName in enumerate(varsToDeAllocate):
			assert(curVarName in args[2])
			newSubTree = AST.Let(AST.ID("cv"+str(args[1]+ii)), 
								AST.Func(AST.Operators.ClearMemSecret if args[2][curVarName] else AST.Operators.ClearMemPublic, 
										 AST.ID(curVarName)), 
								AST.ID(""))
			args[0].visit(newSubTree, mtdForNewASTNodes)
			newSubTree.expr = astSubTree
			node.expr = newSubTree
			astSubTree = node.expr
		self.visit(node.name, [args[0], args[1]+len(varsToDeAllocate), args[2]])
		self.visit(node.decl, [args[0], args[1]+len(varsToDeAllocate), args[2]])
		isCurrentLetDeclarationSecret = True
		if hasattr(node.decl, 'isSecret'):
			isCurrentLetDeclarationSecret = node.decl.isSecret
			assert(type(isCurrentLetDeclarationSecret)==bool)
		self.visit(origNodeExpr, [args[0], args[1]+len(varsToDeAllocate), {**args[2], **{node.name.name: isCurrentLetDeclarationSecret}}])
