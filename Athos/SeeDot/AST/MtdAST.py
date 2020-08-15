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

class MtdAST(ASTVisitor):
	def visitInt(self, node:AST.Int, mtd:dict):
		node.metadata.update(mtd)

	def visitFloat(self, node:AST.Float, mtd:dict):
		node.metadata.update(mtd)

	def visitId(self, node:AST.ID, mtd:dict):
		node.metadata.update(mtd)

	def visitDecl(self, node:AST.Decl, mtd:dict):
		node.metadata.update(mtd)

	def visitTranspose(self, node:AST.Transpose, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitReshape(self, node:AST.Reshape, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitPool(self, node:AST.Pool, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)
	
	def visitUOp(self, node:AST.UOp, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitBOp(self, node:AST.BOp, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr1, mtd)
		self.visit(node.expr2, mtd)

	def visitFunc(self, node:AST.Func, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitLet(self, node:AST.Let, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.name, mtd)
		self.visit(node.decl, mtd)
		self.visit(node.expr, mtd)

	def visitUninterpFuncCall(self, node:AST.UninterpFuncCall, mtd:dict):
		node.metadata.update(mtd)
		for curArg in node.argsList:
			self.visit(curArg, mtd)

	def visitArgMax(self, node:AST.ArgMax, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitReduce(self, node:AST.Reduce, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)
		self.visit(node.dim, mtd)

	def visitInput(self, node:AST.Input, mtd:dict):
		node.metadata.update(mtd)

	def visitFusedBatchNorm(self, node:AST.FusedBatchNorm, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)
		self.visit(node.multExpr, mtd)
		self.visit(node.addExpr, mtd)
