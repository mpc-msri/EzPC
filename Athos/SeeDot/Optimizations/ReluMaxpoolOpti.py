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

class ReluMaxpoolOpti(ASTVisitor):
	def visitLet(self, node:AST.Let, args:dict):
		if isinstance(node.decl, AST.Func) and node.decl.op==AST.Operators.RELU:
			# Relu declaration entered
			if isinstance(node.expr, AST.Let):
				# There is a next let statement
				if isinstance(node.expr.decl, AST.Pool) and (node.expr.decl.poolType==AST.Pool.PoolType.MaxPool):
					# This is the case of relu followed by maxpool declaration
					# Switch here
					print("Found relu followed by maxpool. Performing optimization.")
					# Assuming here that only maxpool's output is subsequently used. 
					# TODO: Do something for above?
					reluDecl, maxpoolDecl = node.decl, node.expr.decl
					maxpoolDecl.expr = reluDecl.expr
					reluDecl.expr = node.name
					node.decl = maxpoolDecl
					node.expr.decl = reluDecl
		self.visit(node.name)
		self.visit(node.decl)
		self.visit(node.expr)

