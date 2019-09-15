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

