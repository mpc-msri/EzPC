import AST.AST as AST
from AST.ASTVisitor import ASTVisitor
import binascii

indent = ""

class PrintAST(ASTVisitor):
	#TODO : fix printing of AST
	def visitInt(self, node:AST.Int, args=None):
		print(indent * node.depth, node.value, end=' ')

	def visitFloat(self, node:AST.Float, args=None):
		print(indent * node.depth, node.value, end=' ')

	def visitId(self, node:AST.ID, args=None):
		print(indent * node.depth, node.name, end=' ')

	def visitDecl(self, node:AST.Decl, args=None):
		if node.valueList:
			print(indent * node.depth, node.shape, list(map(lambda x: x.value, node.valueList)), end=' ')
		else:
			print(indent * node.depth, node.shape, end=' ')

	def visitTransp(self, node:AST.Transp, args=None):
		node.expr.depth = node.depth + 1
		print(indent * node.depth, end=' ')
		self.visit(node.expr)
		print("^T", end=' ')

	def visitReshape(self, node:AST.Reshape, args=None):
		node.expr.depth = node.depth + 1
		print(indent * node.depth, "reshape", end=' ')
		self.visit(node.expr)
		if (node.order):
			print(node.shape, "order", node.order, end=' ')
		else:
			print(node.shape, end=' ')
	
	def visitPool(self, node:AST.Pool, args=None):
		node.expr.depth = node.depth + 1
		print(indent * node.depth, node.poolType, end=' ')
		self.visit(node.expr)

	def visitIndex(self, node:AST.Index, args=None):
		node.expr.depth = node.depth + 1
		print(indent * node.depth, end = ' ')
		self.visit(node.expr)
		print("[", end=' ')
		for x in node.index:
			print(x, end = ' ')
		print("]", end=' ')

	def visitUOp(self, node:AST.UOp, args=None):
		node.expr.depth = node.depth + 1
		print(indent * node.depth, AST.OperatorsSymbolDict[node.op.name], end=' ')
		self.visit(node.expr)

	def visitBOp(self, node:AST.BOp, args=None):
		node.expr1.depth = node.expr2.depth = node.depth + 1
		print(indent * node.depth, end=' ')
		self.visit(node.expr1)
		print(AST.OperatorsSymbolDict[node.op.name], end=' ')
		self.visit(node.expr2)

	def visitFunc(self, node:AST.Func, args=None):
		print(indent * node.depth, AST.OperatorsSymbolDict[node.op.name], end=' ')
		node.expr.depth = node.depth + 1
		self.visit(node.expr)

	def visitLet(self, node:AST.Let, args=None):
		if (node.decl is not None):
			node.decl.depth = node.depth + 1
		if (node.expr is not None):
			node.expr.depth = node.depth + 1
		print(indent * node.depth, "(", end=' ')
		print("let", end=' ')
		self.visit(node.name)
		print("=", end=' ')
		self.visit(node.decl)
		print("in", "{", node.metadata[AST.ASTNode.mtdKeyTFOpName], node.metadata[AST.ASTNode.mtdKeyTFNodeName], "}", end='\n')
		self.visit(node.expr)
		print(')',end='')

	def visitUninterpFuncCall(self, node:AST.UninterpFuncCall, args=None):
		print(indent * node.depth, "UninterpFuncCall", node.funcName, end=' ')
		for x in node.argsList:
			self.visit(x)

	def visitArgMax(self, node:AST.ArgMax, args=None):
		print(indent * node.depth, "ArgMax", end=' ')
		self.visit(node.expr)
		self.visit(node.dim)

	def visitReduce(self, node:AST.Reduce, args=None):
		print(indent * node.depth, "reduce", AST.OperatorsSymbolDict[node.op.name], end=' ')
		self.visit(node.expr)
		self.visit(node.dim)
		self.visit(node.keepdims)

	def visitInput(self, node:AST.Input, args=None):
		print(indent * node.depth, "input( ", node.shape, node.dataType, end='')
		print(" )", end='')

	def visitFusedBatchNorm(self, node:AST.FusedBatchNorm, args=None):
		node.expr.depth = node.multExpr.depth = node.addExpr.depth = node.depth + 1
		print(indent * node.depth, "FusedBatchNorm", end=' ')
		self.visit(node.expr)
		self.visit(node.multExpr)
		self.visit(node.addExpr)


