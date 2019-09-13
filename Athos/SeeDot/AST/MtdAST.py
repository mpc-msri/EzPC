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

	def visitTransp(self, node:AST.Transp, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitReshape(self, node:AST.Reshape, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitPool(self, node:AST.Pool, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)
	
	def visitIndex(self, node:AST.Index, mtd:dict):
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
