import AST.AST as AST

class ASTVisitor:
	def visitInt(self, node:AST.Int, args=None):
		pass

	def visitFloat(self, node:AST.Float, args=None):
		pass

	def visitId(self, node:AST.ID, args=None):
		pass

	def visitDecl(self, node:AST.Decl, args=None):
		if node.valueList:
			for elem in node.valueList:
				self.visit(elem, args)

	def visitTransp(self, node:AST.Transp, args=None):
		self.visit(node.expr, args)

	def visitReshape(self, node:AST.Reshape, args=None):
		self.visit(node.expr, args)
	
	def visitPool(self, node:AST.Pool, args=None):
		self.visit(node.expr, args)

	def visitIndex(self, node:AST.Index, args=None):
		self.visit(node.expr, args)

	def visitUOp(self, node:AST.UOp, args=None):
		self.visit(node.expr, args)

	def visitBOp(self, node:AST.BOp, args=None):
		self.visit(node.expr1, args)
		self.visit(node.expr2, args)

	def visitFunc(self, node:AST.Func, args=None):
		self.visit(node.expr, args)

	def visitLet(self, node:AST.Let, args=None):
		self.visit(node.name, args)
		self.visit(node.decl, args)
		self.visit(node.expr, args)

	def visitUninterpFuncCall(self, node:AST.UninterpFuncCall, args=None):
		for elem in node.argsList:
			self.visit(elem, args)

	def visitArgMax(self, node:AST.ArgMax, args=None):
		self.visit(node.expr, args)
		self.visit(node.dim, args)

	def visitReduce(self, node:AST.Reduce, args=None):
		self.visit(node.expr, args)
		self.visit(node.dim, args)
		self.visit(node.keepdims, args)

	def visitInput(self, node:AST.Input, args=None):
		pass

	def visitFusedBatchNorm(self, node:AST.FusedBatchNorm, args=None):
		self.visit(node.expr, args)
		self.visit(node.multExpr, args)
		self.visit(node.addExpr, args)

	def visit(self, node, args=None):
		if node is None:
			return
		if isinstance(node, AST.Int):
			return self.visitInt(node, args)
		elif isinstance(node, AST.Float):
			return self.visitFloat(node, args)
		elif isinstance(node, AST.ID):
			return self.visitId(node, args)
		elif isinstance(node, AST.Decl):
			return self.visitDecl(node, args)
		elif isinstance(node, AST.Transp):
			return self.visitTransp(node, args)
		elif isinstance(node, AST.Reshape):
			return self.visitReshape(node, args)
		elif isinstance(node, AST.Pool):
			return self.visitPool(node, args)
		elif isinstance(node, AST.Index):
			return self.visitIndex(node, args)
		elif isinstance(node, AST.UOp):
			return self.visitUOp(node, args)
		elif isinstance(node, AST.BOp):
			return self.visitBOp(node, args)
		elif isinstance(node, AST.Func):
			return self.visitFunc(node, args)
		elif isinstance(node, AST.Let):
			return self.visitLet(node, args)
		elif isinstance(node, AST.UninterpFuncCall):
			return self.visitUninterpFuncCall(node, args)
		elif isinstance(node, AST.ArgMax):
			return self.visitArgMax(node, args)
		elif isinstance(node, AST.Reduce):
			return self.visitReduce(node, args)
		elif isinstance(node, AST.Input):
			return self.visitInput(node, args)
		elif isinstance(node, AST.FusedBatchNorm):
			return self.visitFusedBatchNorm(node, args)
		elif node:
			raise Exception('Node instance not matched.')
		else:
			pass
