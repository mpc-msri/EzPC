import Util
import IR.IR as IR
import numpy as np
import Type as Type
import IR.IRUtil as IRUtil
from Codegen.CodegenBase import CodegenBase

class EzPC(CodegenBase):
	def __init__(self, writer, decls):
		self.out = writer
		self.decls = decls
		self.consSFUsed = Util.Config.consSF

	def printAll(self, prog:IR.Prog, expr:IR.Expr):
		self._out_prefix()
		self.print(prog)
		self._out_suffix(expr)

	def _out_prefix(self):
		self.out.printf('\n\ndef void main(){\n')
		self.out.increaseIndent()
		# self.printVarDecls()

	def printVarDecls(self):
		for decl in self.decls:
			typ_str = IR.DataType.getIntStr()
			idf_str = decl
			declProp = self.decls[decl]
			curType = declProp[0]
			if (len(declProp)>1):
				# If label specified in decl, then use that
				assert(len(declProp) >= 2 and len(declProp) <= 3) #For now only type, label and bitlen should be present
				variableLabel = ('pl' if (declProp[1] == 'public') else 'al')
				if (len(declProp) == 3):
					bitlen = declProp[2]
					typ_str = IR.DataType.getIntStrForBitlen(bitlen)
			else:
				# If variable unspecified, then default to secret
				variableLabel = 'al'
			if Type.isInt(curType): shape_str = ''
			elif Type.isTensor(curType): shape_str = ''.join(['[' + str(n) + ']' for n in curType.shape])
			self.out.printf('%s_%s%s %s;\n', typ_str, variableLabel, shape_str, idf_str, indent=True)
		self.out.printf('\n')

	def printFuncCall(self, ir:IR.FuncCall):
		self.out.printf("%s(" % ir.name, indent = True)
		keys = list(ir.argList)
		for i in range(len(keys)):
			arg = keys[i]
			self.print(arg)
			if i != len(keys) - 1:
				self.out.printf(", ")
		self.out.printf(");\n\n")

	def printForHeader(self, ir):
		assert(ir.endInt is not None and ir.endCond is None)
		self.out.printf('for ', indent=True)
		self.print(ir.var)
		self.out.printf(' = [%d: %d]{\n ', ir.st, ir.endInt)

	def printFor(self, ir):
		self.printForHeader(ir)
		self.out.increaseIndent()
		for cmd in ir.cmd_l:
			self.print(cmd)
		self.out.decreaseIndent()
		self.out.printf('};\n', indent=True)

	def printInt(self, ir:IR.Int):
		if (isinstance(ir.n, np.int32)):
			self.out.printf('%d', ir.n)
		elif (isinstance(ir.n, np.int64)):
			self.out.printf('%dL', ir.n)
		else:
			assert False

	def printInput(self, ir:IR.Input):
		self.out.printf('input(CLIENT, ' + ir.expr.idf + ' , ', indent=True)
		#assert(ir.dataType in ["DT_INT32"]) ####TODO: fix this
		if Util.Config.wordLength == 32:
			self.out.printf('int32_')
		elif Util.Config.wordLength == 64:
			self.out.printf('int64_')
		else:
			assert False
		if ir.isSecret:
			self.out.printf('al')
		else:
			self.out.printf('pl')
		for curDim in ir.shape:
			self.out.printf('[' + str(curDim) + ']')
		self.out.printf(');\n\n')

	def printComment(self, ir):
		self.out.printf('(* ' + ir.msg + ' *)\n', indent = True)

	def printDecl(self, ir):
		typ_str = IR.DataType.getIntStrForBitlen(ir.bitlen)
		variableLabel = 'pl' if ir.isSecret=="public" else 'al'

		if Type.isInt(ir.typeExpr): shape_str = ''
		elif Type.isTensor(ir.typeExpr): shape_str = ''.join(['[' + str(n) + ']' for n in ir.typeExpr.shape])
		self.out.printf('%s_%s%s %s;\n', typ_str, variableLabel, shape_str, ir.varIdf, indent=True)
		self.out.printf('\n')

	def _out_suffix(self, expr:IR.Expr):
		self.out.printf('output(SERVER, ' + expr.idf + ');\n', indent=True)
		self.out.decreaseIndent()
		self.out.printf('}\n', indent=True)
	
