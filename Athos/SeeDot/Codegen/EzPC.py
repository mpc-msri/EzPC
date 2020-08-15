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

import Util
import IR.IR as IR
import numpy as np
import Type as Type
import IR.IRUtil as IRUtil
from Codegen.CodegenBase import CodegenBase

class EzPC(CodegenBase):
	def __init__(self, writer, globalDecls, debugVar):
		self.out = writer
		self.globalDecls = globalDecls
		self.consSFUsed = Util.Config.consSF
		self.debugVar = debugVar

	def printAll(self, prog:IR.Prog, expr:IR.Expr):
		self._out_prefix()
		self.print(prog)
		self._out_suffix(expr)

	def _out_prefix(self):
		self.out.printf('\n\ndef void main(){\n')
		self.out.increaseIndent()
		self.printGlobalVarDecls()

	def printGlobalVarDecls(self):
		for decl in self.globalDecls:
			typ_str = IR.DataType.getIntStr()
			idf_str = decl
			declProp = self.globalDecls[decl]
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
		if (ir.inputByParty==0):
			inputByPartyStr = "SERVER"
		elif (ir.inputByParty==1):
			inputByPartyStr = "CLIENT"
		else:
			assert(False) #For now the only supported values of party to input is 0 or 1
		self.out.printf('input({0}, {1}, '.format(inputByPartyStr, ir.expr.idf), indent=True)
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
		variableLabel = 'pl' if not(ir.isSecret) else 'al'

		if Type.isInt(ir.typeExpr): shape_str = ''
		elif Type.isTensor(ir.typeExpr): shape_str = ''.join(['[' + str(n) + ']' for n in ir.typeExpr.shape])
		self.out.printf('%s_%s%s %s', typ_str, variableLabel, shape_str, ir.varIdf, indent=True)
		if (ir.value):
			assert(Type.isInt(ir.typeExpr)) #In EzPC ints can be declared and assigned in same line, not tensors
			self.out.printf(' = %s', str(ir.value[0]))
		self.out.printf(';\n\n')

	def _out_suffix(self, expr:IR.Expr):
		if self.debugVar is None:
			self.out.printf('output(CLIENT, ' + expr.idf + ');\n', indent=True)
		else:
			self.out.printf('output(CLIENT, ' + self.debugVar + ');\n', indent=True)
		self.out.decreaseIndent()
		self.out.printf('}\n', indent=True)
	
