'''

Authors: Sridhar Gopinath, Nishant Kumar.

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

import numpy as np
from enum import Enum

import IR.IR as IR
import Type as Type
from Util import *

class CodegenBase:
	#TODO : Clean this file of extra info
	def __init__(self, writer):
		self.out = writer
	
	def printOp(self, ir):
		self.out.printf('%s', ir.name)

	def printVar(self, ir):
		self.out.printf('%s', ir.idf)
		for e in ir.idx:
			self.out.printf('[')
			self.print(e)
			self.out.printf(']')

	def printBool(self, ir):
		self.out.printf({True:'true', False:'false'}[ir.b])

	def printIntUop(self, ir):
		self.out.printf('(')
		self.print(ir.op)
		self.print(ir.e)
		self.out.printf(')')

	def printTypeCast(self, ir):
		self.out.printf('(')
		self.out.printf('(' + ir.type + ')')
		self.print(ir.expr)
		self.out.printf(')')

	def printIntBop(self, ir):
		self.out.printf('(')
		self.print(ir.e1)
		self.out.printf(' ')
		self.print(ir.op)
		self.out.printf(' ')
		self.print(ir.e2)
		self.out.printf(')')

	def printBoolUop(self, ir):
		self.out.printf('(')
		self.print(ir.op)
		self.print(ir.e)
		self.out.printf(')')

	def printBoolBop(self, ir):
		self.out.printf('(')
		self.print(ir.e1)
		self.out.printf(' ')
		self.print(ir.op)
		self.out.printf(' ')
		self.print(ir.e2)
		self.out.printf(')')

	def printBoolCop(self, ir):
		self.out.printf('(')
		self.print(ir.e1)
		self.out.printf(' ')
		self.print(ir.op)
		self.out.printf(' ')
		self.print(ir.e2)
		self.out.printf(')')

	def printCExpr(self, ir):
		self.out.printf('(')
		self.print(ir.cond)
		self.out.printf(' ? ')
		self.print(ir.et)
		self.out.printf(' : ')
		self.print(ir.ef)
		self.out.printf(')')

	def printAssn(self, ir):
		self.out.printf('', indent=True)
		self.print(ir.var)
		self.out.printf(' = ')
		self.print(ir.e)
		self.out.printf(';\n')

	def printIf(self, ir):
		self.out.printf('if (', indent = True)
		self.print(ir.cond)
		self.out.printf(') {\n')
		
		self.out.increaseIndent()
		for cmd in ir.trueCmds:
			self.print(cmd)
		self.out.decreaseIndent()

		if len(ir.falseCmds) == 0:
			self.out.printf('}\n', indent=True)
			return

		self.out.printf('} else {\n', indent=True)

		self.out.increaseIndent()
		for cmd in ir.falseCmds:
			self.print(cmd)
		self.out.decreaseIndent()

		self.out.printf('}\n', indent=True)

	def printFor(self, ir):
		self.printForHeader(ir)
		self.out.increaseIndent()
		for cmd in ir.cmd_l:
			self.print(cmd)
		self.out.decreaseIndent()
		self.out.printf('}\n', indent=True)

	def printForHeader(self, ir):
		self.out.printf('for (%s ', IR.DataType.getIntStr(), indent=True)
		self.print(ir.var)
		self.out.printf(' = %d; ', ir.st)
		self.print(ir.cond)
		self.out.printf('; ')
		self.print(ir.var)
		self.out.printf('++) {\n')

	def printWhile(self, ir):
		self.out.printf('while (', indent=True)
		self.print(ir.expr)
		self.out.printf(') {\n')
		self.out.increaseIndent()
		for cmd in ir.cmds:
			self.print(cmd)
		self.out.decreaseIndent()
		self.out.printf('}\n', indent=True)

	def printProg(self, ir):
		for cmd in ir.cmd_l:
			self.print(cmd)

	def printPrint(self, ir):
		self.out.printf('cout << ', indent=True)
		self.print(ir.expr)
		self.out.printf(' << endl;\n')

	def printPrintAsFloat(self, ir):
		self.out.printf('cout << ((float)(', indent=True)
		self.print(ir.expr)
		self.out.printf(')) * ' + str(2 ** ir.expnt) + ' << "";\n')

	def print(self, ir):
		if isinstance(ir, IR.Int):
			return self.printInt(ir)
		elif isinstance(ir, IR.Var):
			return self.printVar(ir)
		elif isinstance(ir, IR.Bool):
			return self.printBool(ir)
		elif isinstance(ir, IR.IntUop):
			return self.printIntUop(ir)
		elif isinstance(ir, IR.Exp):
			return self.printExp(ir)
		elif isinstance(ir, IR.TypeCast):
			return self.printTypeCast(ir)
		elif isinstance(ir, IR.IntBop):
			return self.printIntBop(ir)
		elif isinstance(ir, IR.BoolUop):
			return self.printBoolUop(ir)
		elif isinstance(ir, IR.BoolBop):
			return self.printBoolBop(ir)
		elif isinstance(ir, IR.BoolCop):
			return self.printBoolCop(ir)
		elif isinstance(ir, IR.CExpr):
			return self.printCExpr(ir)
		elif isinstance(ir, IR.Assn):
			return self.printAssn(ir)
		elif isinstance(ir, IR.If):
			return self.printIf(ir)
		elif isinstance(ir, IR.For):
			return self.printFor(ir)
		elif isinstance(ir, IR.While):
			return self.printWhile(ir)
		elif isinstance(ir, IR.Comment):
			return self.printComment(ir)
		elif isinstance(ir, IR.Pragmas):
			return self.printPragmas(ir)
		elif isinstance(ir, IR.Prog):
			return self.printProg(ir)
		elif isinstance(ir, IR.Memset):
			return self.printMemset(ir)
		elif isinstance(ir, IR.Print):
			return self.printPrint(ir)
		elif isinstance(ir, IR.PrintAsFloat):
			return self.printPrintAsFloat(ir)
		elif isinstance(ir, IR.FuncCall):
			return self.printFuncCall(ir)
		elif isinstance(ir, IR.Op.Op):
			return self.printOp(ir)
		elif isinstance(ir, IR.Input):
			return self.printInput(ir)
		elif isinstance(ir, IR.Decl):
			return self.printDecl(ir)
		else:
			assert False
