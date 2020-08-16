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

import os, sys
import _pickle as pickle

import Util
import IR.IR as IR
import AST.AST as AST
from Writer import Writer
from Type import InferType
import IR.IRUtil as IRUtil
from AST.PrintAST  import PrintAST
from AST.MtdAST import MtdAST
from IR.IRBuilderCSF import IRBuilderCSF
from Codegen.EzPC import EzPC as EzPCCodegen
import Optimizations.ReluMaxpoolOpti as ReluMaxpoolOpti
import Optimizations.LivenessOpti as LivenessOpti

class Compiler:
	def __init__(self, version, target, sfType, astFile, printASTBool, consSF, bitlen, outputFileName,
				disableRMO, disableLivenessOpti, disableTruncOpti, disableAllOpti, debugVar):
		assert(version == Util.Version.Fixed)
		assert(target == Util.Target.EzPC)
		assert(sfType == Util.SFType.Constant)
		assert(astFile is not None)
		assert(isinstance(printASTBool, bool))
		assert(consSF is not None)
		assert(bitlen is not None)
		assert(outputFileName is not None)
		Util.Config.version = version
		Util.Config.target = target
		Util.Config.sfType = sfType
		Util.Config.astFile = astFile
		Util.Config.printASTBool = printASTBool
		Util.Config.consSF = consSF
		Util.Config.outputFileName = outputFileName
		Util.Config.disableRMO = disableRMO
		Util.Config.disableLivenessOpti = disableLivenessOpti
		Util.Config.disableTruncOpti = disableTruncOpti
		Util.Config.disableAllOpti = disableAllOpti
		Util.Config.debugVar = debugVar
		Util.Config.actualWordLength = int(bitlen)
		if (Util.Config.actualWordLength > 32):
			Util.Config.wordLength = 64
		else:
			Util.Config.wordLength = 32
	
	def insertStartEndFunctionCalls(self, res:(IR.Prog, IR.Expr)):
		prog = res[0]
		expr = res[1]
		for ii in range(len(prog.cmd_l)):
			if not(isinstance(prog.cmd_l[ii], IR.Input)) and not(isinstance(prog.cmd_l[ii], IR.Comment)):
				prog.cmd_l.insert(ii, IR.FuncCall('StartComputation',[]))
				break;
		prog.cmd_l.append(IR.FuncCall('EndComputation', []))
		return (prog, expr)

	def run(self):
		with open(Util.Config.astFile, 'rb') as ff:
			ast = pickle.load(ff)

		if not(Util.Config.disableAllOpti):
			if not(Util.Config.disableRMO):
				print("Performing Relu-maxpool optimization...")
				ReluMaxpoolOpti.ReluMaxpoolOpti().visit(ast)
				print("Relu-maxpool optimization done.")
		
			if not(Util.Config.disableLivenessOpti):
				print("Performing Liveness Optimization...")
				mtdAST = MtdAST()
				LivenessOpti.LivenessAnalysis().visit(ast)
				LivenessOpti.LivenessOpti().visit(ast, [mtdAST, 0, {}])
				print("Liveness optimization done.")
		
		if Util.Config.printASTBool:
			PrintAST().visit(ast)
			sys.stdout.flush()

 		# Perform type inference
		InferType().visit(ast)

		IRUtil.init()
		compiler = IRBuilderCSF()
		res = compiler.visit(ast)

		Util.write_debug_info(compiler.name_mapping) 

		# Insert a generic start_computation and end_computation function call after all input IR statements.
		res = self.insertStartEndFunctionCalls(res);
		writer = Writer(Util.Config.outputFileName)
		debugVarEzPCName = compiler.name_mapping[Util.Config.debugVar] if (Util.Config.debugVar in compiler.name_mapping) else None  

		if Util.forEzPC():
			codegen = EzPCCodegen(writer, compiler.globalDecls, debugVarEzPCName)
		else:
			assert False

		codegen.printAll(*res)
		writer.close()
