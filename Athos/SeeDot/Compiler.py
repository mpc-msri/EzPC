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
				disableRMO, disableLivenessOpti, disableAllOpti):
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
		Util.Config.wordLength = int(bitlen)
		Util.Config.outputFileName = outputFileName
		Util.Config.disableRMO = disableRMO
		Util.Config.disableLivenessOpti = disableLivenessOpti
		Util.Config.disableAllOpti = disableAllOpti
	
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
				# Perform optimizations on the AST
				ReluMaxpoolOpti.ReluMaxpoolOpti().visit(ast)

			if not(Util.Config.disableLivenessOpti):
				print("Performing Liveness Optimization...")
				# Perform liveness analysis optimization on the AST
				mtdAST = MtdAST()
				LivenessOpti.LivenessAnalysis().visit(ast)
				LivenessOpti.LivenessOpti().visit(ast, [mtdAST, 0, {}])
		
		if Util.Config.printASTBool:
			PrintAST().visit(ast)
			sys.stdout.flush()

 		# Perform type inference
		InferType().visit(ast)

		IRUtil.init()
		compiler = IRBuilderCSF()
		res = compiler.visit(ast)

		# Insert a generic start_computation and end_computation function call after all input IR statements.
		res = self.insertStartEndFunctionCalls(res);

		writer = Writer(Util.Config.outputFileName)

		if Util.forEzPC():
			codegen = EzPCCodegen(writer, compiler.decls)
		else:
			assert False

		codegen.printAll(*res)
		writer.close()
