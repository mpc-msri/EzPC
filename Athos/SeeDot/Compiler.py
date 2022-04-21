"""

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

"""

import os, sys
import _pickle as pickle

import Util
import IR.IR as IR
import AST.AST as AST
from Writer import Writer
import Type as Type
from Type import InferType
import IR.IRUtil as IRUtil
from AST.PrintAST import PrintAST
from AST.MtdAST import MtdAST
from IR.IRBuilderCSF import IRBuilderCSF
from Codegen.EzPC import EzPC as EzPCCodegen
import Optimizations.ReluMaxpoolOpti as ReluMaxpoolOpti
import Optimizations.GarbageCollector as GarbageCollector
from collections import OrderedDict


class Compiler:
    def __init__(
        self,
        version,
        target,
        sfType,
        astFile,
        printASTBool,
        consSF,
        bitlen,
        outputFileName,
        disableRMO,
        disableLivenessOpti,
        disableTruncOpti,
        disableAllOpti,
        debugVar,
    ):
        assert version == Util.Version.Fixed
        assert target == Util.Target.EzPC
        assert sfType == Util.SFType.Constant
        assert astFile is not None
        assert isinstance(printASTBool, bool)
        assert consSF is not None
        assert bitlen is not None
        assert outputFileName is not None
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
        if Util.Config.actualWordLength > 32:
            Util.Config.wordLength = 64
        else:
            Util.Config.wordLength = 32

    def insertStartEndFunctionCalls(self, res: (IR.Prog, IR.Expr)):
        prog = res[0]
        expr = res[1]
        for ii in range(len(prog.cmd_l)):
            if not (isinstance(prog.cmd_l[ii], IR.Input)) and not (
                isinstance(prog.cmd_l[ii], IR.Comment)
            ):
                prog.cmd_l.insert(ii, IR.FuncCall("StartComputation", []))
                break

        first_output_pos = None
        for ii in range(len(prog.cmd_l)):
            if isinstance(prog.cmd_l[ii], IR.Output):
                first_output_pos = ii
                break
        prog.cmd_l.insert(first_output_pos, IR.FuncCall("EndComputation", []))
        return (prog, expr)

    def fixNames(self, res: (IR.Prog, IR.Expr), compiler: IRBuilderCSF):
        (prog, expr) = res
        prog = prog.updateName(compiler.expr_mapping)
        expr = expr.updateName(compiler.expr_mapping)
        return (prog, expr)

    def fixOuputScale(self, res: (IR.Prog, IR.Expr), compiler: IRBuilderCSF):
        (prog, expr) = res
        scaledown_cmd_list = []

        first_output_pos = None
        i = 0
        for cmd in prog.cmd_l:
            if type(cmd) == IR.Output:
                if first_output_pos is None:
                    first_output_pos = i
                var = cmd.expr
                assert type(var) == IR.Var
                output_scale = compiler.scaleFacMapping[var.idf]
                if output_scale > Util.Config.consSF:
                    scale_down = output_scale - Util.Config.consSF
                    var_type = compiler.typeInfo[var.idf]
                    if Type.isInt(var_type):
                        output_shape = []
                    if Type.isTensor(var_type):
                        output_shape = var_type.shape
                    argsDict = OrderedDict()
                    funcName = "ScaleDown"
                    for ii, curDimSize in enumerate(output_shape):
                        argsDict[IR.Int(curDimSize, 32)] = "size_" + str(ii)
                    funcName = funcName + str(len(output_shape))
                    argsDict[var] = "expr"
                    argsDict[IR.Int(scale_down, 32)] = "consSF"
                    funcCall = IR.FuncCall(funcName, argsDict)
                    scaledown_cmd_list.append(funcCall)
                # ArgMax sets scale to -1
                if output_scale == -1:
                    continue
                if output_scale < Util.Config.consSF:
                    assert (
                        False
                    ), "Scale up shouldnt be required of final output {} -> {}. We lost precision somewhere".format(
                        output_scale, Util.Config.consSF
                    )
            i += 1
        final_cmd_list = (
            prog.cmd_l[0:first_output_pos]
            + scaledown_cmd_list
            + prog.cmd_l[first_output_pos:]
        )
        prog = IR.Prog(final_cmd_list)
        return (prog, expr)

    def run(self):
        with open(Util.Config.astFile, "rb") as ff:
            ast = pickle.load(ff)

        if not (Util.Config.disableAllOpti):
            if not (Util.Config.disableRMO):
                print("Performing Relu-maxpool optimization...")
                ReluMaxpoolOpti.ReluMaxpoolOpti().visit(ast)
                print("Relu-maxpool optimization done.")

            if not (Util.Config.disableLivenessOpti):
                print("Performing Garbage collection...")
                mtdAST = MtdAST()
                GC = GarbageCollector.GarbageCollector(ast)
                GC.run([mtdAST])
                print("Garbage collection done.")

        # Perform type inference and annotate nodes with type information
        InferType().visit(ast)

        # if Util.Config.printASTBool :
        if False:
            PrintAST().visit(ast)
            print("\n")
            sys.stdout.flush()

        IRUtil.init()
        compiler = IRBuilderCSF()
        res = compiler.visit(ast)
        res = self.fixOuputScale(res, compiler)
        res = self.fixNames(res, compiler)

        Util.write_debug_info(compiler.name_mapping)

        # Insert a generic start_computation and end_computation function call after all input IR statements.
        res = self.insertStartEndFunctionCalls(res)
        writer = Writer(Util.Config.outputFileName)
        debugVarEzPCName = (
            compiler.name_mapping[Util.Config.debugVar]
            if (Util.Config.debugVar in compiler.name_mapping)
            else None
        )

        if Util.forEzPC():
            codegen = EzPCCodegen(writer, compiler.globalDecls, debugVarEzPCName)
        else:
            assert False

        codegen.printAll(*res)
        writer.close()
