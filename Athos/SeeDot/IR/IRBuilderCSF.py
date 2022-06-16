"""

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

"""

import os, math, sys
import operator
import numpy as np
from collections import OrderedDict

import Util
import IR.IR as IR
import Type as Type
import AST.AST as AST
import IR.IRUtil as IRUtil
from AST.ASTVisitor import ASTVisitor
from AST.IRBuilderAST import IRBuilderAST


class IRBuilderCSF(IRBuilderAST):
    varNameDelim = ""

    def __init__(self, intPartBitwidth=-1):
        # For tracking temp variables
        self._var_cnt = 0
        self._iter_cnt = 0

        # Global variables
        #   Used to note declarations which will go before any statements
        #   But since this affects memory consumption, use carefully
        self.globalDecls = (
            {}
        )  # Mapping of (identifier name (string) -> list of [type, secret/public variable, bitlen of decl])
        #   The 2nd arg can be either 'secret' or 'public'.
        #   If public/secret unspecified, default to 'secret'.
        #   The 3rd arg is used to specify the bitlen of the decl.

        # Name mapping from SeeDot names to new names is useful for debugging
        self.name_mapping = {}
        self.expr_mapping = {}

        self.actualbitwidth = Util.Config.actualWordLength

        # This is for optimizing the #truncation calls
        self.scaleFac = Util.Config.consSF
        self.bitwidth = Util.Config.wordLength
        self.intPartBitwidth = intPartBitwidth
        if self.intPartBitwidth == -1:
            self.intPartBitwidth = self.bitwidth - 2 * self.scaleFac
        self.scaleFacMapping = {}

    def getConsSF(self):
        return Util.Config.consSF

    # Variable and iterators creation
    def getTempVars(self, n: int):
        return [self.getTempVar() for i in range(n)]

    def getTempVar(self):
        var = IR.Var("tmp" + str(self._var_cnt))
        self._var_cnt += 1
        return var

    def getTempIterators(self, n: int):
        return [self.getTempIterator() for i in range(n)]

    def getTempIterator(self):
        var = IR.Var("i" + str(self._iter_cnt))
        self._iter_cnt += 1
        return var

    # Computing exponent and intervals
    def get_expnt(self, maxabs: float):  # -> int
        return self.getConsSF()

    def addTruncateFunctionCallHelper(
        self, exprNumToScaleDown: int, expr1: IR.Var, expr2: IR.Var, node: AST.BOp
    ):
        assert isinstance(node, AST.BOp)
        if exprNumToScaleDown == 1:
            exprToScaleDown = expr1
            nodeToScaleDown = node.expr1
        else:
            assert exprNumToScaleDown == 2
            exprToScaleDown = expr2
            nodeToScaleDown = node.expr2
        return (exprToScaleDown, nodeToScaleDown)

    def addTruncateFunctionCall(
        self, node: AST.ASTNode, nodeTypeStr: str, expr: IR.Var, consSF: int
    ):
        comment = IR.Comment("Truncation before {0} node.".format(nodeTypeStr))
        argsDict = OrderedDict()
        funcName = "ScaleDown"
        if not (Type.isInt(node.type)):
            outputShape = node.type.shape
            for ii, curDimSize in enumerate(outputShape):
                argsDict[IR.Int(curDimSize, 32)] = "size_" + str(ii)
            funcName = funcName + str(len(outputShape))
        argsDict[expr] = "expr"
        argsDict[IR.Int(consSF, 32)] = "consSF"
        funcCall = IR.FuncCall(funcName, argsDict)
        prog = IR.Prog([comment, funcCall])
        return prog

    def isModel(self, node: AST.ASTNode):
        if node.type.taint == Type.Taints.SERVER:
            return True
        else:
            return False

    # =================
    # Visit Functions
    # =================

    def visitInt(self, node: AST.Int, args=None):
        n = node.value
        prog = IR.Prog([IR.Comment("Int node, isSecret = {0}.".format(node.isSecret))])
        expr = self.getTempVar()
        bitlen = -1
        if node.bitLen:
            bitlen = node.bitLen
        prog = IRUtil.prog_merge(
            IR.Prog([IR.Decl(expr.idf, node.type, bitlen, node.isSecret, [n])]), prog
        )
        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[expr.idf] = self.scaleFac if node.isScaled else 0
        return (prog, expr)

    def visitFloat(self, node: AST.Float, args=None):
        r = node.value
        p = self.get_expnt(abs(r))
        k = IR.DataType.getInt(np.ldexp(r, p))
        expr = self.getTempVar()
        prog = IR.Prog(
            [
                IR.Comment(
                    "Float to int : {0} to {1}, isSecret = {2}.".format(
                        str(r), str(k), node.isSecret
                    )
                )
            ]
        )
        prog = IRUtil.prog_merge(
            IR.Prog([IR.Decl(expr.idf, node.type, -1, node.isSecret, [k])]), prog
        )
        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[expr.idf] = self.scaleFac
        return (prog, expr)

    def visitId(self, node: AST.ID, args=None):
        idf = node.name
        prog = IR.Prog([])
        expr = IR.Var(idf)
        if not (Util.Config.disableTruncOpti):
            assert expr.idf in self.scaleFacMapping
        return (prog, expr)

    def visitDecl(self, node: AST.Decl, args=None):
        def helperAssignGen(l1, l2, allComb):
            if l2 == []:
                allComb.append(l1)
            else:
                for cur in range(l2[0]):
                    helperAssignGen(l1 + [cur], l2[1:], allComb)

        prog = IR.Prog([])
        expr = self.getTempVar()
        expr.inputVar = True

        # If there is a valueList, then add assignment commands
        specialBitLen = -1
        if node.valueList:
            # Add assignment statements for each element of the tensor in a different array
            comment = IR.Comment(str(node.metadata))
            prog = IRUtil.prog_merge(
                prog,
                IR.Prog([comment, IR.Comment("Element assignments for " + expr.idf)]),
            )
            allComb = []
            helperAssignGen([], node.shape, allComb)
            for i, curComb in enumerate(allComb):
                curVal = node.valueList[i]
                finalVal = None
                if isinstance(curVal, AST.Int):
                    finalVal = IR.Int(curVal.value, curVal.bitLen)
                    if specialBitLen == -1 and curVal.bitLen != Util.Config.wordLength:
                        specialBitLen = curVal.bitLen
                elif isinstance(curVal, AST.Float):
                    finalVal = IR.DataType.getInt(
                        np.ldexp(curVal.value, Util.Config.consSF)
                    )
                else:
                    # Assuming the elements can only be either int or floats
                    assert False
                prog = IRUtil.prog_merge(
                    prog,
                    IR.Prog(
                        [
                            IR.Assn(
                                IRUtil.addIndex(
                                    expr, list(map(lambda x: IR.Int(x), curComb))
                                ),
                                finalVal,
                            )
                        ]
                    ),
                )

        prog = IRUtil.prog_merge(
            IR.Prog(
                [
                    IR.Decl(
                        expr.idf,
                        node.type,
                        Util.Config.wordLength
                        if specialBitLen == -1
                        else specialBitLen,
                        node.isSecret,
                    )
                ]
            ),
            prog,
        )

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[expr.idf] = self.scaleFac if node.isScaled else 0

        return (prog, expr)

    def visitTranspose(self, node: AST.Transpose, args=None):
        (inp_prog, inp_arr) = self.visit(node.expr)
        inp_type = node.expr.type
        out_type = node.type
        inp_iters = self.getTempIterators(inp_type.dim)
        out_iters = []
        perm = node.perm
        if perm is None:
            perm = [i for i in reversed(range(len(inp_type.shape)))]
        for i in perm:
            out_iters.append(inp_iters[i])
        out_arr = self.getTempVar()
        out_arr_expr = IRUtil.addIndex(out_arr, out_iters)
        inp_arr_expr = IRUtil.addIndex(inp_arr, inp_iters)
        assign_expr = IR.Assn(out_arr_expr, inp_arr_expr)
        loop = IRUtil.loop(inp_type.shape, inp_iters, [assign_expr])
        # Finalize
        comment1 = IR.Comment(str(node.metadata))
        comment2 = IR.Comment(
            "transpose("
            + inp_arr.idf
            + ", ["
            + ", ".join(str(e) for e in inp_type.shape)
            + "] --> ["
            + ", ".join(str(e) for e in out_type.shape)
            + "])"
        )
        transpose_prog = IR.Prog([comment1, comment2] + loop)
        final_prog = IRUtil.prog_merge(inp_prog, transpose_prog)

        for var in inp_iters:
            final_prog = IRUtil.prog_merge(
                IR.Prog([IR.Decl(var.idf, Type.Int(), isSecret=False)]), final_prog
            )
        final_prog = IRUtil.prog_merge(
            IR.Prog([IR.Decl(out_arr.idf, out_type)]), final_prog
        )

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[out_arr.idf] = self.scaleFacMapping[inp_arr.idf]

        return (final_prog, out_arr)

    def visitSlice(self, node: AST.Slice, args=None):
        (inp_prog, inp_arr) = self.visit(node.expr)
        inp_type = node.expr.type
        out_type = node.type
        out_iters = self.getTempIterators(out_type.dim)
        inp_iters = []
        subscriptRanges = node.subscriptRanges
        for idx, subrange in enumerate(subscriptRanges):
            start = subrange[0]
            inp_iters.append(IRUtil.add(out_iters[idx], IR.Int(start)))

        out_arr = self.getTempVar()
        out_arr_expr = IRUtil.addIndex(out_arr, out_iters)
        inp_arr_expr = IRUtil.addIndex(inp_arr, inp_iters)
        assign_expr = IR.Assn(out_arr_expr, inp_arr_expr)
        loop = IRUtil.loop(out_type.shape, out_iters, [assign_expr])
        # Finalize
        comment1 = IR.Comment(str(node.metadata))
        comment2 = IR.Comment(
            "slice("
            + inp_arr.idf
            + ", ["
            + ", ".join(str(e) for e in inp_type.shape)
            + "] --> ["
            + ", ".join(str(e) for e in out_type.shape)
            + "])"
        )
        slice_prog = IR.Prog([comment1, comment2] + loop)
        final_prog = IRUtil.prog_merge(inp_prog, slice_prog)

        for var in out_iters:
            final_prog = IRUtil.prog_merge(
                IR.Prog([IR.Decl(var.idf, Type.Int(), isSecret=False)]), final_prog
            )
        final_prog = IRUtil.prog_merge(
            IR.Prog([IR.Decl(out_arr.idf, out_type)]), final_prog
        )

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[out_arr.idf] = self.scaleFacMapping[inp_arr.idf]

        return (final_prog, out_arr)

    def visitExpand():
        assert False

    def visitGather(self, node: AST.Gather, args=None):
        (prog_1, expr_1) = self.visit(node.expr)

        out_arr = self.getTempVar()
        out_decl = IR.Prog(
            [
                IR.Decl(
                    out_arr.idf,
                    Type.Tensor(
                        node.shape[1:],
                        node.expr.type.bitlen,
                        node.expr.type.isSecret,
                        node.expr.type.taint,
                    ),
                    isSecret=node.type.isSecret,
                )
            ]
        )

        index_iter = self.getTempIterators(1)
        iter_decl = IR.Prog([IR.Decl(index_iter[0].idf, Type.Int(), isSecret=False)])
        iter_assn = IR.Prog([IR.Assn(index_iter[0], IRUtil.Int(node.index))])

        loopShape = node.shape[1:]
        loopIters = self.getTempIterators(len(node.shape) - 1)

        iter_decl_lst = []
        for var in loopIters:
            iter_decl_lst.append(IR.Decl(var.idf, Type.Int(), isSecret=False))

        loop = IRUtil.loop(
            loopShape,
            loopIters,
            [
                IR.Assn(
                    IRUtil.addIndex(out_arr, loopIters),
                    IRUtil.addIndex(expr_1, index_iter + loopIters),
                )
            ],
        )

        prog_out = IRUtil.prog_merge(prog_1, out_decl)
        prog_out = IRUtil.prog_merge(prog_out, iter_decl)
        prog_out = IRUtil.prog_merge(prog_out, iter_assn)
        prog_out = IRUtil.prog_merge(prog_out, IR.Prog(iter_decl_lst))
        prog_out = IRUtil.prog_merge(prog_out, IR.Prog(loop))

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[out_arr.idf] = self.scaleFacMapping[expr_1.idf]

        return (prog_out, out_arr)

    def visitUnsqueeze(self, node: AST.Unsqueeze, args=None):
        (prog_1, expr_1) = self.visit(node.expr)

        out_arr = self.getTempVar()
        out_decl = IR.Prog(
            [
                IR.Decl(
                    out_arr.idf,
                    Type.Tensor(
                        node.shape[: node.axis] + [1] + node.shape[node.axis :],
                        node.expr.type.bitlen,
                        node.expr.type.isSecret,
                        node.expr.type.taint,
                    ),
                    isSecret=node.type.isSecret,
                )
            ]
        )

        index_iter = self.getTempIterators(1)
        iter_decl = IR.Prog([IR.Decl(index_iter[0].idf, Type.Int(), isSecret=False)])
        iter_assn = IR.Prog([IR.Assn(index_iter[0], IRUtil.zero)])

        loopShape = node.shape
        loopIters = self.getTempIterators(len(node.shape))

        iter_decl_lst = []
        for var in loopIters:
            iter_decl_lst.append(IR.Decl(var.idf, Type.Int(), isSecret=False))

        loop = IRUtil.loop(
            loopShape,
            loopIters,
            [
                IR.Assn(
                    IRUtil.addIndex(
                        out_arr,
                        loopIters[: node.axis] + index_iter + loopIters[node.axis :],
                    ),
                    IRUtil.addIndex(expr_1, loopIters),
                )
            ],
        )

        prog_out = IRUtil.prog_merge(prog_1, out_decl)
        prog_out = IRUtil.prog_merge(prog_out, iter_decl)
        prog_out = IRUtil.prog_merge(prog_out, iter_assn)
        prog_out = IRUtil.prog_merge(prog_out, IR.Prog(iter_decl_lst))
        prog_out = IRUtil.prog_merge(prog_out, IR.Prog(loop))

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[out_arr.idf] = self.scaleFacMapping[expr_1.idf]

        return (prog_out, out_arr)

    def visitReshape(self, node: AST.Reshape, args=None):
        (prog_1, expr_1) = self.visit(node.expr)

        """
        reshape(A, n, h, w)
        cmd1:  t1 = t2 = t3 = 0;
        loop2: for n in 0:N:
                 for h in 0:H:
                   for w in 0:W:
        cmd3:        B[n][h][w] = A[t1][t2][t3]
        cmd4:        t3++;
        cmd5:        if (t3 == WW)
                       t3 = 0;
                       t2++;
                       if (t2 == HH)
                         t2 = 0;
                         t1++;
        """

        # print(node.expr)

        t1 = node.expr.type
        t2 = node.type
        # [] 0
        # [] 0
        # print("Shapes, dim ---")
        # print(t1.shape, t1.dim)
        # print(t2.shape, t2.dim)

        typ_1 = node.expr.type
        typ_2 = node.type

        # Declare variables
        expr_2 = self.getTempVar()
        iters_1 = self.getTempIterators(typ_1.dim)
        iters_2 = self.getTempIterators(typ_2.dim)

        # Initialize to 0
        cmd1 = [IR.Assn(var, IRUtil.zero) for var in iters_1]

        # Incrementing the first index
        # print(iters_1)
        first_iter = iters_1[0]
        cmd4 = IRUtil.incCmd(first_iter)

        # print("Bye")
        # sys.exit(0)

        # Incrementing other indices using a loop
        cmd5 = [cmd4]
        for i in range(1, typ_1.dim):
            curr_iter = iters_1[i]
            curr_size = IR.Int(typ_1.shape[i])
            cmd5 = [
                IRUtil.incCmd(curr_iter),
                IR.If(
                    IRUtil.eq(curr_iter, curr_size),
                    [IRUtil.initVarToZero(curr_iter)] + cmd5,
                ),
            ]

        # Outer loop
        # The iterators are selected based on the selection order specified by the user
        loopShape = []
        loopIters = []

        if node.order:
            for order in node.order:
                order = order - 1
                loopShape.append(typ_2.shape[order])
                loopIters.append(iters_2[order])
        else:
            loopShape = typ_2.shape
            loopIters = iters_2

        loop2 = IRUtil.loop(
            loopShape,
            loopIters,
            [
                IR.Assn(
                    IRUtil.addIndex(expr_2, iters_2), IRUtil.addIndex(expr_1, iters_1)
                )
            ]
            + cmd5,
        )

        # Finalize
        comment1 = IR.Comment(str(node.metadata))
        comment2 = IR.Comment(
            "reshape("
            + expr_1.idf
            + ", "
            + ", ".join(str(e) for e in typ_2.shape)
            + ")"
        )
        reshape_prog = IR.Prog([comment1, comment2] + cmd1 + loop2)
        prog_2 = IRUtil.prog_merge(prog_1, reshape_prog)

        for var in iters_1:
            prog_2 = IRUtil.prog_merge(
                IR.Prog([IR.Decl(var.idf, Type.Int(), isSecret=False)]), prog_2
            )
        for var in iters_2:
            prog_2 = IRUtil.prog_merge(
                IR.Prog([IR.Decl(var.idf, Type.Int(), isSecret=False)]), prog_2
            )
        prog_2 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_2.idf, typ_2)]), prog_2)

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[expr_2.idf] = self.scaleFacMapping[expr_1.idf]

        return (prog_2, expr_2)

    def visitPool(self, node: AST.Pool, args=None):
        (prog_1, expr_1) = self.visit(node.expr)

        [N, H, W, CI] = node.expr.type.shape
        [N1, outH, outW, CI1] = node.type.shape
        assert N == N1 and CI == CI1
        [expr_2] = self.getTempVars(1)

        comment = IR.Comment(str(node.metadata))
        funcCallArgsDict = OrderedDict()
        funcCallArgsDict[IR.Int(N1, 32)] = "N1"
        funcCallArgsDict[IR.Int(outH, 32)] = "outH"
        funcCallArgsDict[IR.Int(outW, 32)] = "outW"
        funcCallArgsDict[IR.Int(CI1, 32)] = "CI1"
        funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.FH], 32)] = "FH"
        funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.FW], 32)] = "FW"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.zPadHLeft], 32)
        ] = "zPadHLeft"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.zPadHRight], 32)
        ] = "zPadHRight"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.zPadWLeft], 32)
        ] = "zPadWLeft"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.zPadWRight], 32)
        ] = "zPadWRight"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.strideH], 32)
        ] = "strideH"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.strideW], 32)
        ] = "strideW"
        funcCallArgsDict[IR.Int(N, 32)] = "N"
        funcCallArgsDict[IR.Int(H, 32)] = "H"
        funcCallArgsDict[IR.Int(W, 32)] = "W"
        funcCallArgsDict[IR.Int(CI, 32)] = "CI"

        funcCallArgsDict[expr_1] = "input"
        funcCallArgsDict[expr_2] = "output"

        funcCall = IR.FuncCall(node.poolType, funcCallArgsDict)
        prog_pool = IR.Prog([comment, funcCall])
        prog_2 = IRUtil.prog_merge(prog_1, prog_pool)
        prog_2 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_2.idf, node.type)]), prog_2)

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[expr_2.idf] = self.scaleFacMapping[expr_1.idf]

        return (prog_2, expr_2)

    def visitUOp(self, node: AST.UOp, args=None):
        (prog_1, expr_1) = self.visit(node.expr)
        op = node.op
        if op == AST.Operators.ADD:
            return (prog_1, expr_1)
        assert op == AST.Operators.SUB

        typ_2 = node.type
        expr_2 = self.getTempVar()

        if Type.isInt(typ_2):
            comment = IR.Comment(str(node.metadata))
            bitlen = node.expr.bitlen
            decl = IR.Decl(expr_2.idf, node.type, typ_2.bitlen, typ_2.isSecret)
            assign = IR.Assn(expr_2, IRUtil.negate(expr_1))
            prog_2 = IRUtil.prog_merge(prog_1, IR.Prog([comment, decl, assign]))
        else:
            # decl fresh vars
            iters = self.getTempIterators(typ_2.dim)

            # cmdl_assn
            expr_1_elt = IRUtil.addIndex(expr_1, iters)
            expr_2_elt = IRUtil.addIndex(expr_2, iters)
            cmdl_assn = IRUtil.loop(
                typ_2.shape,
                iters,
                [IR.Assn(expr_2_elt, IRUtil.sub(IRUtil.zero, expr_1_elt))],
            )
            comment = IR.Comment(str(node.metadata))
            prog_2 = IRUtil.prog_merge(prog_1, IR.Prog([comment] + cmdl_assn))
            prog_2 = IRUtil.prog_merge(
                IR.Prog([IR.Decl(expr_2.idf, node.type)]), prog_2
            )

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[expr_2.idf] = self.scaleFacMapping[expr_1.idf]

        return (prog_2, expr_2)

    def visitBOp(self, node: AST.BOp, args=None):
        op = node.op
        if op in [AST.Operators.ADD, AST.Operators.SUB, AST.Operators.Equal]:
            return self.visitBopAddOrSubLike(node)
        elif op in [AST.Operators.ElemWiseMul, AST.Operators.ElemWiseDiv]:
            return self.visitBopElemWiseOp(node)
        elif op == AST.Operators.MUL:
            return self.visitBopMul(node)
        elif op == AST.Operators.CONV:
            return self.visitBopConv(node)
        elif op == AST.Operators.CONVTRANSPOSE:
            return self.visitBopConvTranspose(node)
        else:
            assert False

    def visitBopAddOrSubLike(self, node: AST.BOp, args=None):
        (prog_1, expr_1) = self.visit(node.expr1)
        (prog_2, expr_2) = self.visit(node.expr2)

        op = node.op
        if op == AST.Operators.ADD:
            op_ir = IR.Op.Op["+"]
        elif op == AST.Operators.SUB:
            op_ir = IR.Op.Op["-"]
        elif op == AST.Operators.Equal:
            op_ir = IR.Op.Op["=="]
        else:
            assert False

        node_type = node.type
        out_arr = self.getTempVar()
        cmd0 = IR.Comment(expr_1.idf + " " + op_ir.name + " " + expr_2.idf)
        comment = IR.Comment(str(node.metadata))

        if not (Util.Config.disableTruncOpti):
            expr1_sf = self.scaleFacMapping[expr_1.idf]
            expr2_sf = self.scaleFacMapping[expr_2.idf]
            scaleUpFactor = -1
            if expr1_sf > expr2_sf:
                exprToScale = expr_2
                typeOfExprToScale = node.expr2.type
                scaleUpFactor = expr1_sf - expr2_sf
                self.scaleFacMapping[expr_2.idf] = expr1_sf
            elif expr2_sf > expr1_sf:
                exprToScale = expr_1
                typeOfExprToScale = node.expr1.type
                scaleUpFactor = expr2_sf - expr1_sf
                self.scaleFacMapping[expr_1.idf] = expr2_sf

            if scaleUpFactor != -1:
                comm = IR.Comment(
                    "Scale up of args needed was found while doing OptimizeTruncations."
                )
                argsDict = OrderedDict()
                curFuncName = "ScaleUp"
                if not (Type.isInt(typeOfExprToScale)):
                    outputShape = typeOfExprToScale.shape
                    for ii, curDimSize in enumerate(outputShape):
                        argsDict[IR.Int(curDimSize, 32)] = "size_" + str(ii)
                    curFuncName += str(len(outputShape))
                argsDict[exprToScale] = "exprToScale, arg#{0}".format(
                    2 if (expr1_sf > expr2_sf) else 1
                )
                argsDict[IR.Int(scaleUpFactor, 32)] = "ScaleUpFactor"
                funcCall = IR.FuncCall(curFuncName, argsDict)

                if Type.isInt(typeOfExprToScale) or typeOfExprToScale.shape == []:
                    assn_expr = IR.Assn(exprToScale, funcCall)
                    curProg = IR.Prog([comm, assn_expr])
                else:
                    curProg = IR.Prog([comm, funcCall])
                prog_1 = IRUtil.prog_merge(curProg, prog_1)

            self.scaleFacMapping[out_arr.idf] = self.scaleFacMapping[expr_1.idf]

        decl = IR.Decl(out_arr.idf, node_type, node_type.bitlen, node_type.isSecret)

        if Type.isInt(node_type):
            assign = IR.Assn(out_arr, IR.IntBop(expr_1, op_ir, expr_2))
            out_prog = IR.Prog([assign])
        else:
            outputShape = node_type.shape
            inp1_shape = [] if Type.isInt(node.expr1.type) else node.expr1.type.shape
            inp2_shape = [] if Type.isInt(node.expr2.type) else node.expr2.type.shape

            expected_output_shape, _, _ = Util.getBroadcastShapes(
                inp1_shape, inp2_shape
            )
            assert outputShape == expected_output_shape
            out_prog = IRUtil.generateBroadcastLoopBOp(
                expr_1, inp1_shape, expr_2, inp2_shape, out_arr, op_ir
            )

        out_prog = IRUtil.prog_merge(IR.Prog([comment, cmd0, decl]), out_prog)
        out_prog = IRUtil.prog_merge(prog_1, prog_2, out_prog)
        return (out_prog, out_arr)

    # We first reshape both inputs and flatten them into 1d dims.
    # For simplicity consider a non-broadcast example:
    # inputs : inp1_arr[s1][s2], inp2_arr[s1][s2]
    # after flattening : inp1_arr_flat[s1*s2], inp2_arr_flat[s1*s2]
    # for i1=[0:s1]
    #   for i2=[0:s2]
    #     idx = i1*s2 + i2
    #     inp1_arr_flat[idx] = inp1_arr[i1][i2]
    #     inp2_arr_flat[idx] = inp2_arr[i1][i2]
    # If one input is from server and the other from model we can call an optimized version of mul
    #   ElemWiseActModelVectorMult(s1*s2, inp1_arr_flat, inp2_arr_flat, out_arr_flat) <- optimized
    # OR
    #  ElemWiseSecretSharedVectorMult(s1*s2, inp1_arr_flat, inp2_arr_flat, out_arr_flat)
    # Finally we reshape the flattened output
    # for i1=[0:s1]
    #   for i2=[0:s2]
    #     idx = i1*s2 + i2
    #     out_arr[i1][i2] = out_arr_flat[idx]
    # Standard broadcast rules apply to generate these flattened tensors.
    def visitBopElemWiseOp(self, node: AST.BOp, args=None):
        (prog_1, expr_1) = self.visit(node.expr1)
        (prog_2, expr_2) = self.visit(node.expr2)

        if node.op == AST.Operators.ElemWiseMul:
            op_ir = IR.Op.Op[".*"]
            funcName = "ElemWiseMul"
        elif node.op == AST.Operators.ElemWiseDiv:
            op_ir = IR.Op.Op["./"]
            funcName = "ElemWiseDiv"
            assert False, "Did not implement div yet"
        else:
            assert False, "Non mul/div elemwise op"

        comment = IR.Comment(str(node.metadata))
        cmd0 = IR.Comment(expr_1.idf + " " + op_ir.name + " " + expr_2.idf)

        node_type = node.type
        # outArr[s1][s2]
        out_arr = self.getTempVar()
        decl_out_arr = IR.Decl(
            out_arr.idf, node_type, node_type.bitlen, node_type.isSecret
        )

        if Type.isInt(node_type):
            assign = IR.Assn(out_arr, IR.IntBop(expr_1, op_ir, expr_2))
            out_prog = IR.Prog([assign])
        else:
            # Flattening inputs
            output_shape = node_type.shape
            inp1_shape = [] if Type.isInt(node.expr1.type) else node.expr1.type.shape
            inp2_shape = [] if Type.isInt(node.expr2.type) else node.expr2.type.shape
            out_iters = self.getTempIterators(len(output_shape))
            (
                expected_output_shape,
                broadcast_mask_1,
                broadcast_mask_2,
            ) = Util.getBroadcastShapes(inp1_shape, inp2_shape)
            assert expected_output_shape == output_shape

            # inp1_arr[i1][i2], inp2_arr[i1][i2], out_arr[i1][i2]
            inp1_iters = IRUtil.getMaskedIters(broadcast_mask_1, out_iters, inp1_shape)
            inp2_iters = IRUtil.getMaskedIters(broadcast_mask_2, out_iters, inp2_shape)
            inp1_arr_expr = IRUtil.addIndex(expr_1, inp1_iters)
            inp2_arr_expr = IRUtil.addIndex(expr_2, inp2_iters)
            out_arr_expr = IRUtil.addIndex(out_arr, out_iters)

            flat_size = Util.get_volume(output_shape)
            inp1_arr_flat = self.getTempVar()
            inp2_arr_flat = self.getTempVar()
            out_arr_flat = self.getTempVar()
            flat_type = Type.Tensor(
                [flat_size],
                node.expr1.type.bitlen,
                node.expr1.type.isSecret,
                node.expr1.type.taint,
            )
            # inp1_arr_flat[s1*s2]
            # inp2_arr_flat[s1*s2]
            # out_arr_flat[s1*s2]
            decl_inp1_arr_flat = IR.Decl(
                inp1_arr_flat.idf,
                flat_type,
                node.expr1.type.bitlen,
                node.expr1.type.isSecret,
            )
            decl_inp2_arr_flat = IR.Decl(
                inp2_arr_flat.idf,
                flat_type,
                node.expr2.type.bitlen,
                node.expr2.type.isSecret,
            )
            decl_out_arr_flat = IR.Decl(
                out_arr_flat.idf, flat_type, node.type.bitlen, node.type.isSecret
            )
            # idx
            flat_idx = self.getTempVar()
            decl_flat_idx = IR.Decl(
                flat_idx.idf, Type.Int(bitlen=32), bitlen=32, isSecret=False
            )
            # For 4d, generate (i1*s2*s3*s4) + (i2*s3*s4) + (i3*s4) + (i4);
            flat_idx_expr = IR.Int(0, 32)
            for i in range(len(out_iters)):
                vol = Util.get_volume(output_shape[i + 1 :])
                flat_idx_expr = IRUtil.add(
                    flat_idx_expr, IRUtil.mul(out_iters[i], IR.Int(vol, 32))
                )
            # inp1_arr_flat[idx], inp2_arr_flat[idx], out_arr_flat[idx]
            inp1_arr_flat_expr = IRUtil.addIndex(inp1_arr_flat, [flat_idx])
            inp2_arr_flat_expr = IRUtil.addIndex(inp2_arr_flat, [flat_idx])
            out_arr_flat_expr = IRUtil.addIndex(out_arr_flat, [flat_idx])
            # idx = i1*s2 + i2;
            # inp1_arr_flat[idx] = inp1_arr[i1][i2]
            # inp2_arr_flat[idx] = inp2_arr[i1][i2]
            assign_flat_idx_expr = IR.Assn(flat_idx, flat_idx_expr)
            assign_inp1_arr_flat = IR.Assn(inp1_arr_flat_expr, inp1_arr_expr)
            assign_inp2_arr_flat = IR.Assn(inp2_arr_flat_expr, inp2_arr_expr)
            # Flattening loop
            # for i1=[0:s1]
            #   for i2=[0:s2]
            #     idx = i1*s2 + i2
            #     inp1_arr_flat[idx] = inp1_arr[i1][i2]
            #     inp2_arr_flat[idx] = inp2_arr[i1][i2]
            out_loop = IRUtil.loop(
                output_shape,
                out_iters,
                [assign_flat_idx_expr, assign_inp1_arr_flat, assign_inp2_arr_flat],
            )
            out_prog = IRUtil.Prog(out_loop)
            decls = [
                decl_out_arr,
                decl_inp1_arr_flat,
                decl_inp2_arr_flat,
                decl_out_arr_flat,
                decl_flat_idx,
            ]
            out_prog = IRUtil.prog_merge(IRUtil.Prog(decls), out_prog)

            # Insert call to mul/div functionality
            argsDict = OrderedDict()
            argsDict[IR.Int(flat_size, 32)] = "input_shape"
            if node.op == AST.Operators.ElemWiseDiv:
                argsDict[inp1_arr_flat] = "A"
                argsDict[inp2_arr_flat] = "B"
                funcName = "ElemwiseSuperDuperSecretDiv"
                assert False, "Elemwise div not implemented"
            else:
                # If either input is a model weight we can use an optimised version for mul
                # Otherwise if both are derived from client input we use the hadmaard version
                isMulOptimised = False
                if not (self.isModel(node.expr1)) and not (self.isModel(node.expr2)):
                    argsDict[inp1_arr_flat] = "A"
                    argsDict[inp2_arr_flat] = "B"
                else:
                    isMulOptimised = True
                    # Optimised version expects the second parameter to be an input from server
                    if self.isModel(node.expr2):
                        argsDict[inp1_arr_flat] = "A"
                        argsDict[inp2_arr_flat] = "B"
                    else:
                        # Shuffle the params.
                        argsDict[inp2_arr_flat] = "A"
                        argsDict[inp1_arr_flat] = "B"
                funcName = (
                    "ElemWiseActModelVectorMult"
                    if isMulOptimised
                    else "ElemWiseSecretSharedVectorMult"
                )
            argsDict[out_arr_flat] = "Output"
            funcCall = IR.FuncCall(funcName, argsDict)
            out_prog = IRUtil.prog_merge(out_prog, IRUtil.Prog([funcCall]))

            # Clear temp arrays
            argsDict = OrderedDict()
            argsDict[IR.Int(flat_size, 32)] = "size"
            argsDict[inp1_arr_flat] = "A"
            funcCall = IR.FuncCall("ClearMemSecret1", argsDict)
            out_prog = IRUtil.prog_merge(out_prog, IRUtil.Prog([funcCall]))
            argsDict = OrderedDict()
            argsDict[IR.Int(flat_size, 32)] = "size"
            argsDict[inp2_arr_flat] = "A"
            funcCall = IR.FuncCall("ClearMemSecret1", argsDict)
            out_prog = IRUtil.prog_merge(out_prog, IRUtil.Prog([funcCall]))

            # Unflatten output
            assign_out_arr_flat = IR.Assn(out_arr_expr, out_arr_flat_expr)
            out_loop = IRUtil.loop(
                output_shape, out_iters, [assign_flat_idx_expr, assign_out_arr_flat]
            )
            out_prog = IRUtil.prog_merge(out_prog, IRUtil.Prog(out_loop))

            argsDict = OrderedDict()
            argsDict[IR.Int(flat_size, 32)] = "size"
            argsDict[out_arr_flat] = "A"
            funcCall = IR.FuncCall("ClearMemSecret1", argsDict)
            out_prog = IRUtil.prog_merge(out_prog, IRUtil.Prog([funcCall]))

        progExtraBefore = IR.Prog([])
        progExtraAfter = IR.Prog([])
        if Util.Config.disableTruncOpti:
            progExtraAfter = self.addTruncateFunctionCall(
                node, "ElemWiseMul", out_arr, Util.Config.consSF
            )
        else:
            inputs_same = expr_1.idf == expr_2.idf
            expr1_sf = self.scaleFacMapping[expr_1.idf]
            expr2_sf = self.scaleFacMapping[expr_2.idf]
            if expr1_sf > self.scaleFac:
                progExtraBefore = self.addTruncateFunctionCall(
                    node.expr1, "ElemWiseMul", expr_1, expr1_sf - self.scaleFac
                )
                self.scaleFacMapping[expr_1.idf] = self.scaleFac
            if (not inputs_same) and (expr2_sf > self.scaleFac):
                progExtraBefore = IRUtil.prog_merge(
                    progExtraBefore,
                    self.addTruncateFunctionCall(
                        node.expr2, "ElemWiseMul", expr_2, expr2_sf - self.scaleFac
                    ),
                )
                self.scaleFacMapping[expr_2.idf] = self.scaleFac
            self.scaleFacMapping[out_arr.idf] = 2 * self.scaleFac

        out_prog = IRUtil.prog_merge(
            IRUtil.Prog([comment, cmd0]), progExtraBefore, out_prog, progExtraAfter
        )
        return (out_prog, out_arr)

    def visitBopMul(self, node: AST.BOp, args=None):
        typ_1 = node.expr1.type
        typ_2 = node.expr2.type
        typ_3 = node.type
        if Type.isInt(typ_3):
            return self.visitBopMulInt(node)
        elif typ_1.dim == 0 or Type.isInt(typ_1) or typ_2.dim == 0 or Type.isInt(typ_2):
            return self.visitBopMulScalar1DTensor(node)
        else:
            return self.visitBopMul2DTensor(node)

    def visitBopMulInt(self, node: AST.BOp, args=None):
        (prog_1, expr_1) = self.visit(node.expr1)
        (prog_2, expr_2) = self.visit(node.expr2)

        expr_3 = self.getTempVar()
        comment = IR.Comment(str(node.metadata))
        bitlen = node.expr.bitlen
        decl = IR.Decl(expr_3.idf, node.type, node.type.bitlen, node.type.isSecret)
        assign = IR.Assn(expr_3, IRUtil.mul(expr_1, expr_2))
        prog_3 = IRUtil.prog_merge(prog_1, prog_2, IR.Prog([comment, decl, assign]))

        progExtraBefore = IR.Prog([])
        progExtraAfter = IR.Prog([])
        if Util.Config.disableTruncOpti:
            progExtraAfter = self.addTruncateFunctionCall(
                node, "MulInt", expr_3, Util.Config.consSF
            )
        else:
            inputs_same = expr_1.idf == expr_2.idf
            expr1_sf = self.scaleFacMapping[expr_1.idf]
            expr2_sf = self.scaleFacMapping[expr_2.idf]
            if expr1_sf > self.scaleFac:
                progExtraBefore = self.addTruncateFunctionCall(
                    node.expr1, "MulInt", expr_1, expr1_sf - self.scaleFac
                )
                self.scaleFacMapping[expr_1.idf] = self.scaleFac
            if (not inputs_same) and (expr2_sf > self.scaleFac):
                progExtraBefore = IRUtil.prog_merge(
                    progExtraBefore,
                    self.addTruncateFunctionCall(
                        node.expr2, "MulInt", expr_2, expr2_sf - self.scaleFac
                    ),
                )
                self.scaleFacMapping[expr_2.idf] = self.scaleFac
            self.scaleFacMapping[expr_3.idf] = 2 * self.scaleFac

        prog_3 = IRUtil.prog_merge(progExtraBefore, prog_3, progExtraAfter)
        return (prog_3, expr_3)

    def visitBopMulScalar1DTensor(self, node: AST.BOp, args=None):
        (prog_1, expr_1) = self.visit(node.expr1)
        (prog_2, expr_2) = self.visit(node.expr2)

        """
        A 100
        B 100

        int64_al[100] A ;
        int64_al[100] B ;
        """

        typ_1 = node.expr1.type
        typ_2 = node.expr2.type
        typ_3 = node.type

        isIntMult = False
        if typ_1.dim == 0 or Type.isInt(typ_1):
            a, b = expr_1, expr_2
            outputShape = typ_2.shape
            isIntMult = Type.isInt(typ_1)
        else:
            a, b = expr_2, expr_1
            outputShape = typ_1.shape
            isIntMult = Type.isInt(typ_2)

        # decl fresh vars
        expr_3 = self.getTempVar()
        cmd0 = IR.Comment(expr_1.idf + " * " + expr_2.idf)
        funcCallArgsDict = OrderedDict()
        for ii, curDimSize in enumerate(outputShape):
            funcCallArgsDict[IR.Int(curDimSize, 32)] = "size_" + str(ii)
        funcCallArgsDict[a] = "A"
        funcCallArgsDict[b] = "B"
        funcCallArgsDict[expr_3] = "C"
        progExtraBefore = IR.Prog([])
        progExtraAfter = IR.Prog([])
        if Util.Config.disableTruncOpti:
            progExtraAfter = self.addTruncateFunctionCall(
                node, "ScalarMul", expr_3, Util.Config.consSF
            )
        else:
            inputs_same = expr_1.idf == expr_2.idf
            expr1_sf = self.scaleFacMapping[expr_1.idf]
            expr2_sf = self.scaleFacMapping[expr_2.idf]
            if expr1_sf > self.scaleFac:
                progExtraBefore = self.addTruncateFunctionCall(
                    node.expr1, "ScalarMul", expr_1, expr1_sf - self.scaleFac
                )
                self.scaleFacMapping[expr_1.idf] = self.scaleFac
            if (not inputs_same) and (expr2_sf > self.scaleFac):
                progExtraBefore = IRUtil.prog_merge(
                    progExtraBefore,
                    self.addTruncateFunctionCall(
                        node.expr2, "ScalarMul", expr_2, expr2_sf - self.scaleFac
                    ),
                )
                self.scaleFacMapping[expr_2.idf] = self.scaleFac
            self.scaleFacMapping[expr_3.idf] = 2 * self.scaleFac

        funcCall = IR.FuncCall(
            "ScalarMul" + self.varNameDelim + str(len(outputShape)), funcCallArgsDict
        )
        prog_3 = IRUtil.prog_merge(
            prog_1, prog_2, progExtraBefore, IR.Prog([cmd0, funcCall])
        )
        prog_3 = IRUtil.prog_merge(
            IR.Prog([IR.Decl(expr_3.idf, node.type)]), prog_3, progExtraAfter
        )
        return (prog_3, expr_3)

    def visitBopMul2DTensor(self, node: AST.BOp, args=None):
        (prog_1, expr_1) = self.visit(node.expr1)
        (prog_2, expr_2) = self.visit(node.expr2)

        # decl fresh vars
        expr_3 = self.getTempVar()

        typ_1 = node.expr1.type
        typ_2 = node.expr2.type
        typ_3 = node.type

        [I, J] = typ_1.shape
        [J, K] = typ_2.shape
        typ_mul = Type.Tensor([J])

        shrT = Util.Config.consSF

        cmd0 = IR.Comment(expr_1.idf + " * " + expr_2.idf)
        funcCallArgsDict = OrderedDict()
        funcCallArgsDict[IR.Int(I, 32)] = "I"
        funcCallArgsDict[IR.Int(J, 32)] = "J"
        funcCallArgsDict[IR.Int(K, 32)] = "K"
        funcCallArgsDict[expr_1] = "A"
        funcCallArgsDict[expr_2] = "B"
        funcCallArgsDict[expr_3] = "C"

        # Add an arg as to which arg out of A or B is a model weight
        # This is ok, since Athos is right now tailored for neural network inference
        # and in inference, in every linear layer, either of A or B will be a model weight.
        # This is required because for some backends, knowing which of A or B is a model weight
        # can make a difference in their performance.

        assert self.isModel(node.expr1) or self.isModel(
            node.expr2
        ), "Expecting one of A or B to be an input by the server (model weight)."
        modelIsA = True
        if not self.isModel(node.expr1):
            modelIsA = False
        funcCallArgsDict[IR.Bool(modelIsA)] = "modelIsA"

        progExtraBefore = IR.Prog([])
        progExtraAfter = IR.Prog([])
        if Util.Config.disableTruncOpti:
            progExtraAfter = self.addTruncateFunctionCall(
                node, "MatMul2D", expr_3, Util.Config.consSF
            )
        else:
            inputs_same = expr_1.idf == expr_2.idf
            expr1_sf = self.scaleFacMapping[expr_1.idf]
            expr2_sf = self.scaleFacMapping[expr_2.idf]
            if expr1_sf > self.scaleFac:
                progExtraBefore = self.addTruncateFunctionCall(
                    node.expr1, "MatMul2D", expr_1, expr1_sf - self.scaleFac
                )
                self.scaleFacMapping[expr_1.idf] = self.scaleFac
            if (not inputs_same) and (expr2_sf > self.scaleFac):
                progExtraBefore = IRUtil.prog_merge(
                    progExtraBefore,
                    self.addTruncateFunctionCall(
                        node.expr2, "MatMul2D", expr_2, expr2_sf - self.scaleFac
                    ),
                )
                self.scaleFacMapping[expr_2.idf] = self.scaleFac
            self.scaleFacMapping[expr_3.idf] = 2 * self.scaleFac

        funcCall = IR.FuncCall("MatMul2D", funcCallArgsDict)
        comment = IR.Comment(str(node.metadata))
        prog_3 = IRUtil.prog_merge(
            prog_1, prog_2, progExtraBefore, IR.Prog([comment, cmd0, funcCall])
        )
        prog_3 = IRUtil.prog_merge(
            IR.Prog([IR.Decl(expr_3.idf, node.type)]), prog_3, progExtraAfter
        )

        return (prog_3, expr_3)

    def visitBopConv(self, node: AST.BOp, args=None):
        (prog1, expr_1) = self.visit(node.expr1)
        (prog2, expr_2) = self.visit(node.expr2)

        convDim = 2
        if AST.PaddingKeysDict.ConvDim in node.options:
            convDim = node.options[AST.PaddingKeysDict.ConvDim]

        if convDim == 2:
            [N, H, W, CI] = node.expr1.type.shape
            [FH, FW, CI1, CO] = node.expr2.type.shape
        elif convDim == 3:
            [N, D, H, W, CI] = node.expr1.type.shape
            [FD, FH, FW, CI1, CO] = node.expr2.type.shape
        else:
            assert False

        returnExpr = self.getTempVar()
        comment = IR.Comment(
            expr_1.idf + " # " + expr_2.idf + ", convDim = " + str(convDim)
        )
        funcCallArgsDict = OrderedDict()
        funcCallArgsDict[IR.Int(N, 32)] = "N"
        if convDim == 3:
            funcCallArgsDict[IR.Int(D, 32)] = "D"
        funcCallArgsDict[IR.Int(H, 32)] = "H"
        funcCallArgsDict[IR.Int(W, 32)] = "W"
        funcCallArgsDict[IR.Int(CI, 32)] = "CI"
        if convDim == 3:
            funcCallArgsDict[IR.Int(FD, 32)] = "FD"
        funcCallArgsDict[IR.Int(FH, 32)] = "FH"
        funcCallArgsDict[IR.Int(FW, 32)] = "FW"
        funcCallArgsDict[IR.Int(CO, 32)] = "CO"
        if convDim == 3:
            funcCallArgsDict[
                IR.Int(node.options[AST.PaddingKeysDict.zPadDLeft], 32)
            ] = "zPadDLeft"
            funcCallArgsDict[
                IR.Int(node.options[AST.PaddingKeysDict.zPadDRight], 32)
            ] = "zPadDRight"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.zPadHLeft], 32)
        ] = "zPadHLeft"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.zPadHRight], 32)
        ] = "zPadHRight"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.zPadWLeft], 32)
        ] = "zPadWLeft"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.zPadWRight], 32)
        ] = "zPadWRight"
        if convDim == 3:
            funcCallArgsDict[
                IR.Int(node.options[AST.PaddingKeysDict.strideD], 32)
            ] = "strideD"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.strideH], 32)
        ] = "strideH"
        funcCallArgsDict[
            IR.Int(node.options[AST.PaddingKeysDict.strideW], 32)
        ] = "strideW"

        isGroupConv = False
        if AST.PaddingKeysDict.group in node.options.keys():
            funcCallArgsDict[IR.Int(node.options[AST.PaddingKeysDict.group], 32)] = "G"
            isGroupConv = True

        funcCallArgsDict[expr_1] = "input"
        funcCallArgsDict[expr_2] = "filter"
        if convDim == 3:
            funcCallArgsDict[IR.Int(Util.Config.consSF, 32)] = "consSF"
        funcCallArgsDict[returnExpr] = "output"

        if convDim == 2:
            funcCallName = "Conv2D"
        else:
            funcCallName = "Conv3D"

        if isGroupConv:
            funcCallName += "Group"

        funcCallName += "Wrapper"

        funcCall = IR.FuncCall(funcCallName, funcCallArgsDict)
        progConv = IR.Prog([comment, funcCall])

        progExtraBefore = IR.Prog([])
        progExtraAfter = IR.Prog([])
        if Util.Config.disableTruncOpti:
            progExtraAfter = self.addTruncateFunctionCall(
                node, "Conv", returnExpr, Util.Config.consSF
            )
        else:
            inputs_same = expr_1.idf == expr_2.idf
            expr1_sf = self.scaleFacMapping[expr_1.idf]
            expr2_sf = self.scaleFacMapping[expr_2.idf]
            if expr1_sf > self.scaleFac:
                progExtraBefore = self.addTruncateFunctionCall(
                    node.expr1, "Conv", expr_1, expr1_sf - self.scaleFac
                )
                self.scaleFacMapping[expr_1.idf] = self.scaleFac
            if (not inputs_same) and (expr2_sf > self.scaleFac):
                progExtraBefore = IRUtil.prog_merge(
                    progExtraBefore,
                    self.addTruncateFunctionCall(
                        node.expr2, "Conv", expr_2, expr2_sf - self.scaleFac
                    ),
                )
                self.scaleFacMapping[expr_2.idf] = self.scaleFac
            self.scaleFacMapping[returnExpr.idf] = 2 * self.scaleFac

        returnProg = IRUtil.prog_merge(prog1, prog2, progExtraBefore, progConv)
        returnProg = IRUtil.prog_merge(
            IR.Prog([IR.Decl(returnExpr.idf, node.type)]), returnProg, progExtraAfter
        )
        return (returnProg, returnExpr)

    def visitBopConvTranspose(self, node: AST.BOp, args=None):
        (prog1, expr_1) = self.visit(node.expr1)
        (prog2, expr_2) = self.visit(node.expr2)

        convDim = 2
        if AST.PaddingKeysDict.ConvDim in node.options:
            convDim = node.options[AST.PaddingKeysDict.ConvDim]

        if convDim == 2:
            [N, H_prime, W_prime, CI1] = node.expr1.type.shape
            [FH, FW, CO, CI] = node.expr2.type.shape
        elif convDim == 3:
            [N, D_prime, H_prime, W_prime, CI1] = node.expr1.type.shape
            [FD, FH, FW, CO, CI] = node.expr2.type.shape
        else:
            assert False
        assert CI1 == CI

        H = node.options[AST.PaddingKeysDict.outputImgH]  # outputH
        W = node.options[AST.PaddingKeysDict.outputImgW]  # outputW
        pad_h_total = (
            node.options[AST.PaddingKeysDict.zPadHLeft]
            + node.options[AST.PaddingKeysDict.zPadHRight]
        )
        pad_w_total = (
            node.options[AST.PaddingKeysDict.zPadWLeft]
            + node.options[AST.PaddingKeysDict.zPadWRight]
        )
        strideH = node.options[AST.PaddingKeysDict.strideH]
        strideW = node.options[AST.PaddingKeysDict.strideW]
        [
            pad_h_tr_total,
            stride_h_tr,
            h_prime_tilde,
        ] = AST.Operators.findConvTransposePadding(H, H_prime, FH, pad_h_total, strideH)
        [
            pad_w_tr_total,
            stride_w_tr,
            w_prime_tilde,
        ] = AST.Operators.findConvTransposePadding(W, W_prime, FW, pad_w_total, strideW)

        [
            pad_h_tr_left,
            pad_h_tr_right,
        ] = AST.Operators.findLeftRightPaddingFromTotalPadding(pad_h_tr_total)
        [
            pad_w_tr_left,
            pad_w_tr_right,
        ] = AST.Operators.findLeftRightPaddingFromTotalPadding(pad_w_tr_total)

        assert (
            AST.Operators.findConvOutputImgSize(
                h_prime_tilde, pad_h_tr_total, FH, stride_h_tr
            )
            == H
        )
        assert (
            AST.Operators.findConvOutputImgSize(
                w_prime_tilde, pad_w_tr_total, FW, stride_w_tr
            )
            == W
        )

        if convDim == 3:
            D = node.options[AST.PaddingKeysDict.outputImgD]  # outputD
            pad_d_total = (
                node.options[AST.PaddingKeysDict.zPadDLeft]
                + node.options[AST.PaddingKeysDict.zPadDRight]
            )
            strideD = node.options[AST.PaddingKeysDict.strideD]
            [
                pad_d_tr_total,
                stride_d_tr,
                d_prime_tilde,
            ] = AST.Operators.findConvTransposePadding(
                D, D_prime, FD, pad_d_total, strideD
            )
            [
                pad_d_tr_left,
                pad_d_tr_right,
            ] = AST.Operators.findLeftRightPaddingFromTotalPadding(pad_d_tr_total)
            assert (
                AST.Operators.findConvOutputImgSize(
                    d_prime_tilde, pad_d_tr_total, FD, stride_d_tr
                )
                == D
            )

        returnExpr = self.getTempVar()
        comment = IR.Comment(
            expr_1.idf + " #T " + expr_2.idf + ", convDim = " + str(convDim)
        )
        funcCallArgsDict = OrderedDict()
        funcCallArgsDict[IR.Int(N, 32)] = "N"
        if convDim == 3:
            funcCallArgsDict[IR.Int(D_prime, 32)] = "D_prime"
        funcCallArgsDict[IR.Int(H_prime, 32)] = "H_prime"
        funcCallArgsDict[IR.Int(W_prime, 32)] = "W_prime"
        funcCallArgsDict[IR.Int(CI, 32)] = "CI"
        if convDim == 3:
            funcCallArgsDict[IR.Int(FD, 32)] = "FD"
        funcCallArgsDict[IR.Int(FH, 32)] = "FH"
        funcCallArgsDict[IR.Int(FW, 32)] = "FW"
        funcCallArgsDict[IR.Int(CO, 32)] = "CO"
        if convDim == 3:
            funcCallArgsDict[IR.Int(D, 32)] = "D"
        funcCallArgsDict[IR.Int(H, 32)] = "H"
        funcCallArgsDict[IR.Int(W, 32)] = "W"
        if convDim == 3:
            funcCallArgsDict[IR.Int(pad_d_tr_left, 32)] = "pad_d_tr_left"
            funcCallArgsDict[IR.Int(pad_d_tr_right, 32)] = "pad_d_tr_right"
        funcCallArgsDict[IR.Int(pad_h_tr_left, 32)] = "pad_h_tr_left"
        funcCallArgsDict[IR.Int(pad_h_tr_right, 32)] = "pad_h_tr_right"
        funcCallArgsDict[IR.Int(pad_w_tr_left, 32)] = "pad_w_tr_left"
        funcCallArgsDict[IR.Int(pad_w_tr_right, 32)] = "pad_w_tr_right"
        if convDim == 3:
            funcCallArgsDict[IR.Int(strideD, 32)] = "strideD"
        funcCallArgsDict[IR.Int(strideH, 32)] = "strideH"
        funcCallArgsDict[IR.Int(strideW, 32)] = "strideW"

        funcCallArgsDict[expr_1] = "input"
        funcCallArgsDict[expr_2] = "filter"
        if convDim == 3:
            funcCallArgsDict[IR.Int(Util.Config.consSF, 32)] = "consSF"
        funcCallArgsDict[returnExpr] = "output"

        if convDim == 2:
            funcCallName = "ConvTranspose2D"
        else:
            funcCallName = "ConvTranspose3D"
        funcCallName += "Wrapper"
        funcCall = IR.FuncCall(funcCallName, funcCallArgsDict)

        progConv = IR.Prog([comment, funcCall])

        progExtraBefore = IR.Prog([])
        progExtraAfter = IR.Prog([])
        if Util.Config.disableTruncOpti:
            progExtraAfter = self.addTruncateFunctionCall(
                node, "ConvTranspose", returnExpr, self.scaleFac
            )
        else:
            inputs_same = expr_1.idf == expr_2.idf
            expr1_sf = self.scaleFacMapping[expr_1.idf]
            expr2_sf = self.scaleFacMapping[expr_2.idf]
            if expr1_sf > self.scaleFac:
                progExtraBefore = self.addTruncateFunctionCall(
                    node.expr1, "ConvTranspose", expr_1, expr1_sf - self.scaleFac
                )
                self.scaleFacMapping[expr_1.idf] = self.scaleFac
            if (not inputs_same) and (expr2_sf > self.scaleFac):
                progExtraBefore = IRUtil.prog_merge(
                    progExtraBefore,
                    self.addTruncateFunctionCall(
                        node.expr2, "ConvTranspose", expr_2, expr2_sf - self.scaleFac
                    ),
                )
                self.scaleFacMapping[expr2.idf] = self.scaleFac
            self.scaleFacMapping[returnExpr.idf] = 2 * self.scaleFac

        returnProg = IRUtil.prog_merge(prog1, prog2, progExtraBefore, progConv)
        returnProg = IRUtil.prog_merge(
            IR.Prog([IR.Decl(returnExpr.idf, node.type)]), returnProg, progExtraAfter
        )
        return (returnProg, returnExpr)

    def visitFunc(self, node: AST.Func, args=None):
        op = node.op
        assert op in [
            AST.Operators.Floor,
            AST.Operators.Shape,
            AST.Operators.RELU,
            AST.Operators.TANH,
            AST.Operators.SIGMOID,
            AST.Operators.HARDSIGMOID,
            AST.Operators.SQRT,
            AST.Operators.RSQRT,
            AST.Operators.ClearMemSecret,
            AST.Operators.ClearMemPublic,
        ]
        return self.visitFloorLike(node)

    def visitFloorLike(self, node: AST.Func, args=None):
        (prog1, expr1) = self.visit(node.expr)
        out_expr = self.getTempVar()

        if node.op == AST.Operators.Floor:
            funcName = "Floor"
        elif node.op == AST.Operators.Shape:
            funcName = "Shape"
        elif node.op == AST.Operators.RELU:
            funcName = "Relu"
        elif node.op == AST.Operators.TANH:
            funcName = "Tanh"
        elif node.op == AST.Operators.SIGMOID:
            funcName = "Sigmoid"
        elif node.op == AST.Operators.HARDSIGMOID:
            funcName = "HardSigmoid"
        elif node.op == AST.Operators.SQRT:
            funcName = "Sqrt"
        elif node.op == AST.Operators.RSQRT:
            funcName = "Sqrt"
        elif node.op == AST.Operators.ClearMemSecret:
            funcName = "ClearMemSecret"
        elif node.op == AST.Operators.ClearMemPublic:
            funcName = "ClearMemPublic"
        else:
            assert False

        # We don't need to clear scalars.
        if (
            node.op == AST.Operators.ClearMemSecret
            or node.op == AST.Operators.ClearMemPublic
        ):
            if Type.isInt(node.expr.type):
                return (prog1, expr1)
            if node.expr.type.dim == 0:
                return (prog1, expr1)

        argsList = OrderedDict()

        inputType = node.expr.type
        if Type.isTensor(inputType):
            for ii, curDim in enumerate(inputType.shape):
                argsList[IR.Int(curDim, 32)] = "inShape_" + str(ii)

        argsList[expr1] = "inArr"

        if Type.isTensor(node.type):
            argsList[out_expr] = "outArr"

        if node.op == AST.Operators.Floor:
            argsList[IR.Int(Util.Config.consSF, 32)] = "curScale"

        progExtraBefore = IR.Prog([])
        if Util.Config.disableTruncOpti:
            if node.op in [
                AST.Operators.RELU,
                AST.Operators.HARDSIGMOID,
            ]:
                argsList[IR.Int(Util.Config.consSF, 32)] = "consSF"
                argsList[IR.Bool(False)] = "doTruncation"
            if node.op in [
                AST.Operators.TANH,
                AST.Operators.SIGMOID,
                AST.Operators.SQRT,
                AST.Operators.RSQRT,
            ]:
                assert (
                    self.scaleFac > 31
                ), "The program scaling factor {} is invalid. Should be lesser than 32 if network has tan/sig/sqrt as those only support 32 bitlengths".format(
                    self.scaleFac
                )
                argsList[IR.Int(self.scaleFac, 32)] = "sA"
                argsList[IR.Int(self.scaleFac, 32)] = "sB"
        else:
            final_sf = self.scaleFacMapping[expr1.idf]
            if node.op in [
                AST.Operators.RELU,
                AST.Operators.HARDSIGMOID,
            ]:
                argsList[IR.Int(final_sf - self.scaleFac, 32)] = "consSF"
                if final_sf > self.scaleFac:
                    # If it can't tolerate one more mult operation, then scale down here
                    assert final_sf - self.scaleFac == self.scaleFac
                    final_sf = self.scaleFac
                    argsList[IR.Bool(True)] = "doTruncation"
                else:
                    argsList[IR.Bool(False)] = "doTruncation"
            if node.op in [
                AST.Operators.TANH,
                AST.Operators.SIGMOID,
                AST.Operators.SQRT,
                AST.Operators.RSQRT,
            ]:
                # Since these class of functions can only handle input of 32 bitlength, we have to scale down
                # inputs before calling them.
                # 32 bit fixedpt |(+/-)| _ _ _ _ _ _ _ _ | _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ |
                # 31 bits available. For small values with max precision we can do 0.31 split
                # Consider fp number: 2^-25 (2.98023223876953125 × 10^-8)
                # | 1 | 1 1 0 0 1 1 0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 |
                #   = 1 * 1.0 *  2^(102 - 127) = 2^-25
                # If we scale by mantissa number of bits i.e. 23, we get a fixedpt val
                #   = floor(1 * (2 ^ (-25 + 23))) = floor(2^-2) = floor(0.025) = 0
                # We have lost the precision. It is possible to scale more than mantissa bits to get precision.
                # if we had scaled by 25 bits, we wouldve got fixedpt val = 1 and we could then do a 6.25 split while
                # storing it. Upper bound of scaling is not mantissa, it is fixedpt bitlen available.
                if final_sf > 31:
                    assert (
                        final_sf > self.scaleFac
                    ), "The program scaling factor {} is invalid. Should be lesser than 32 if network has tan/sig/sqrt as those only support 32 bitlengths".format(
                        self.scaleFac
                    )
                    assert final_sf - self.scaleFac == self.scaleFac
                    progExtraBefore = IRUtil.prog_merge(
                        progExtraBefore,
                        self.addTruncateFunctionCall(
                            node.expr, node.op.name, expr1, final_sf - self.scaleFac
                        ),
                    )
                    self.scaleFacMapping[expr1.idf] = self.scaleFac
                    final_sf = self.scaleFac
                argsList[IR.Int(final_sf, 32)] = "sA"
                argsList[IR.Int(final_sf, 32)] = "sB"
            self.scaleFacMapping[out_expr.idf] = final_sf

        # Tanh/Sigmoid/Sqrt impl only supports upto 32 bitwidth for input
        if node.op in [
            AST.Operators.TANH,
            AST.Operators.SIGMOID,
            AST.Operators.SQRT,
            AST.Operators.RSQRT,
        ]:
            argsList[IR.Int(min(32, self.actualbitwidth), 32)] = "bwA"
            # argsList[IR.Int(min(32, self.actualbitwidth), 32)] = "bwB"
            argsList[IR.Int(self.actualbitwidth, 32)] = "bwB"
            if node.op == AST.Operators.SQRT:
                argsList[IR.Bool(False)] = "inverse"
            if node.op == AST.Operators.RSQRT:
                argsList[IR.Bool(True)] = "inverse"
            argsList[IR.Int(8, 32)] = "LUTBITS"

        comment = IR.Comment(str(node.metadata))
        funcNameSuffix = ""
        if Type.isTensor(inputType):
            funcNameSuffix = str(len(inputType.shape))

        progFinal = IR.Prog(
            [
                comment,
                IR.FuncCall(funcName + self.varNameDelim + funcNameSuffix, argsList),
            ]
        )
        if Type.isTensor(node.type):
            progFinal = IRUtil.prog_merge(
                IR.Prog([IR.Decl(out_expr.idf, node.type)]), progFinal
            )

        progFinal = IRUtil.prog_merge(prog1, progExtraBefore, progFinal)
        return (progFinal, out_expr)

    def visitLet(self, node: AST.Let, args=None):
        (prog_1, expr_1) = self.visit(node.decl)
        typ_1 = node.decl.type
        idf = node.name.name
        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[idf] = self.scaleFacMapping[expr_1.idf]
        (prog_2, expr_2) = self.visit(node.expr)

        self.name_mapping[idf] = expr_1.idf
        self.expr_mapping[idf] = expr_1
        prog_3 = IRUtil.prog_merge(prog_1, prog_2)
        return (prog_3, expr_2)

    def visitUninterpFuncCall(self, node: AST.UninterpFuncCall, args=None):
        progList = []
        exprList = []
        for ii, curArg in enumerate(node.argsList):
            (progN, exprN) = self.visit(curArg)
            progList.append(progN)
            exprList.append(exprN)

        returnExpr = self.getTempVar()

        funcName = node.funcName
        funcName += self.varNameDelim + str(len(node.outputShape))
        for ii, curArg in enumerate(node.argsList):
            if Type.isTensor(curArg.type):
                curShape = curArg.type.shape

                # If len(shape) == 0 : that means its a float - no need to qualify
                #   the function name with 0 in that case, since its essentially
                #   become an int.
                if len(curShape) > 0:
                    funcName += self.varNameDelim + str(len(curShape))
                ### TODO : right now if random strings like int are passed, its being set as datatype int -- int datatype in
                #          unintrepreted func call is being used in a hacky way right now

        # Policy :
        #   First output tensor sizes are inserted in args.
        #   Then for each input tensor, its shape is inserted in args, followed by the input tensor itself.
        #   If the current input tensor has the same shape as any of the previous tensors, then its shape is not inserted.
        funcArgsList = OrderedDict()

        if not (Util.Config.disableTruncOpti):
            # TODO -- remove CreateTensor from uninterp function calls
            for ii, curArg in enumerate(node.argsList):
                curExpr = exprList[ii]
                curScale = self.scaleFacMapping[curExpr.idf]
                curType = curArg.type
                if (
                    (not (Type.isInt(curType)))
                    and (curScale > self.scaleFac)
                    and (curType.isSecret)
                ):
                    curProg = self.addTruncateFunctionCall(
                        curArg, "UninterpFuncCall", curExpr, curScale - self.scaleFac
                    )
                    progList.insert(0, curProg)
                    self.scaleFacMapping[curExpr.idf] = self.scaleFac

            self.scaleFacMapping[returnExpr.idf] = self.scaleFac

        tensorShapesFound = {}
        outputShape = node.type.shape
        for ii, curDim in enumerate(outputShape):
            funcArgsList[IR.Int(curDim, 32)] = "OutputShape_" + str(ii)
        tensorShapesFound[tuple(outputShape)] = True
        for ii in range(0, len(node.argsList)):
            if node.outputDiffInpDims < 2 and Type.isTensor(node.argsList[ii].type):
                curInpShape = node.argsList[ii].type.shape
                if (node.outputDiffInpDims == 1) or (
                    node.outputDiffInpDims == 0
                    and tuple(curInpShape) not in tensorShapesFound
                ):
                    for jj, curDim in enumerate(curInpShape):
                        funcArgsList[IR.Int(curDim, 32)] = (
                            "Input_" + str(ii) + self.varNameDelim + str(jj)
                        )
                    tensorShapesFound[tuple(curInpShape)] = True
            funcArgsList[exprList[ii]] = "inpExpr_" + str(ii)
        funcArgsList[returnExpr] = "output"

        comment = IR.Comment(str(node.metadata))
        progFinal = progList[0]
        if len(progList) > 1:
            for ii in range(1, len(progList)):
                progFinal = IRUtil.prog_merge(progFinal, progList[ii])
        progFinal = IRUtil.prog_merge(
            progFinal, IR.Prog([comment, IR.FuncCall(funcName, funcArgsList)])
        )

        progFinal = IRUtil.prog_merge(
            IR.Prog(
                [
                    IR.Decl(
                        returnExpr.idf,
                        node.type,
                        isSecret=False if node.isSecret is False else "secret",
                    )
                ]
            ),
            progFinal,
        )
        return (progFinal, returnExpr)

    def visitArgMax(self, node: AST.ArgMax, args=None):
        (prog_1, expr1) = self.visit(node.expr)
        (prog_2, expr2) = self.visit(node.dim)

        tmpExpr = self.getTempVar()
        outputShape = node.type.shape

        funcArgsList = OrderedDict()
        outputShape = node.type.shape
        for ii, curDim in enumerate(outputShape):
            funcArgsList[IR.Int(curDim, 32)] = "OutputShape_" + str(ii)
        for ii, curDim in enumerate(node.inShape):
            funcArgsList[IR.Int(curDim, 32)] = "OutputShape_" + str(ii)
        funcArgsList[expr1] = "inArr"
        funcArgsList[expr2] = "dim"
        funcArgsList[tmpExpr] = "outArr"

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[tmpExpr.idf] = -1

        funcCall = IR.FuncCall(
            "ArgMax" + self.varNameDelim + str(len(outputShape)), funcArgsList
        )
        comment = IR.Comment(str(node.metadata))
        prog_3 = IRUtil.prog_merge(prog_1, prog_2, IR.Prog([comment, funcCall]))
        prog_3 = IRUtil.prog_merge(IR.Prog([IR.Decl(tmpExpr.idf, node.type)]), prog_3)
        return (prog_3, tmpExpr)

    def visitInput(self, node: AST.Input, args=None):
        returnExpr = self.getTempVar()
        returnExpr.inputVar = True
        comment = IR.Comment(str(node.metadata))
        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[returnExpr.idf] = self.scaleFac
        return (
            IR.Prog(
                [
                    comment,
                    IR.Input(
                        returnExpr,
                        node.shape,
                        node.dataType,
                        node.isSecret,
                        node.inputByParty,
                    ),
                ]
            ),
            returnExpr,
        )

    def visitOutput(self, node: AST.Output, args=None):
        (prog_0, expr_0) = self.visit(node.expr)
        output = IR.Output(expr_0, node.outputToParty)
        prog = IRUtil.prog_merge(prog_0, IR.Prog([output]))
        expr = self.getTempVar()
        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[expr.idf] = self.scaleFac
        return (prog, expr)

    def visitReduce(self, node: AST.Reduce, args=None):
        (prog_1, expr1) = self.visit(node.expr)
        assert node.op in [AST.Operators.ADD, AST.Operators.Mean]

        # We already have the output shape so we dont need to calculate with keep_dims

        """
            We need to reduce across axes.
            Example: Say reduction axes are specified as 0,3 and keep dim = false
            output rank -> len(input_shape) - len(reduction_axes)
            output is 2D.
            for i1=[0:s1]
                for i2=[0:s2]
                    sum = 0
                    for i0=[0:s0]
                        for i3=[0:s3]
                            sum  = sum + input[i0][i1][i2][i3]
                    output[i1][i2] = sum / (s0 * s3)
            if keep dim == true, output rank is same as input. We generate:
                    output[0][i1][i2][0] = sum / (s0 * s3)

            Ideally the above loop nest is what we would want to generate. But since we have
            a division, we need to make calls to the div functionality and flatten the tensors.
            temp_flat[s1*s2];
            out_flat[s1*s2];
            for i1=[0:s1]
                for i2=[0:s2]
                    sum = 0
                    for i0=[0:s0]
                        for i3=[0:s3]
                            sum  = sum + input[i0][i1][i2][i3]
                    temp_flat[i1*s2 + i2] = sum
            ElemWiseVectorPublicDiv(size=s1*s2, inp=temp_flat, divisor=s0*s3, out=out_flat)
            for i1=[0:s1]
                for i2=[0:s2]
                  output[i1][i2] = out_flat[i1*s2 + i2]

        """
        reduced_dims = node.reductionAxesList
        inputShape = node.expr.type.shape
        perm = []
        calculated_shape = []
        inputiters = self.getTempIterators(node.expr.type.dim)
        outputiters = []
        no_elems = 1
        j = 0

        for i in range(len(inputShape)):
            if i not in reduced_dims:
                perm.append(i)
        # perm will now be [ 1 ,2 ] + [ 0, 3]
        perm.extend(reduced_dims)
        loop_shape = [inputShape[perm[i]] for i in range(len(inputShape))]
        shuffled_inputiters = [inputiters[perm[i]] for i in range(len(inputShape))]

        for i in range(len(inputShape)):
            if i not in reduced_dims:
                calculated_shape.append(inputShape[i])
                outputiters.append(inputiters[j])
                j = j + 1
            else:
                no_elems = no_elems * inputShape[i]
                if node.keepdims == 1:
                    calculated_shape.append(1)
                    outputiters.append(IR.Int(0, 32))

        if calculated_shape == []:
            calculated_shape = [1]
            outputiters.append(IR.Int(0, 32))

        outputShape = node.type.shape
        assert (
            calculated_shape == outputShape
        ), "calculate shape:{} - real_shape: {}".format(calculated_shape, outputShape)

        sumExpr = self.getTempVar()
        sumExpr_decl = IR.Decl(sumExpr.idf, Type.Int())
        initSumCmd = IR.Assn(sumExpr, IRUtil.zero)
        updateSumCmd = IR.Assn(
            sumExpr, IRUtil.add(sumExpr, IRUtil.addIndex(expr1, shuffled_inputiters))
        )

        if node.op == AST.Operators.Mean:
            outer_nesting = len(inputShape) - len(reduced_dims)
            temp_flat = self.getTempVar()
            temp_flat_decl = IR.Decl(
                temp_flat.idf,
                Type.Tensor(
                    [Util.get_volume(loop_shape[:outer_nesting])],
                    node.type.bitlen,
                    node.type.isSecret,
                    node.type.taint,
                ),
                isSecret=node.type.isSecret,
            )
            # i1*s2 + i2
            flat_idx_expr = IRUtil.getFlatArrIdxExpr(
                inputiters[:outer_nesting], loop_shape[:outer_nesting]
            )
            # temp_flat[i1*s2 + i2] = sum
            temp_flat_expr = IRUtil.addIndex(temp_flat, [flat_idx_expr])
            updateOutCmd = IR.Assn(temp_flat_expr, sumExpr)
        elif node.op == AST.Operators.ADD:
            output = self.getTempVar()
            output_decl = IR.Decl(output.idf, node.type)
            out_expr = IRUtil.addIndex(output, outputiters)
            updateOutCmd = IR.Assn(out_expr, sumExpr)

        # Generate the sum loop
        inner_loops_processed = 0
        sum_loop = [updateSumCmd]
        for i in reversed(range(len(loop_shape))):
            sum_loop = [IR.For(inputiters[i], 0, sum_loop, 0, endInt=loop_shape[i])]
            inner_loops_processed += 1
            if inner_loops_processed == len(reduced_dims):
                sum_loop = [initSumCmd] + sum_loop + [updateOutCmd]

        if node.op == AST.Operators.ADD:
            comment = IR.Comment(str(node.metadata))
            final_prog = IRUtil.prog_merge(
                prog_1,
                IR.Prog([comment]),
                IR.Prog([sumExpr_decl, output_decl]),
                IR.Prog(sum_loop),
            )
            if not (Util.Config.disableTruncOpti):
                self.scaleFacMapping[output.idf] = self.scaleFacMapping[expr1.idf]

            return (final_prog, output)

        # Insert call to ElemWiseVectorPublicDiv(size=s1*s2, inp=temp_flat, divisor=s0*s3, out=out_flat)
        out_flat = self.getTempVar()
        out_flat_decl = IR.Decl(
            out_flat.idf,
            Type.Tensor(
                [Util.get_volume(loop_shape[:outer_nesting])],
                node.type.bitlen,
                node.type.isSecret,
                node.type.taint,
            ),
            isSecret=node.type.isSecret,
        )
        argsDict = OrderedDict()
        argsDict[IR.Int(Util.get_volume(loop_shape[:outer_nesting]), 32)] = "size"
        argsDict[temp_flat] = "input"
        argsDict[IR.Int(Util.get_volume(loop_shape[outer_nesting:]), 32)] = "divisor"
        argsDict[out_flat] = "output"
        div_call = IR.FuncCall("ElemWiseVectorPublicDiv", argsDict)

        # Free temp_flat here
        # Clear temp arrays
        argsDict = OrderedDict()
        argsDict[IR.Int(Util.get_volume(loop_shape[:outer_nesting]), 32)] = "size"
        argsDict[temp_flat] = "A"
        free_temp_flat_call = IR.FuncCall("ClearMemSecret1", argsDict)

        # Unflatten the output
        output = self.getTempVar()
        output_decl = IR.Decl(output.idf, node.type)
        out_expr = IRUtil.addIndex(output, outputiters)
        out_flat_expr = IRUtil.addIndex(out_flat, [flat_idx_expr])
        out_assn_expr = IR.Assn(out_expr, out_flat_expr)
        unflatten_loop = IRUtil.loop(
            loop_shape[:outer_nesting], inputiters[:outer_nesting], [out_assn_expr]
        )

        # Free out_flat here
        argsDict = OrderedDict()
        argsDict[IR.Int(Util.get_volume(loop_shape[:outer_nesting]), 32)] = "size"
        argsDict[out_flat] = "A"
        free_out_flat_call = IR.FuncCall("ClearMemSecret1", argsDict)

        if not (Util.Config.disableTruncOpti):
            self.scaleFacMapping[output.idf] = self.scaleFacMapping[expr1.idf]

        comment = IR.Comment(str(node.metadata))
        final_prog = IRUtil.prog_merge(
            prog_1,
            IR.Prog([comment]),
            IR.Prog([sumExpr_decl, temp_flat_decl, out_flat_decl, output_decl]),
            IR.Prog(sum_loop),
            IR.Prog([div_call]),
            IR.Prog([free_temp_flat_call]),
            IR.Prog(unflatten_loop),
            IR.Prog([free_out_flat_call]),
        )

        return (final_prog, output)

    def visitFusedBatchNorm(self, node: AST.FusedBatchNorm, args=None):
        (prog1, expr1) = self.visit(node.expr)
        (prog2, expr2) = self.visit(node.multExpr)
        (prog3, expr3) = self.visit(node.addExpr)

        returnExpr = self.getTempVar()

        funcArgsList = OrderedDict()
        for ii, elem in enumerate(node.type.shape):
            funcArgsList[IR.Int(elem, 32)] = "elem_" + str(ii)
        funcArgsList[expr1] = "expr"
        funcArgsList[expr2] = "multExpr"
        funcArgsList[expr3] = "addExpr"

        progExtraBefore = IR.Prog([])
        multExprScaleDownSf = self.scaleFac
        addExprScaleUpSf = 0
        if not (Util.Config.disableTruncOpti):
            # TruncOpti is on
            multExprScaleDownSf = 0
            addExprScaleUpSf = 0

            expr_sf = self.scaleFacMapping[expr1.idf]
            multExpr_sf = self.scaleFacMapping[expr2.idf]
            addExpr_sf = self.scaleFacMapping[expr3.idf]
            if expr_sf > self.scaleFac:
                # Scale down needed
                progExtraBefore = IRUtil.prog_merge(
                    progExtraBefore,
                    self.addTruncateFunctionCall(
                        node.expr, "FusedBatchNorm", expr1, expr_sf - self.scaleFac
                    ),
                )
                self.scaleFacMapping[expr1.idf] = self.scaleFac

            if multExpr_sf > self.scaleFac:
                # Scale down needed
                progExtraBefore = IRUtil.prog_merge(
                    progExtraBefore,
                    self.addTruncateFunctionCall(
                        node.multExpr,
                        "FusedBatchNorm",
                        expr2,
                        multExpr_sf - self.scaleFac,
                    ),
                )
                self.scaleFacMapping[expr2.idf] = self.scaleFac

            final_sf = 2 * self.scaleFac
            assert final_sf >= addExpr_sf
            if final_sf > addExpr_sf:
                addExprScaleUpSf = final_sf - addExpr_sf
                self.scaleFacMapping[expr3.idf] += addExprScaleUpSf

            self.scaleFacMapping[returnExpr.idf] = final_sf

        funcArgsList[IR.Int(multExprScaleDownSf, 32)] = "multExprScaleDownSf"
        funcArgsList[IR.Int(addExprScaleUpSf, 32)] = "addExprScaleUpSf"
        funcArgsList[returnExpr] = "returnExpr"

        funcCallIR = IR.FuncCall(
            "FusedBatchNorm"
            + self.varNameDelim
            + str(len(node.type.shape))
            + self.varNameDelim  # one for output
            + str(len(node.type.shape))
            + self.varNameDelim  # one for input
            + str(len(node.multExpr.type.shape))
            + self.varNameDelim
            + str(len(node.addExpr.type.shape)),
            funcArgsList,
        )

        comment = IR.Comment(str(node.metadata))
        returnProg = IRUtil.prog_merge(
            prog1, prog2, prog3, progExtraBefore, IR.Prog([comment, funcCallIR])
        )

        returnProg = IRUtil.prog_merge(
            IR.Prog([IR.Decl(returnExpr.idf, node.type)]), returnProg
        )
        return (returnProg, returnExpr)
