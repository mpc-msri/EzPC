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

import numpy as np

from IR.IR import *
from Util import *


def init():
    global zero, one, negone, negmax

    zero = Int(0)
    one = Int(1)
    negone = Int(-1)
    negmax = Int.negMax()


def add(e1: Expr, e2: Expr) -> Expr:
    return IntBop(e1, Op.Op["+"], e2)


def sub(e1: Expr, e2: Expr) -> Expr:
    return IntBop(e1, Op.Op["-"], e2)


def mul(e1: Expr, e2: Expr) -> Expr:
    return IntBop(e1, Op.Op["*"], e2)


def div(e1: Expr, e2: Expr) -> Expr:
    return IntBop(e1, Op.Op["/"], e2)


def inc(e: Expr) -> Expr:
    return add(e, one)


def dec(e: Expr) -> Expr:
    return sub(e, one)


def andd(e1: Expr, e2: Expr) -> Expr:
    return BoolBop(e1, Op.Op["&&"], e2)


def orr(e1: Expr, e2: Expr) -> Expr:
    return BoolBop(e1, Op.Op["||"], e2)


def eq(e1: Expr, e2: Expr) -> Expr:
    return BoolCop(e1, Op.Op["=="], e2)


def neq(e1: Expr, e2: Expr) -> Expr:
    return BoolCop(e1, Op.Op["!="], e2)


def lt(e1: Expr, e2: Expr) -> Expr:
    return BoolCop(e1, Op.Op["<"], e2)


def lte(e1: Expr, e2: Expr) -> Expr:
    return BoolCop(e1, Op.Op["<="], e2)


def gt(e1: Expr, e2: Expr) -> Expr:
    return BoolCop(e1, Op.Op[">"], e2)


def gte(e1: Expr, e2: Expr) -> Expr:
    return BoolCop(e1, Op.Op[">="], e2)


def bitAnd(e1: Expr, e2: Expr) -> Expr:
    return IntBop(e1, Op.Op["&"], e2)


def max(e1: Expr, e2: Expr) -> Expr:
    return CExpr(BoolCop(e1, Op.Op[">"], e2), e1, e2)


def max_uint(e1: Expr, e2: Expr) -> Expr:
    return CExpr(BoolCop(e1, Op.Op[">"], e2), e1, e2)


def max_sint(e1: Expr, e2: Expr) -> Expr:
    return cond_zero(
        e1, cond_zero(e2, max_uint(e1, e2), e1), cond_zero(e2, e2, max_uint(e1, e2))
    )


def negate(e: Expr) -> Expr:
    return IntUop(Op.Op["-"], e)


def shl(e: Expr, n: int) -> Expr:
    assert n >= 0
    if n == 0:
        return e
    return IntBop(e, Op.Op["<<"], Int(n))


def shrUint(e: Expr, n: int) -> Expr:
    assert n >= 0
    if n == 0:
        return e
    return IntBop(e, Op.Op[">>"], Int(n))


def shr(e: Expr, n: int) -> Expr:
    return shrDefault(e, n)


def shrDefault(e: Expr, n: int) -> Expr:
    assert n >= 0
    if n == 0:
        return e
    return cond_zero(
        e,
        IntBop(e, Op.Op[">>"], Int(n)),
        IntBop(
            IntBop(IntBop(e, Op.Op["^"], negone), Op.Op[">>"], Int(n)),
            Op.Op["^"],
            negone,
        ),
    )


def shrVar(e: Expr, n: Var) -> Expr:
    return cond_zero(
        e,
        IntBop(e, Op.Op[">>"], n),
        IntBop(
            IntBop(IntBop(e, Op.Op["^"], negone), Op.Op[">>"], n), Op.Op["^"], negone
        ),
    )


def castToInt(e: Expr):
    return TypeCast(DataType.getIntStr(), e)


def castToFloat(e: Expr):
    return TypeCast(DataType.getFloatStr(), e)


def addIndex(var: Var, indices: list, prefix: bool = False) -> Var:
    if prefix == False:
        return Var(var.idf, var.idx + indices, var.inputVar)
    else:
        return Var(var.idf, indices + var.idx, var.inputVar)


def cond_zero(e: Expr, et: Expr, ef: Expr) -> Expr:
    return CExpr(BoolCop(e, Op.Op[">"], zero), et, ef)


def relu(e: Expr):
    return cond_zero(e, e, zero)


def loop_shr(lhs: Expr, rhs: Expr, shape: list, iters: list, n: int) -> CmdList:
    lhs_elt = addIndex(lhs, iters)
    rhs_elt = addIndex(rhs, iters)
    return loop(shape, iters, [Assn(lhs_elt, shr(rhs_elt, n))])


def initVarToZero(e: Expr) -> Cmd:
    return Assn(e, Int(0))


def incCmd(e: Var) -> Cmd:
    return Assn(e, inc(e))


def decCmd(e: Var) -> Cmd:
    return Assn(e, dec(e))


def prog_merge(*prog_l, resource=0):
    cmd_l = flatten([prog.cmd_l for prog in prog_l])
    Res = 0
    for x in prog_l:
        Res = Res + x.resource
    return Prog(cmd_l, resource=Res)


# multiplexer
def add_idx_priv(var: Var, e: Expr, n: int, offset: int = 0) -> Expr:
    assert n >= 1
    mask = 1 << (n - 1)

    # use if-else
    if False:
        # for n=3:
        # if e & 100 == 0:
        #   if e & 010 == 0:
        #     if e & 001 == 0: var[000]
        #     else: var[001]
        #   else:
        #     if e & 001 == 0: var[010]
        #     else: var[011]
        # else: ...
        expr_cmp = eq(IntBop(e, Op.Op["&"], Int(mask)), zero)
        if n == 1:
            return CExpr(
                expr_cmp,
                addIndex(var, [Int(offset + 0)]),
                addIndex(var, [Int(offset + mask)]),
            )
        else:
            return CExpr(
                expr_cmp,
                add_idx_priv(var, e, n - 1, offset + 0),
                add_idx_priv(var, e, n - 1, offset + mask),
            )
    # use *, +
    else:
        # for n=2:
        # (1-(e&10)>>1) * ((1-(e&01)>>0)*var[00] + ((e&01)>>0)*var[01]) +
        # ( (e&10)>>1) * ((1-(e&01)>>0)*var[10] + ((e&01)>>0)*var[11])
        expr_1 = shrUint(IntBop(e, Op.Op["&"], Int(mask)), n - 1)
        expr_0 = sub(one, expr_1)
        if n == 1:
            return add(
                mul(expr_0, addIndex(var, [Int(offset + 0)])),
                mul(expr_1, addIndex(var, [Int(offset + mask)])),
            )
        else:
            return add(
                mul(expr_0, add_idx_priv(var, e, n - 1, offset + 0)),
                mul(expr_1, add_idx_priv(var, e, n - 1, offset + mask)),
            )


# iteration
def loop(shape: list, iters: list, cmdl_body: CmdList, factor=0) -> CmdList:
    cmdl_for = cmdl_body
    for i in reversed(range(len(shape))):
        cmdl_for = [For(iters[i], 0, cmdl_for, factor, endInt=shape[i])]
    return cmdl_for


def print_loop(shape: list, iters: list, cmdl_body: CmdList, factor=0) -> CmdList:
    cmdl_for = cmdl_body
    for i in reversed(range(len(shape))):
        cmdl_for = [
            For(iters[i], 0, lt(iters[i], Int(shape[i])), cmdl_for, factor),
            Print(Var('""')),
        ]
    return cmdl_for


# For tensor A of shape = 7 x 1 x 5
# And out_iters         = [i0, i1, i2, i3]
# Broadcast mask        = [True, False, True, False]
# We generate iters     = A[i1][0][i3]
# If input is scalar, broadcast_mask=[] and inp_shape=[]
def getMaskedIters(broadcast_mask: list, out_iters: list, inp_shape: list):
    base_idx = len(out_iters) - len(inp_shape)
    masked_iters = []
    for i in range(len(broadcast_mask)):
        if broadcast_mask[i]:
            masked_iters.append(Int(0, 32))
        else:
            masked_iters.append(out_iters[base_idx])
        base_idx += 1
    return masked_iters


# Given input
# A      (4d array):  8 x 1 x 6 x 1
# B      (3d array):      7 x 1 x 5
# We generate a loop with
# Result (4d array):  8 x 7 x 6 x 5
# for i0=[0:8]
#   for i1=[0:7]
#     for i2=[0:6]
#       for i3=[0:8]
#         Result[i0][i1][i2][i3] = A[i0][0][i2][0] + B[i1][0][i3]
def generateBroadcastLoopBOp(
    expr_1, inp1_shape: list, expr_2, inp2_shape: list, expr_out, op: Op.Op
):
    output_shape, broadcast_mask_1, broadcast_mask_2 = Util.getBroadcastShapes(
        inp1_shape, inp2_shape
    )
    out_iters = [Var("i" + str(i)) for i in range(len(output_shape))]
    inp1_iters = getMaskedIters(broadcast_mask_1, out_iters, inp1_shape)
    inp2_iters = getMaskedIters(broadcast_mask_2, out_iters, inp2_shape)

    inp1_arr_expr = addIndex(expr_1, inp1_iters)
    inp2_arr_expr = addIndex(expr_2, inp2_iters)
    out_arr_expr = addIndex(expr_out, out_iters)

    assign_expr = Assn(out_arr_expr, IntBop(inp1_arr_expr, op, inp2_arr_expr))
    out_loop = loop(output_shape, out_iters, [assign_expr])
    out_prog = Prog(out_loop)
    return out_prog


# Generates the index into a flattened tensor.
# Example:
# for i1=[0:s1]
#   for i2=[0:s2]
#     for i3=[0:s3]
#       for i4=[0:s4]
# generate (i1*s2*s3*s4) + (i2*s3*s4) + (i3*s4) + (i4);
def getFlatArrIdxExpr(iters: list, shape: list):
    assert len(iters) == len(
        shape
    ), "No. of loop idx vars should be equal to loop shapes"
    flat_idx_expr = Int(0, 32)
    for i in range(len(iters)):
        vol = get_volume(shape[i + 1 :])
        flat_idx_expr = add(flat_idx_expr, mul(iters[i], Int(vol, 32)))
    return flat_idx_expr
