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

import Util
import operator
from functools import reduce
import AST.AST as AST
from AST.ASTVisitor import ASTVisitor
from enum import Enum, auto
import copy
import sys


class Type:
    pass


"""
We want to analyse the taint of every tensor that flows in the graph.
The possible taints for tensors are:
{
    Client: Input to the ML model (eg: the image input)
    Server: The weights of the model
    ClientXServer[C&S]: A tensor that is dervied after operations on both client and server tensors.
    Secret_constant: A tensor that is a constant but declared as a secret
    Public_constant: A tensor that is a constant but declared as public
}
Note: For ML models we don't expect to encounter any secret_constants and instead expect them
to be encoded as weights of the model and so instead has the server taint.
We infer taints in the following manner:
                    Client         Server      C&S     Secret_constant     Public_constant
Client              Client         C&S         C&S     Client              Client
Server              C&S            Server      C&S     Server              Server
C&S                 C&S            C&S         C&S     C&S                 C&S
Secret_constant     C&S            C&S         C&S     Secret_constant     Secret_constant
Public_constant     Client         Server      C&S     Secret_constant     Public_constant
"""


class Taints(Enum):
    CLIENT = auto()
    SERVER = auto()
    CLIENT_SERVER = auto()
    SECRET_C = auto()
    PUBLIC_C = auto()


constantTaintsMapping = {True: Taints.SECRET_C, False: Taints.PUBLIC_C}

TaintsTable = {
    Taints.CLIENT: {
        Taints.CLIENT: Taints.CLIENT,
        Taints.SERVER: Taints.CLIENT_SERVER,
        Taints.CLIENT_SERVER: Taints.CLIENT_SERVER,
        Taints.SECRET_C: Taints.CLIENT,
        Taints.PUBLIC_C: Taints.CLIENT,
    },
    Taints.SERVER: {
        Taints.CLIENT: Taints.CLIENT_SERVER,
        Taints.SERVER: Taints.SERVER,
        Taints.CLIENT_SERVER: Taints.CLIENT_SERVER,
        Taints.SECRET_C: Taints.SERVER,
        Taints.PUBLIC_C: Taints.SERVER,
    },
    Taints.CLIENT_SERVER: {
        Taints.CLIENT: Taints.CLIENT_SERVER,
        Taints.SERVER: Taints.CLIENT_SERVER,
        Taints.CLIENT_SERVER: Taints.CLIENT_SERVER,
        Taints.SECRET_C: Taints.CLIENT_SERVER,
        Taints.PUBLIC_C: Taints.CLIENT_SERVER,
    },
    Taints.SECRET_C: {
        Taints.CLIENT: Taints.CLIENT,
        Taints.SERVER: Taints.SERVER,
        Taints.CLIENT_SERVER: Taints.CLIENT_SERVER,
        Taints.SECRET_C: Taints.SECRET_C,
        Taints.PUBLIC_C: Taints.SECRET_C,
    },
    Taints.PUBLIC_C: {
        Taints.CLIENT: Taints.CLIENT,
        Taints.SERVER: Taints.SERVER,
        Taints.CLIENT_SERVER: Taints.CLIENT_SERVER,
        Taints.SECRET_C: Taints.SECRET_C,
        Taints.PUBLIC_C: Taints.PUBLIC_C,
    },
}


def getTaint_taint(t1: Taints, t2: Taints):
    return TaintsTable[t1][t2]


def getTaint_type(t1: Type, t2: Type):
    return TaintsTable[t1.taint][t2.taint]


class Int(Type):
    def __init__(self, bitlen=-1, isSecret=False, taint=Taints.PUBLIC_C):
        if bitlen == -1:
            self.bitlen = Util.Config.wordLength
        else:
            self.bitlen = bitlen
        self.isSecret = isSecret
        self.taint = taint

    def __copy__(self):
        return type(self)(self.bitlen, self.isSecret, self.taint)


class Float(Type):
    def __init__(self, bitlen=-1, isSecret=False, taint=Taints.PUBLIC_C):
        if bitlen == -1:
            self.bitlen = Util.Config.wordLength
        else:
            self.bitlen = bitlen
        self.isSecret = isSecret
        self.taint = taint

    def __copy__(self):
        return type(self)(self.bitlen, self.isSecret, self.taint)


class Unit(Type):
    pass


class Tensor(Type):
    def __init__(
        self, shape: list, dataType, bitlen=-1, isSecret=True, taint=Taints.PUBLIC_C
    ):
        # print(f"Type.py : Tensor : __init__ --> shape = {shape}, dataType = {dataType}")
        self.shape = shape
        self.dim = len(shape)
        if bitlen == -1:
            self.bitlen = Util.Config.wordLength
        else:
            self.bitlen = bitlen
        self.isSecret = isSecret
        self.taint = taint
        self.dataType = dataType

    def __copy__(self):
        return type(self)(self.shape, self.bitlen, self.isSecret, self.taint)

    def size(self):
        return reduce(operator.mul, self.shape, 1)

    # Tensor without any dimension (float) or a tensor with all dimensions equal to 1
    def isShapeOne(self):
        return self.dim == 0 or self.size() == 1


def isInt(type: Type):
    return isinstance(type, Int)


def isFloat(type: Type):
    return isinstance(type, Float)


def isNumeric(type: Type):
    return isinstance(type, Int) or isinstance(type, Float)


def isTensor(type: Type):
    return isinstance(type, Tensor)


def isUnit(type: Type):
    return isinstance(type, Unit)


def isEqual(type1: Type, type2: Type):
    if isNumeric(type1) and isNumeric(type2):
        return True
    elif isTensor(type1) and isTensor(type2):
        if type1.dim != type2.dim:
            return False
        return type1.shape == type2.shape
    else:
        assert False


class InferType(ASTVisitor):
    def __init__(self, debug=False):
        self.debug = debug
        self.indent = 0

    def visitInt(self, node: AST.Int, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitInt")
            self.indent += 1
        bitlen = Util.Config.wordLength
        if node.bitLen:
            bitlen = node.bitLen
        node.type = Int(bitlen, node.isSecret, constantTaintsMapping[node.isSecret])

        if self.debug:
            self.indent -= 1
        return node.type

    def visitFloat(self, node: AST.Float, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitFloat")
            self.indent += 1
        # Float is represented as an int in fixedpt.
        node.type = Float(
            isSecret=node.isSecret, taint=constantTaintsMapping[node.isSecret]
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitId(self, node: AST.ID, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitId")
            self.indent += 1
        if node.name not in node.gamma:
            print(
                "Error in type checking: Found id which is not contained in gamma.",
                node.name,
                file=sys.stderr,
            )
            assert False
        else:
            node.type = node.gamma[node.name]

        if self.debug:
            self.indent -= 1
        return node.type

    def visitDecl(self, node: AST.Decl, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitDecl")
            self.indent += 1
        # TODO -- fill in bitlen properly
        if node.shape == []:
            node.type = Int(
                isSecret=node.isSecret, taint=constantTaintsMapping[node.isSecret]
            )
        else:
            node.type = Tensor(
                shape=node.shape,
                dataType=node.dataType,
                # baseType=node.dataType
                isSecret=node.isSecret,
                taint=constantTaintsMapping[node.isSecret],
            )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitTranspose(self, node: AST.Transpose, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitTranspose")
            self.indent += 1
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType)

        perm = node.perm
        shape = exprType.shape
        if perm is None:
            perm = [i for i in reversed(range(len(shape)))]
        new_shape = []
        for i in perm:
            new_shape.append(shape[i])
        node.type = Tensor(
            new_shape,
            node.expr.dataType,
            exprType.bitlen,
            exprType.isSecret,
            exprType.taint,
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitSlice(self, node: AST.Slice, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitSlice")
            self.indent += 1
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)
        assert isTensor(exprType)

        subscriptRanges = node.subscriptRanges
        shape = []
        for i in subscriptRanges:
            start = i[0]
            end = i[1]
            size = end - start + 1
            shape.append(size)

        assert len(shape) == len(exprType.shape)
        for i in range(0, len(shape)):
            assert shape[i] <= exprType.shape[i], " for {}".format(node.metadata)

        node.type = Tensor(
            shape, exprType.dataType, exprType.bitlen, exprType.isSecret, exprType.taint
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitReshape(self, node: AST.Reshape, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitReshape")
            self.indent += 1

        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        assert isTensor(exprType) and exprType.dim >= 1

        # Reshape is valid if the total number of elements remain same after reshape
        assert reduce(operator.mul, exprType.shape, 1) == reduce(
            operator.mul, node.shape, 1
        )
        node.type = Tensor(
            node.shape,
            node.expr.dataType,
            exprType.bitlen,
            exprType.isSecret,
            exprType.taint,
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitPool(self, node: AST.Pool, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitPool")
            self.indent += 1
        node.expr.gamma = dict(node.gamma)
        exprType = self.visit(node.expr)

        # Implementation only performs maxpool over a 4D input
        assert isTensor(exprType) and exprType.dim == 4
        [N, H, W, CI] = exprType.shape
        FH = node.options[AST.PaddingKeysDict.FH]
        FW = node.options[AST.PaddingKeysDict.FW]
        zPadHLeft = node.options[AST.PaddingKeysDict.zPadHLeft]
        zPadHRight = node.options[AST.PaddingKeysDict.zPadHRight]
        zPadWLeft = node.options[AST.PaddingKeysDict.zPadWLeft]
        zPadWRight = node.options[AST.PaddingKeysDict.zPadWRight]
        strideH = node.options[AST.PaddingKeysDict.strideH]
        strideW = node.options[AST.PaddingKeysDict.strideW]

        newH = ((H + zPadHLeft + zPadHRight - FH) // strideH) + 1
        newW = ((W + zPadWLeft + zPadWRight - FW) // strideW) + 1

        node.type = Tensor(
            [N, newH, newW, CI],
            exprType.dataType,
            exprType.bitlen,
            exprType.isSecret,
            exprType.taint,
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitUOp(self, node: AST.UOp, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitUOp")
            self.indent += 1
        node.expr.gamma = dict(node.gamma)
        node.type = self.visit(node.expr)

        if self.debug:
            self.indent -= 1
        return node.type

    def visitBOp(self, node: AST.BOp, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitBOp")
            self.indent += 1
        node.expr1.gamma = dict(node.gamma)
        eType = self.visit(node.expr1)

        node.expr2.gamma = dict(node.gamma)
        fType = self.visit(node.expr2)

        if node.op in [
            AST.Operators.ADD,
            AST.Operators.SUB,
            AST.Operators.Equal,
            AST.Operators.ElemWiseMul,
            AST.Operators.ElemWiseDiv,
        ]:
            # Ops supporting broadcasting
            return self.typeCheckBroadcastOps(node, eType, fType)
        elif node.op == AST.Operators.MUL:
            return self.visitBopMul(node, eType, fType)
        elif node.op == AST.Operators.CONV:
            return self.visitBopConv(node, eType, fType)
        elif node.op == AST.Operators.CONVTRANSPOSE:
            return self.visitBopConvTranspose(node, eType, fType)
        else:
            assert False

    def typeCheckBroadcastOps(self, node: AST.BOp, eType: Type, fType: Type):
        # Ops which support broadcasting have different type checking
        # If adding a new op here which supports broadcasting, then be careful!
        # Currently, its assumed the op is commutative. If that is not true, following will be wrong !

        assert node.op in [
            AST.Operators.ADD,
            AST.Operators.SUB,
            AST.Operators.Equal,
            AST.Operators.ElemWiseMul,
            AST.Operators.ElemWiseDiv,
        ]
        if isInt(eType) and isInt(fType):
            node.type = Int(eType.bitlen)
        elif isFloat(eType) and isFloat(fType):
            node.type = Float()
        elif isTensor(eType) and isTensor(fType):
            output_shape, _, _ = Util.getBroadcastShapes(eType.shape, fType.shape)
            # print(f"Type.py : typeCheckBroadcastOps : eType = {eType}, fType = {fType}")
            node.type = Tensor(
                shape=output_shape, dataType=eType.dataType, bitlen=eType.bitlen
            )
        elif isTensor(eType) and isNumeric(fType):
            output_shape, _, _ = Util.getBroadcastShapes(eType.shape, [])
            node.type = Tensor(
                shape=output_shape, dataType=eType.dataType, bitlen=eType.bitlen
            )
        elif isNumeric(eType) and isTensor(fType):
            output_shape, _, _ = Util.getBroadcastShapes([], fType.shape)
            node.type = Tensor(
                shape=output_shape, dataType=eType.dataType, bitlen=eType.bitlen
            )
        else:
            print(eType, fType)
            assert False

        node.type.taint = getTaint_type(eType, fType)
        node.type.isSecret = eType.isSecret or fType.isSecret

        if self.debug:
            self.indent -= 1
        return node.type

    def visitBopMul(self, node: AST.BOp, eType: Type, fType: Type, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitBopMul")
            self.indent += 1
        if isInt(eType) and isInt(fType):
            node.type = Int(eType.bitlen, eType.isSecret)
        elif isFloat(eType) and isFloat(fType):
            node.type = Float(eType.bitlen, eType.isSecret)
        elif isTensor(eType) and isTensor(fType):
            if eType.dim == 0:
                node.type = copy.copy(fType)
            elif fType.dim == 0:
                node.type = copy.copy(eType)
            else:
                assert eType.dim == 2 and fType.dim == 2
                [n1, n2] = eType.shape
                [n3, n4] = fType.shape
                assert n2 == n3
                node.type = Tensor([n1, n4], eType.dataType, eType.bitlen)
        else:
            print("Error: Unknown condition in type checking.", file=sys.stderr)
            assert False

        node.type.taint = getTaint_type(eType, fType)
        node.type.isSecret = eType.isSecret or fType.isSecret

        if self.debug:
            self.indent -= 1
        return node.type

    def visitBopConv(self, node: AST.BOp, eType: Type, fType: Type, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitBopConv")
            self.indent += 1
        assert isTensor(eType) and isTensor(fType)
        convDim = 2
        group = 1
        if AST.PaddingKeysDict.ConvDim in node.options:
            convDim = node.options[AST.PaddingKeysDict.ConvDim]

        if convDim == 2:
            assert eType.dim == 4 and fType.dim == 4
        elif convDim == 3:
            assert eType.dim == 5 and fType.dim == 5
        else:
            assert False

        N = D = H = W = CI = FD = FH = FW = CI1 = CO = -1
        newD = -1
        if convDim == 2:
            [N, H, W, CI] = eType.shape
            [FH, FW, CI1, CO] = fType.shape
        elif convDim == 3:
            [N, D, H, W, CI] = eType.shape
            [FD, FH, FW, CI1, CO] = fType.shape
            assert FD == node.options[AST.PaddingKeysDict.FD]
            zPadDLeft = node.options[AST.PaddingKeysDict.zPadDLeft]
            zPadDRight = node.options[AST.PaddingKeysDict.zPadDRight]
            strideD = node.options[AST.PaddingKeysDict.strideD]

            newD = ((D + zPadDLeft + zPadDRight - FD) // strideD) + 1
        else:
            assert False

        if AST.PaddingKeysDict.group in node.options:
            group = node.options[AST.PaddingKeysDict.group]

        assert FH == node.options[AST.PaddingKeysDict.FH]
        assert FW == node.options[AST.PaddingKeysDict.FW]
        assert CI1 * group == CI, "FCI={} group={} CI={}".format(CI1, group, CI)
        zPadHLeft = node.options[AST.PaddingKeysDict.zPadHLeft]
        zPadHRight = node.options[AST.PaddingKeysDict.zPadHRight]
        zPadWLeft = node.options[AST.PaddingKeysDict.zPadWLeft]
        zPadWRight = node.options[AST.PaddingKeysDict.zPadWRight]
        strideH = node.options[AST.PaddingKeysDict.strideH]
        strideW = node.options[AST.PaddingKeysDict.strideW]

        newH = ((H + zPadHLeft + zPadHRight - FH) // strideH) + 1
        newW = ((W + zPadWLeft + zPadWRight - FW) // strideW) + 1

        if convDim == 2:
            shape = [N, newH, newW, CO]
        elif convDim == 3:
            shape = [N, newD, newH, newW, CO]
        node.type = Tensor(
            shape,
            eType.dataType,
            eType.bitlen,
            eType.isSecret or fType.isSecret,
            getTaint_type(eType, fType),
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitBopConvTranspose(self, node: AST.BOp, eType: Type, fType: Type, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitBopConvTranspose")
            self.indent += 1
        assert isTensor(eType) and isTensor(fType)

        convDim = 2
        if AST.PaddingKeysDict.ConvDim in node.options:
            convDim = node.options[AST.PaddingKeysDict.ConvDim]

        if convDim == 2:
            [N, HP, WP, CI1] = eType.shape
            [FH, FW, CO, CI] = fType.shape
        elif convDim == 3:
            [N, DP, HP, WP, CI1] = eType.shape
            [FD, FH, FW, CO, CI] = fType.shape
        else:
            assert False
        assert CI1 == CI
        if convDim == 3:
            outputImgD = node.options[AST.PaddingKeysDict.outputImgD]
        outputImgH = node.options[AST.PaddingKeysDict.outputImgH]
        outputImgW = node.options[AST.PaddingKeysDict.outputImgW]

        if convDim == 2:
            shape = [N, outputImgH, outputImgW, CO]
        else:
            shape = [N, outputImgD, outputImgH, outputImgW, CO]

        # Logic explanation:
        #   ConvTranpose can be thought of as the inverse of some convolution for which it is doing the upsampling.
        #   For calculation of padding in the convTranspose operation, the output image size is required.
        #   This is why TF also mandates the operator to be specified with output size.
        #   This conv transpose operation can be thought of as conv between output
        #       of size shape = [N, outputImgH, outputImgW, CI], and filter of size [FH, FW, CI, CO].
        #       Hence, the input for this convTranspose would be [N, HP, WP, CO]

        node.type = Tensor(
            shape,
            eType.dataType,
            eType.bitlen,
            eType.isSecret | fType.isSecret,
            getTaint_type(eType, fType),
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitFunc(self, node: AST.Func, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitFunc")
            self.indent += 1
        node.expr.gamma = dict(node.gamma)
        eType = self.visit(node.expr)

        if node.op == AST.Operators.RELU:
            assert isTensor(eType) and eType.dim >= 1
            node.type = copy.copy(eType)
        elif node.op == AST.Operators.TANH:
            assert isTensor(eType)
            node.type = copy.copy(eType)
        elif node.op == AST.Operators.SIGMOID:
            assert isTensor(eType)
            node.type = copy.copy(eType)
        elif node.op == AST.Operators.SOFTMAX:
            assert isTensor(eType)
            node.type = copy.copy(eType)
        elif node.op == AST.Operators.SQRT:
            assert isTensor(eType)
            node.type = copy.copy(eType)
        elif node.op == AST.Operators.RSQRT:
            assert isTensor(eType)
            node.type = copy.copy(eType)
        elif node.op == AST.Operators.Floor:
            node.type = copy.copy(eType)
        elif node.op == AST.Operators.Shape:
            assert isTensor(eType)
            node.type = Tensor(
                [len(eType.shape)],
                eType.dataType,
                eType.bitlen,
                eType.isSecret,
                eType.taint,
            )
        elif node.op == AST.Operators.ClearMemSecret:
            node.type = Unit()
        elif node.op == AST.Operators.ClearMemPublic:
            node.type = Unit()
        else:
            print("Type inference not implemented for", node.op)
            assert False

        if self.debug:
            self.indent -= 1
        return node.type

    def visitLet(self, node: AST.Let, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitLet")
            self.indent += 1
        node.decl.gamma = dict(node.gamma)
        eType = self.visit(node.decl)

        node.name.gamma = {node.name.name: eType}
        self.visit(node.name)

        node.expr.gamma = dict(node.gamma)
        node.expr.gamma[node.name.name] = eType
        fType = self.visit(node.expr)

        node.type = copy.copy(fType)

        if self.debug:
            self.indent -= 1
        return node.type

    def visitUninterpFuncCall(self, node: AST.UninterpFuncCall, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitUninterpFuncCall")
            self.indent += 1
        # Assert that outputShape and inputDims are lists of int astNode.
        assert len(node.argsList) > 0
        isSecret = False
        taint = Taints.PUBLIC_C
        for curArg in node.argsList:
            curArg.gamma = dict(node.gamma)
            eType = self.visit(
                curArg
            )  # This should set the type of each of the input nodes
            isSecret = isSecret or eType.isSecret
            taint = getTaint_taint(taint, eType.taint)
        outputShape = node.outputShape
        node.type = Tensor(outputShape, "float32", isSecret=isSecret, taint=taint)

        if self.debug:
            self.indent -= 1
        return node.type

    def visitArgMax(self, node: AST.ArgMax, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitArgMax")
            self.indent += 1
        node.expr.gamma = dict(node.gamma)
        eType = self.visit(node.expr)

        node.dim.gamma = dict(node.gamma)
        dimType = self.visit(node.dim)
        assert isInt(dimType) or (isTensor(dimType) and (len(dimType.shape) == 0))

        node.type = Tensor(
            node.outputShape, eType.dataType, eType.bitlen, eType.isSecret, eType.taint
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitReduce(self, node: AST.Reduce, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitReduce")
            self.indent += 1
        cur_gamma = dict(node.gamma)
        node.expr.gamma = cur_gamma
        eType = self.visit(node.expr)

        node.type = Tensor(
            node.outShape, eType.dataType, eType.bitlen, eType.isSecret, eType.taint
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitInput(self, node: AST.Input, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitInput")
            self.indent += 1
        # print("Type.py : visitInput : Creating an input tensor")
        node.type = Tensor(
            node.shape,
            node.dataType,
            isSecret=node.isSecret,
            taint=Taints[node.inputByParty.name],
        )

        if self.debug:
            self.indent -= 1
        return node.type

    def visitOutput(self, node: AST.Output, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitOutput")
            self.indent += 1
        node.expr.gamma = dict(node.gamma)
        self.visit(node.expr)
        node.type = Unit()

        if self.debug:
            self.indent -= 1
        return node.type

    def visitFusedBatchNorm(self, node: AST.FusedBatchNorm, args=None):
        if self.debug:
            print(f"{' '*self.indent}||visitFusedBatchNorm")
            self.indent += 1
        cur_gamma = dict(node.gamma)
        node.expr.gamma = cur_gamma
        node.multExpr.gamma = cur_gamma
        node.addExpr.gamma = cur_gamma

        exprType = self.visit(node.expr)
        multExprType = self.visit(node.multExpr)
        addExprType = self.visit(node.addExpr)

        assert len(multExprType.shape) == 1
        assert len(addExprType.shape) == 1

        [C1] = multExprType.shape
        [C2] = addExprType.shape

        assert exprType.shape[-1] == C1 and C1 == C2

        taint = getTaint_taint(exprType.taint, multExprType.taint)
        taint = getTaint_taint(taint, addExprType.taint)

        # node.type = copy.copy(exprType)
        node.type = Tensor(exprType.shape, multExprType.dataType)
        node.type.taint = taint

        # print(f"Type.py : FusedBatchNorm : {multExprType.dataType}, {node.shape}")
        if self.debug:
            self.indent -= 1
        return node.type
