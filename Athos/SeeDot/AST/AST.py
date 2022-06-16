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

from enum import Enum, auto

OperatorsSymbolDict = {
    "ADD": "+",
    "SUB": "-",
    "DIV": "/",
    "ClearMemPublic": "clearmempublic",
    "ClearMemSecret": "clearmemsecret",
    "CONV": "#",
    "CONVTRANSPOSE": "#T",  # ConvTranspose
    "ElemWiseDiv": "./",
    "ElemWiseMul": ".*",
    "Equal": "==",
    "Floor": "floor",
    "Mean": "mean",
    "MUL": "*",
    "RELU": "relu",
    "RSQRT": "rsqrt",
    "Shape": "shape",
    "Unsqueeze": "unsqueeze",
    "Gather": "gather",
    "SIGMOID": "sigmoid",
    "HARDSIGMOID": "hardsigmoid",
    "SQRT": "sqrt",
    "SUB": "-",
    "TANH": "tanh",
}


class Party(Enum):
    SERVER = 0
    CLIENT = 1


class Operators(Enum):
    ADD = auto()
    SUB = auto()
    DIV = auto()
    MUL = auto()
    CONV = auto()
    CONVTRANSPOSE = auto()
    RELU = auto()
    TANH = auto()
    SIGMOID = auto()
    HARDSIGMOID = auto()
    SQRT = auto()
    RSQRT = auto()
    Equal = auto()
    ElemWiseMul = auto()
    ElemWiseDiv = auto()
    Floor = auto()
    Shape = auto()
    Unsqueeze = auto()
    Gather = auto()
    Mean = auto()
    ClearMemSecret = auto()
    ClearMemPublic = auto()

    def convSymbolToEnumValue(symbolStr):
        enumStr = None
        for k, v in OperatorsSymbolDict.items():
            if v == symbolStr:
                enumStr = k
        assert enumStr is not None
        return Operators[enumStr]

    def findConvTransposePadding(i, i_prime, f, p_total, stride):
        # The parameters have the following semantics:
        # 	i = conv input img size
        # 	i_prime = convTranspose input img Size
        # 	f = filter size
        # 	p_total = conv input padding total
        # 	stride = conv input stride
        p_total_tr = 2 * f - p_total - 2 + ((i + p_total - f) % stride)
        stride_tr = 1
        i_prime_tilde = i_prime + (i_prime - 1) * (stride - 1)
        return [p_total_tr, stride_tr, i_prime_tilde]

    def findLeftRightPaddingFromTotalPadding(totalPadding):
        leftPadding = totalPadding // 2
        rightPadding = totalPadding - leftPadding
        return [leftPadding, rightPadding]

    def findConvOutputImgSize(imgSize, totalPadding, filterSize, stride):
        return ((imgSize + totalPadding - filterSize) // stride) + 1


class PaddingKeysDict:
    ConvDim = 2  # 2D or 3D convolution, default to 2D
    # Also used for convTranpose
    FH = "FH"
    FW = "FW"
    FD = "FD"
    zPadHLeft = "zPadHLeft"
    zPadHRight = "zPadHRight"
    zPadWLeft = "zPadWLeft"
    zPadWRight = "zPadWRight"
    zPadDLeft = "zPadDLeft"
    zPadDRight = "zPadDRight"
    strideH = "strideH"
    strideW = "strideW"
    strideD = "strideD"
    inputImgH = "inputImgH"
    inputImgW = "inputImgW"
    inputImgD = "inputImgD"
    outputImgH = "outputImgH"
    outputImgW = "outputImgW"
    outputImgD = "outputImgD"
    paddingUsedStr = "paddingUsedStr"
    group = "group"


# If this is marked true, each astNode checks the types of its inputs to confirm it satisfies the assumption
# Turn this off to get speedup in compilation
assertInputTypes = True

# Represents expression. All other nodes are specific types of expr.
class ASTNode:
    mtdKeyTFOpName = "TFOpName"
    mtdKeyTFNodeName = "TFNodeName"

    def __init__(self):
        self.gamma = {}
        self.metadata = {}
        self.decls = {}
        self.depth = 0
        self.optidict = {}


class Int(ASTNode):
    def __init__(self, value: int, bitLen=None, isSecret=True, isScaled=False):
        if assertInputTypes:
            assert isinstance(value, int)
            if bitLen:
                assert isinstance(bitLen, int)
                assert (bitLen == 32) or (bitLen == 64)
            assert isinstance(isSecret, bool)
            assert isinstance(isScaled, bool)
        super().__init__()
        self.value = value
        self.bitLen = bitLen
        self.isSecret = isSecret
        self.isScaled = isScaled


class Float(ASTNode):
    def __init__(self, value: float, isSecret=True):
        if assertInputTypes:
            assert isinstance(value, float)
            assert isinstance(isSecret, bool)
        super().__init__()
        self.value = value
        self.isSecret = isSecret


class ID(ASTNode):
    def __init__(self, name: str):
        if assertInputTypes:
            assert isinstance(name, str)
        super().__init__()
        self.name = name


# shape : list of int, valueList : list of int/float AST Nodes
class Decl(ASTNode):
    def __init__(
        self, shape: list, dataType: str, valueList: list, isSecret=True, isScaled=False
    ):
        if assertInputTypes:
            for elem in shape:
                assert isinstance(elem, int)
            if dataType:
                assert isinstance(dataType, str)
            if valueList:
                for elem in valueList:
                    assert isinstance(elem, (Int, Float))
            assert isinstance(isSecret, bool)
            assert isinstance(isScaled, bool)
        super().__init__()
        self.shape = shape
        self.dataType = dataType
        self.valueList = valueList
        self.isSecret = isSecret
        self.isScaled = isScaled


# expr : ASTNode, perm : list of ints
class Transpose(ASTNode):
    def __init__(self, expr: ASTNode, perm: list = None):
        if assertInputTypes:
            assert isinstance(expr, ASTNode)
            if perm:
                for elem in perm:
                    assert isinstance(elem, int)
        super().__init__()
        self.expr = expr
        self.perm = perm


# expr : ASTNode, perm : list of ints
class Slice(ASTNode):
    def __init__(self, expr: ASTNode, subscriptRanges: list = None):
        if assertInputTypes:
            assert isinstance(expr, ID)
            if subscriptRanges:
                for elem in subscriptRanges:
                    assert isinstance(elem[0], int)
                    assert isinstance(elem[1], int)
        super().__init__()
        self.expr = expr
        self.subscriptRanges = subscriptRanges


# expr : ASTNode, shape : list of int, order : int : optional
class Reshape(ASTNode):
    def __init__(self, expr: ASTNode, shape: list, order: list):
        if assertInputTypes:
            assert isinstance(expr, ASTNode)
            for elem in shape:
                assert isinstance(elem, int)
            if order:
                for elem in order:
                    assert isinstance(elem, int)
        super().__init__()
        self.expr = expr
        self.shape = shape
        self.order = order


class Gather(ASTNode):
    def __init__(self, expr: ASTNode, shape: list, axis: int, index: int):
        if assertInputTypes:
            assert isinstance(expr, ASTNode)

            for elem in shape:
                assert isinstance(elem, int)

            assert isinstance(axis, int)
            assert isinstance(index, int)

        super().__init__()
        self.expr = expr
        self.shape = shape
        self.axis = axis
        self.index = index


# expr : ASTNode
# options : Other options required by maxpool
# 			Order: [FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW]
class Pool(ASTNode):
    class PoolType:
        MaxPool = "MaxPool"
        AvgPool = "AvgPool"

    def __init__(self, poolType: str, expr: ASTNode, options: dict):
        if assertInputTypes:
            assert (
                poolType == Pool.PoolType.MaxPool or poolType == Pool.PoolType.AvgPool
            )
            assert isinstance(expr, ASTNode)
            assert isinstance(options, dict)
            assert PaddingKeysDict.FH in options
            assert PaddingKeysDict.FW in options
            assert PaddingKeysDict.zPadHLeft in options
            assert PaddingKeysDict.zPadHRight in options
            assert PaddingKeysDict.zPadWLeft in options
            assert PaddingKeysDict.zPadWRight in options
            assert PaddingKeysDict.strideH in options
            assert PaddingKeysDict.strideW in options

        super().__init__()
        self.poolType = poolType
        self.expr = expr
        self.options = options


class UOp(ASTNode):
    def __init__(self, op: Operators, expr: ASTNode):
        if assertInputTypes:
            assert isinstance(op, Operators)
            assert isinstance(expr, ASTNode)
        super().__init__()
        self.op = op
        self.expr = expr


class BOp(ASTNode):
    # Options is used to convey extra info if the operator needs so
    # For example, it will be useful for convolution to convey strides etc.

    # IMPORTANT NOTE: The options parameter coming for ConvTranspose is for the conv of which it is an inverse

    def __init__(self, expr1: ASTNode, op: Operators, expr2: ASTNode, options=None):
        if assertInputTypes:
            assert isinstance(expr1, ASTNode)
            assert isinstance(op, Operators)
            assert isinstance(expr2, ASTNode)
            if options:
                assert isinstance(options, dict)
            if op == Operators.CONV or op == Operators.CONVTRANSPOSE:
                assert PaddingKeysDict.FH in options
                assert PaddingKeysDict.FW in options
                assert PaddingKeysDict.zPadHLeft in options
                assert PaddingKeysDict.zPadHRight in options
                assert PaddingKeysDict.zPadWLeft in options
                assert PaddingKeysDict.zPadWRight in options
                assert PaddingKeysDict.strideH in options
                assert PaddingKeysDict.strideW in options
                if PaddingKeysDict.ConvDim in options:
                    assert (
                        options[PaddingKeysDict.ConvDim] == 2
                        or options[PaddingKeysDict.ConvDim] == 3
                    )
                    if options[PaddingKeysDict.ConvDim] == 3:
                        # 3D conv - assert over the depth dimension
                        assert PaddingKeysDict.FD in options
                        assert PaddingKeysDict.zPadDLeft in options
                        assert PaddingKeysDict.zPadDRight in options
                        assert PaddingKeysDict.strideD in options
            if op == Operators.CONVTRANSPOSE:
                # In addition if this op is convTranspose, then
                # 	the output size should also be specified
                assert PaddingKeysDict.outputImgH in options
                assert PaddingKeysDict.outputImgW in options
                if (PaddingKeysDict.ConvDim in options) and (
                    options[PaddingKeysDict.ConvDim] == 3
                ):
                    assert PaddingKeysDict.outputImgD in options
        super().__init__()
        self.expr1 = expr1
        self.op = op
        self.expr2 = expr2
        self.options = options


class Func(ASTNode):
    def __init__(self, op: Operators, expr: ASTNode, **kwargs):
        if assertInputTypes:
            assert isinstance(op, Operators)
            assert isinstance(expr, ASTNode)
        super().__init__()
        self.op = op
        self.expr = expr
        for k, v in kwargs.items():
            if k == "alpha":
                self.alpha = v
            elif k == "beta":
                self.beta = v


class Unsqueeze(ASTNode):
    def __init__(self, expr: ID, shape: list, axis: int):
        if assertInputTypes:
            assert isinstance(expr, ID)
            assert isinstance(shape, list)

            for elem in shape:
                assert isinstance(elem, int)

            assert isinstance(axis, int)

        super().__init__()
        self.expr = expr
        self.shape = shape
        self.axis = axis


class Let(ASTNode):
    def __init__(self, name: ID, decl: ASTNode, expr: ASTNode):
        if assertInputTypes:
            assert isinstance(name, ID)
            assert isinstance(decl, ASTNode)
            assert isinstance(expr, ASTNode)
        super().__init__()
        self.name = name
        self.decl = decl
        self.expr = expr


# Assumption is that the output of this is always a tensor
# outputShape : list of int, funcName : str, argsList : list of ASTNodes
# isSecret : whether the output of this node is public or secret
# outputDiffInpDims = 0 => output only different input dims
# 					= 1 => always output input dims
# 					= 2 => never output input dims
# 					: NOTE this doesn't apply for function names
class UninterpFuncCall(ASTNode):
    def __init__(
        self,
        outputShape: list,
        funcName: str,
        argsList: list,
        isSecret=True,
        outputDiffInpDims=0,
    ):
        if assertInputTypes:
            for elem in outputShape:
                assert isinstance(elem, int)
            assert isinstance(funcName, str)
            for arg in argsList:
                assert isinstance(arg, ASTNode)
            assert isinstance(isSecret, bool)
            assert isinstance(outputDiffInpDims, int)
        super().__init__()
        self.outputShape = outputShape
        self.funcName = funcName
        self.argsList = argsList
        self.isSecret = isSecret
        self.outputDiffInpDims = outputDiffInpDims


class ArgMax(ASTNode):
    def __init__(self, outputShape: list, expr: ID, dim: ASTNode, inShape: list):
        if assertInputTypes:
            for elem in outputShape:
                assert isinstance(elem, int)
            assert isinstance(expr, ID)
            assert isinstance(dim, ASTNode)
            for elem in inShape:
                assert isinstance(elem, int)
        super().__init__()
        self.outputShape = outputShape
        self.expr = expr
        self.dim = dim
        self.inShape = inShape


class Reduce(ASTNode):
    def __init__(
        self,
        expr: ID,
        keepdims: bool,
        outShape: list,
        op: Operators,
        reductionAxesList: list,
    ):
        # keepdims is unused for now
        if assertInputTypes:
            assert isinstance(expr, ID)
            assert isinstance(keepdims, bool)
            assert isinstance(outShape, list)
            for elem in outShape:
                assert isinstance(elem, int)
            assert isinstance(op, Operators)
        super().__init__()
        self.expr = expr
        self.keepdims = keepdims
        self.outShape = outShape
        self.op = op
        self.reductionAxesList = reductionAxesList


# shape : list of int, dataType : ID
# NOTE: Though datatype is being passed to this function, the output code eventually only has
# 		int in the apt bitlen for which the whole compilation is done
# Also, take note of the last parameter - "inputByParty". This can be used to set the party which
# 	which will do the input for this variable. Defaults to SERVER.
class Input(ASTNode):
    def __init__(
        self, shape: list, dataType: str, isSecret=True, inputByParty=Party.SERVER
    ):
        if assertInputTypes:
            for elem in shape:
                assert isinstance(elem, int)
            assert isinstance(dataType, str)
            assert isinstance(inputByParty, Party)
            assert (
                inputByParty == Party.CLIENT or inputByParty == Party.SERVER
            )  # Right now EzPC supports input by two parties.
        super().__init__()
        self.shape = shape
        self.dataType = dataType
        self.isSecret = isSecret
        self.inputByParty = inputByParty


class Output(ASTNode):
    def __init__(self, expr: ASTNode, outputToParty=Party.CLIENT):
        if assertInputTypes:
            assert outputToParty in [Party.CLIENT, Party.SERVER]
        super().__init__()
        self.expr = expr
        self.outputToParty = outputToParty


# Since some optimizations are possible around batchnorm, keep this as an interpreted node
class FusedBatchNorm(ASTNode):
    def __init__(self, expr: ID, multExpr: ID, addExpr: ID):
        if assertInputTypes:
            assert isinstance(expr, ID)
            assert isinstance(multExpr, ID)
            assert isinstance(addExpr, ID)
        super().__init__()
        self.expr = expr
        self.multExpr = multExpr
        self.addExpr = addExpr
