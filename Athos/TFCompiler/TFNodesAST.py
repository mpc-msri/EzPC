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

import Graph
import AST.AST as AST
from enum import Enum, auto

# Contains code for each of the TF nodes encountered in the benchmarks.
# For each such TF node, outputs the corresponding SeeDot AST.


class TFNodesAST:
    class UninterpFuncCallNames(Enum):
        """
        NOTE : SeeDot when compiling uninterpreted function calls, adds a new declaration for each uninterpreted function call.
        """

        Input = auto()
        CreateCopy = auto()
        CreateIdentity = auto()
        CreateTensor = auto()
        CopyTensor = auto()
        Const = auto()
        Cast = auto()
        TruncatedNormal = auto()
        RandomUniform = auto()
        Tile = auto()
        MaxPool = auto()
        Pack = auto()
        Concat = auto()
        ExpandDims = auto()
        MaxPoolGrad = auto()
        Conv2DBackpropInput = auto()
        Conv2DBackpropFilter = auto()
        AvgPool = auto()
        Pad = auto()
        Squeeze = auto()

    def getOperatorsIdx(token):
        return AST.Operators.convSymbolToEnumValue(token)

    def MatMul(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        inp1Str = dictNodeNameToOutVarStr[inputsRef[0]]
        inp2Str = dictNodeNameToOutVarStr[inputsRef[1]]
        inp1AST = AST.ID(inp1Str)
        inp2AST = AST.ID(inp2Str)

        attrMapRef = curNode.getAttrMapRef()
        transposeABool = transposeBBool = False
        if "transpose_a" in attrMapRef:
            transposeABool = attrMapRef["transpose_a"].getB()
        if "transpose_b" in attrMapRef:
            transposeBBool = attrMapRef["transpose_b"].getB()
        if transposeABool:
            inp1AST = AST.Transp(inp1AST)
        if transposeBBool:
            inp2AST = AST.Transp(inp2AST)
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    inp1AST, TFNodesAST.getOperatorsIdx("*"), inp2AST
                )
            },
        )

    def Placeholder(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        curNodeShapeLi = extraNodeInfoDict[curNode.getName()][0]
        curNodeInputType = curNode.getAttrMapRef()["dtype"].getDataType()
        assert curNodeInputType is not Graph.DataTypeEnum.DT_INVALID

        # NOTE: There has to be some way for Athos to differentiate model from image, since in the compiled code
        # 	(in the scenario of secure inference), model is input by server and image by client.
        # 	We assume in the following that the PlaceHolder op node represents the image and
        # 	all model parameters are represented using Variable op nodes.
        # 	Hence, in the call to AST.Input, we pass inputByParty=1.

        return (
            None,
            {
                curNode.getName(): AST.Input(
                    curNodeShapeLi,
                    curNodeInputType.name,
                    isSecret=True,
                    inputByParty=AST.Party.CLIENT,
                )
            },
        )

    def Equal(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx("=="),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                )
            },
        )

    def Identity(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        # In SeeDot, J2=J1 creates a new reference for J1 -- so
        # 	the corresponding code in Seedot cannot simply be J2 = J1.
        # 	Instead create a new tensor first and then assign the old one to the new one.
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1

        curNodeDataType = curNode.getAttrMapRef()["T"].getDataType()
        assert curNodeDataType is not Graph.DataTypeEnum.DT_INVALID

        curNodeShape = extraNodeInfoDict[curNode.getName()][0]
        retAST = AST.UninterpFuncCall(
            curNodeShape,
            TFNodesAST.UninterpFuncCallNames.CreateIdentity.name,
            [AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])],
        )
        return (None, {curNode.getName(): retAST})

    def Add(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx("+"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                )
            },
        )

    def AddV2(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx("+"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                )
            },
        )

    def Mul(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx(".*"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                )
            },
        )

    def Neg(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.UOp(
                    TFNodesAST.getOperatorsIdx("-"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def Sub(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx("+"),
                    AST.UOp(
                        TFNodesAST.getOperatorsIdx("-"),
                        AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    ),
                )
            },
        )

    def Floor(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.Func(
                    TFNodesAST.getOperatorsIdx("floor"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def RealDiv(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx("./"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                )
            },
        )

    def FloorDiv(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        realDivAST = AST.BOp(
            AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
            TFNodesAST.getOperatorsIdx("./"),
            AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
        )
        return (
            None,
            {
                curNode.getName(): AST.Func(
                    TFNodesAST.getOperatorsIdx("floor"), realDivAST
                )
            },
        )

    def VariableV2(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        curNodeShapeLi = curNode.getAttrMapRef()["shape"].getShape().getDimRef()[:]
        curNodeInputType = curNode.getAttrMapRef()["dtype"].getDataType()

        # NOTE : since this becomes an input node right now, i have also added to be prefixed at top in ProcessTFGraph::prefixAllPlaceHolderNodes()
        # NOTE: There has to be some way for Athos to differentiate model from image, since in the compiled code
        # 	(in the scenario of secure inference), model is input by server and image by client.
        # 	We assume in the following that the PlaceHolder op node represents the image and
        # 	all model parameters are represented using Variable op nodes.
        # 	Hence, in the call to AST.Input, we pass inputByParty as SERVER.
        return (
            None,
            {
                curNode.getName(): AST.Input(
                    curNodeShapeLi,
                    curNodeInputType.name,
                    isSecret=True,
                    inputByParty=AST.Party.SERVER,
                )
            },
        )

    def Const(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        assert len(curNode.getInputsRef()) == 0
        tensor = curNode.getAttrMapRef()["value"].getTensor()
        curNodeDataType = curNode.getAttrMapRef()["dtype"].getDataType()
        curNodeShape = tensor.getShapeRef()[
            :
        ]  # create a different copy to not change the original copy

        tensorConstantVal = tensor.getConstantVal()
        if tensorConstantVal is not None:
            # Use uinterpreted call of CreateTensor to create the tensor and fill it with a constant value
            dataPassed = None
            if curNodeDataType == Graph.DataTypeEnum.DT_INT32:
                dataPassed = AST.Int(tensorConstantVal, 32, isSecret=False)
            elif curNodeDataType == Graph.DataTypeEnum.DT_FLOAT:
                dataPassed = AST.Float(tensorConstantVal, isSecret=False)
            else:
                assert False

            if len(curNodeShape) == 0:
                # This is a constant element
                retAST = dataPassed
            else:
                retAST = AST.UninterpFuncCall(
                    curNodeShape,
                    TFNodesAST.UninterpFuncCallNames.CreateTensor.name,
                    [dataPassed],
                    isSecret=False,
                )
        else:
            # The tensor content is given as byte array. Extract val array from the byte array and create ast.
            if curNodeDataType == Graph.DataTypeEnum.DT_INT32:
                dataPassed = list(
                    map(
                        lambda x: AST.Int(x, 32, isSecret=False),
                        tensor.getContentAsValArr()[:],
                    )
                )
            elif curNodeDataType == Graph.DataTypeEnum.DT_FLOAT:
                dataPassed = list(
                    map(
                        lambda x: AST.Float(x, isSecret=False),
                        tensor.getContentAsValArr()[:],
                    )
                )
            else:
                assert False
            retAST = AST.Decl(curNodeShape, None, dataPassed, isSecret=False)
        return (None, {curNode.getName(): retAST})

    def Relu(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.Func(
                    TFNodesAST.getOperatorsIdx("relu"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def Tanh(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.Func(
                    TFNodesAST.getOperatorsIdx("tanh"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def Sqrt(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.Func(
                    TFNodesAST.getOperatorsIdx("sqrt"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def Rsqrt(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.Func(
                    TFNodesAST.getOperatorsIdx("rsqrt"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def Sigmoid(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.Func(
                    TFNodesAST.getOperatorsIdx("sigmoid"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def Shape(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.Func(
                    TFNodesAST.getOperatorsIdx("shape"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def Cast(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        sourceType = curNode.getAttrMapRef()["SrcT"].getDataType()
        destType = curNode.getAttrMapRef()["DstT"].getDataType()
        return (
            None,
            {
                curNode.getName(): AST.UninterpFuncCall(
                    extraNodeInfoDict[curNode.getName()][0],
                    TFNodesAST.UninterpFuncCallNames.Cast.name,
                    [
                        AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                        AST.ID(sourceType.name),
                        AST.ID(destType.name),
                    ],
                )
            },
        )

    def ZerosLike(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        curNodeOutputType = curNode.getAttrMapRef()["T"].getDataType()
        assert curNodeOutputType is not Graph.DataTypeEnum.DT_INVALID
        retAST = AST.UninterpFuncCall(
            extraNodeInfoDict[curNode.getName()][0],
            TFNodesAST.UninterpFuncCallNames.CreateTensor.name,
            [AST.Int(0, isSecret=False)],
            isSecret=False,
        )
        return (None, {curNode.getName(): retAST})

    def Fill(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        curNodeOutputShape = extraNodeInfoDict[inputsRef[0]][0]
        assert (
            len(curNodeOutputShape) == 1
        )  # inputsRef[0] denotes a shape and should have a rank of 1

        curNodeOutputType = curNode.getAttrMapRef()["T"].getDataType()
        assert curNodeOutputType is not Graph.DataTypeEnum.DT_INVALID

        retAST = AST.UninterpFuncCall(
            extraNodeInfoDict[curNode.getName()][0],
            TFNodesAST.UninterpFuncCallNames.CreateTensor.name,
            [AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])],
            isSecret=False,
        )
        return (None, {curNode.getName(): retAST})

    def Reshape(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        # assert(len(inputsRef) == 2)
        return (
            None,
            {
                curNode.getName(): AST.Reshape(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    extraNodeInfoDict[curNode.getName()][0],
                    None,
                )
            },
        )

    def helper_findPadding(
        imgH,
        imgW,
        FH,
        FW,
        strideH,
        strideW,
        paddingUsedStr,
        imgD=None,
        FD=None,
        strideD=None,
    ):
        if imgD:
            assert FD
            assert strideD
        zPadHLeft = zPadHRight = zPadWLeft = zPadWRight = zPadDLeft = zPadDRight = -1
        if paddingUsedStr == "SAME":
            # Reference for following:
            #       https://web.archive.org/web/20171223022012/https://www.tensorflow.org/api_guides/python/nn
            totalPaddingH = totalPaddingW = totalPaddingD = 0

            if imgH % strideH == 0:
                totalPaddingH = max(FH - strideH, 0)
            else:
                totalPaddingH = max(FH - (imgH % strideH), 0)

            if imgW % strideW == 0:
                totalPaddingW = max(FW - strideW, 0)
            else:
                totalPaddingW = max(FW - (imgW % strideW), 0)

            if imgD:
                if imgD % strideD == 0:
                    totalPaddingD = max(FD - strideD, 0)
                else:
                    totalPaddingD = max(FD - (imgD % strideD), 0)

            zPadHLeft = totalPaddingH // 2
            zPadHRight = totalPaddingH - zPadHLeft

            zPadWLeft = totalPaddingW // 2
            zPadWRight = totalPaddingW - zPadWLeft

            zPadDLeft = totalPaddingD // 2
            zPadDRight = totalPaddingD - zPadDLeft

        elif paddingUsedStr == "VALID":
            zPadHLeft = zPadHRight = zPadWLeft = zPadWRight = zPadDLeft = zPadDRight = 0
        else:
            zPadHLeft = (
                zPadHRight
            ) = zPadWLeft = zPadWRight = zPadDLeft = zPadDRight = -1

        if imgD:
            return [zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight]
        else:
            return [zPadHLeft, zPadHRight, zPadWLeft, zPadWRight]

    def Conv2D(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2

        stridesUsed = curNode.getAttrMapRef()["strides"].getList().getILi()
        assert stridesUsed[0] == 1 and stridesUsed[3] == 1
        strideH = stridesUsed[1]
        strideW = stridesUsed[2]

        inputShape = extraNodeInfoDict[inputsRef[0]][0]
        imgH = inputShape[1]
        imgW = inputShape[2]

        filterShape = extraNodeInfoDict[inputsRef[1]][0]
        FH = filterShape[0]
        FW = filterShape[1]

        paddingUsedStr = curNode.getAttrMapRef()["padding"].getS()

        [zPadHLeft, zPadHRight, zPadWLeft, zPadWRight] = TFNodesAST.helper_findPadding(
            imgH, imgW, FH, FW, strideH, strideW, paddingUsedStr
        )

        options = {}
        options[AST.PaddingKeysDict.FH] = FH
        options[AST.PaddingKeysDict.FW] = FW
        options[AST.PaddingKeysDict.zPadHLeft] = zPadHLeft
        options[AST.PaddingKeysDict.zPadHRight] = zPadHRight
        options[AST.PaddingKeysDict.zPadWLeft] = zPadWLeft
        options[AST.PaddingKeysDict.zPadWRight] = zPadWRight
        options[AST.PaddingKeysDict.strideH] = strideH
        options[AST.PaddingKeysDict.strideW] = strideW
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx("#"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    options,
                )
            },
        )

    # A depthwise separable convolution is equivalent to a grouped convolution
    # with no. of groups = the no. of input channels (G=CI)
    # This however requires a reshape of the filter.
    # Regular filter is [ FH, FW, CI, CM (channel_multiplier)]
    # Doing depthwise conv results in CO = CI * CM
    # Grouped conv expects [FH, FW, CI/G, (CO/G)*G]
    # So we reshape to [FH, FW, 1, CI * CM]
    def DepthwiseConv2dNative(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        # Reshape of filter is done in simplifyGraph
        img_shape = extraNodeInfoDict[inputsRef[0]][0]
        in_channels = img_shape[3]  # NHWC
        groups = in_channels
        _, nodeToAST = TFNodesAST.Conv2D(
            graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict
        )
        nodeToAST[curNode.getName()].options[AST.PaddingKeysDict.group] = groups
        return (None, nodeToAST)

    def Conv3D(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2

        stridesUsed = curNode.getAttrMapRef()["strides"].getList().getILi()
        assert stridesUsed[0] == 1 and stridesUsed[4] == 1
        strideD = stridesUsed[1]
        strideH = stridesUsed[2]
        strideW = stridesUsed[3]

        inputShape = extraNodeInfoDict[inputsRef[0]][0]
        imgD = inputShape[1]
        imgH = inputShape[2]
        imgW = inputShape[3]

        filterShape = extraNodeInfoDict[inputsRef[1]][0]
        FD = filterShape[0]
        FH = filterShape[1]
        FW = filterShape[2]

        paddingUsedStr = curNode.getAttrMapRef()["padding"].getS()

        [
            zPadDLeft,
            zPadDRight,
            zPadHLeft,
            zPadHRight,
            zPadWLeft,
            zPadWRight,
        ] = TFNodesAST.helper_findPadding(
            imgH, imgW, FH, FW, strideH, strideW, paddingUsedStr, imgD, FD, strideD
        )

        options = {}
        options[AST.PaddingKeysDict.FD] = FD
        options[AST.PaddingKeysDict.FH] = FH
        options[AST.PaddingKeysDict.FW] = FW
        options[AST.PaddingKeysDict.zPadDLeft] = zPadDLeft
        options[AST.PaddingKeysDict.zPadDRight] = zPadDRight
        options[AST.PaddingKeysDict.zPadHLeft] = zPadHLeft
        options[AST.PaddingKeysDict.zPadHRight] = zPadHRight
        options[AST.PaddingKeysDict.zPadWLeft] = zPadWLeft
        options[AST.PaddingKeysDict.zPadWRight] = zPadWRight
        options[AST.PaddingKeysDict.strideD] = strideD
        options[AST.PaddingKeysDict.strideH] = strideH
        options[AST.PaddingKeysDict.strideW] = strideW
        options[AST.PaddingKeysDict.ConvDim] = 3
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx("#"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    options,
                )
            },
        )

    def Conv3DBackpropInputV2(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 3  # output_shape, filter, input

        stridesUsed = curNode.getAttrMapRef()["strides"].getList().getILi()
        assert stridesUsed[0] == 1 and stridesUsed[4] == 1
        strideD = stridesUsed[1]
        strideH = stridesUsed[2]
        strideW = stridesUsed[3]

        filterShape = extraNodeInfoDict[inputsRef[1]][0]
        FD = filterShape[0]
        FH = filterShape[1]
        FW = filterShape[2]

        inputShape = extraNodeInfoDict[inputsRef[2]][0]
        inputD = inputShape[1]
        inputH = inputShape[2]
        inputW = inputShape[3]

        outputShape = extraNodeInfoDict[curNode.getName()][0]
        outputD = outputShape[1]
        outputH = outputShape[2]
        outputW = outputShape[3]

        paddingUsedStr = curNode.getAttrMapRef()["padding"].getS()

        # Important: Using outputH and outputW in the below is not an error!
        # 			For convTranspose, the parameters passed in the node are of the conv of which this convTranspose is an inverse.
        # 			Which is why the call to helper_findPadding makes sense.
        # 			The zPads below are of the conv of which this convTranspose is an inverse.
        [
            zPadDLeft,
            zPadDRight,
            zPadHLeft,
            zPadHRight,
            zPadWLeft,
            zPadWRight,
        ] = TFNodesAST.helper_findPadding(
            outputH,
            outputW,
            FH,
            FW,
            strideH,
            strideW,
            paddingUsedStr,
            imgD=outputD,
            FD=FD,
            strideD=strideD,
        )

        options = {}
        options[AST.PaddingKeysDict.FD] = FD
        options[AST.PaddingKeysDict.FH] = FH
        options[AST.PaddingKeysDict.FW] = FW
        options[AST.PaddingKeysDict.zPadDLeft] = zPadDLeft
        options[AST.PaddingKeysDict.zPadDRight] = zPadDRight
        options[AST.PaddingKeysDict.zPadHLeft] = zPadHLeft
        options[AST.PaddingKeysDict.zPadHRight] = zPadHRight
        options[AST.PaddingKeysDict.zPadWLeft] = zPadWLeft
        options[AST.PaddingKeysDict.zPadWRight] = zPadWRight
        options[AST.PaddingKeysDict.strideD] = strideD
        options[AST.PaddingKeysDict.strideH] = strideH
        options[AST.PaddingKeysDict.strideW] = strideW
        options[AST.PaddingKeysDict.ConvDim] = 3
        options[AST.PaddingKeysDict.outputImgD] = outputD
        options[AST.PaddingKeysDict.outputImgH] = outputH
        options[AST.PaddingKeysDict.outputImgW] = outputW
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[2]]),
                    TFNodesAST.getOperatorsIdx("#T"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    options,
                )
            },
        )

    def helper_processPool(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
        typeOfPool: str,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1

        options = {}

        stridesUsed = curNode.getAttrMapRef()["strides"].getList().getILi()
        assert (stridesUsed[0] == 1) and (stridesUsed[3] == 1)
        strideH = stridesUsed[1]
        strideW = stridesUsed[2]

        kSizeUsed = curNode.getAttrMapRef()["ksize"].getList().getILi()
        assert (kSizeUsed[0] == 1) and (kSizeUsed[3] == 1)
        kSizeH = kSizeUsed[1]
        kSizeW = kSizeUsed[2]

        inputShape = extraNodeInfoDict[inputsRef[0]][0]
        imgH = inputShape[1]
        imgW = inputShape[2]

        paddingUsedStr = curNode.getAttrMapRef()["padding"].getS()
        [zPadHLeft, zPadHRight, zPadWLeft, zPadWRight] = TFNodesAST.helper_findPadding(
            imgH, imgW, kSizeH, kSizeW, strideH, strideW, paddingUsedStr
        )

        poolType = None
        if typeOfPool == "MAXPOOL":
            poolType = AST.Pool.PoolType.MaxPool
        elif typeOfPool == "AVGPOOL":
            poolType = AST.Pool.PoolType.AvgPool
        else:
            print("Unknown type of pooling layer.", file=sys.stderr)
            assert False
        return (
            None,
            {
                curNode.getName(): AST.Pool(
                    poolType,
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    {
                        AST.PaddingKeysDict.FH: kSizeH,
                        AST.PaddingKeysDict.FW: kSizeW,
                        AST.PaddingKeysDict.zPadHLeft: zPadHLeft,
                        AST.PaddingKeysDict.zPadHRight: zPadHRight,
                        AST.PaddingKeysDict.zPadWLeft: zPadWLeft,
                        AST.PaddingKeysDict.zPadWRight: zPadWRight,
                        AST.PaddingKeysDict.strideH: strideH,
                        AST.PaddingKeysDict.strideW: strideW,
                    },
                )
            },
        )

    def MaxPool(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        return TFNodesAST.helper_processPool(
            graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict, "MAXPOOL"
        )

    def AvgPool(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        return TFNodesAST.helper_processPool(
            graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict, "AVGPOOL"
        )

    def ConcatV2(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        N = curNode.getAttrMapRef()["N"].getI()
        assert len(inputsRef) == N + 1  # One extra for axis
        # TODO : Since the axis of concat is constant, therefore, its known here - the input's sizes along that dim should be
        # 		passed as input to the below function.
        # 		For now hardcoding.
        retAST = AST.UninterpFuncCall(
            extraNodeInfoDict[curNode.getName()][0],
            TFNodesAST.UninterpFuncCallNames.Concat.name + str(N) + "T",
            list(map(lambda x: AST.ID(dictNodeNameToOutVarStr[x]), inputsRef)),
            outputDiffInpDims=1,
        )
        return (None, {curNode.getName(): retAST})

    def ExpandDims(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        retAST = AST.UninterpFuncCall(
            extraNodeInfoDict[curNode.getName()][0],
            TFNodesAST.UninterpFuncCallNames.ExpandDims.name,
            list(map(lambda x: AST.ID(dictNodeNameToOutVarStr[x]), inputsRef)),
        )
        return (None, {curNode.getName(): retAST})

    def Slice(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 3
        beginNode = graph.__getitem__(inputsRef[1])
        sizeNode = graph.__getitem__(inputsRef[2])
        assert (
            beginNode.getAttrVal("value") is not None
        ), "begin {} of Slice node {} has to be a constant".format(
            inputsRef[1], curNode.getName()
        )
        assert (
            sizeNode.getAttrVal("value") is not None
        ), "size {} of Slice node {} has to be a constant".format(
            inputsRef[2], curNode.getName()
        )
        begin = beginNode.getAttrVal("value").getTensor().getContentAsValArr()
        size = sizeNode.getAttrVal("value").getTensor().getContentAsValArr()
        assert begin is not None
        assert size is not None
        assert len(begin) == len(size)
        subscriptRanges = []
        for i in range(0, len(size)):
            subscriptRanges.append((begin[i], begin[i] + size[i] - 1))

        return (
            None,
            {
                curNode.getName(): AST.Slice(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]), subscriptRanges
                )
            },
        )

    def Tile(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.UninterpFuncCall(
                    extraNodeInfoDict[curNode.getName()][0],
                    TFNodesAST.UninterpFuncCallNames.Tile.name,
                    [
                        AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                        AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    ],
                )
            },
        )

    def Sum(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        attrMapRef = curNode.getAttrMapRef()
        assert len(inputsRef) == 2
        keepdims = False
        if "keep_dims" in attrMapRef:
            keepdims = attrMapRef["keep_dims"].getB()

        reductionAxesNodeName = inputsRef[1]
        redAxesN = graph.__getitem__(reductionAxesNodeName)
        redAxesT = redAxesN.getAttrVal("value").getTensor()
        rank = redAxesT.getShapeRef().getRank()
        if rank != 0:
            reductionAxesList = redAxesT.getContentAsValArr()
        else:
            reductionAxesList = [redAxesT.getConstantVal()]

        curNodeShapeLi = extraNodeInfoDict[curNode.getName()][0]
        return (
            None,
            {
                curNode.getName(): AST.Reduce(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    keepdims,
                    curNodeShapeLi,
                    TFNodesAST.getOperatorsIdx("+"),
                    reductionAxesList,
                )
            },
        )

    def Mean(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        attrMapRef = curNode.getAttrMapRef()
        assert len(inputsRef) == 2
        keepdims = False
        if "keep_dims" in attrMapRef:
            keepdims = attrMapRef["keep_dims"].getB()

        reductionAxesNodeName = inputsRef[1]
        redAxesN = graph.__getitem__(reductionAxesNodeName)
        redAxesT = redAxesN.getAttrVal("value").getTensor()
        rank = redAxesT.getShapeRef().getRank()
        if rank != 0:
            reductionAxesList = redAxesT.getContentAsValArr()
        else:
            reductionAxesList = [redAxesT.getConstantVal()]

        curNodeShapeLi = extraNodeInfoDict[curNode.getName()][0]
        return (
            None,
            {
                curNode.getName(): AST.Reduce(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    keepdims,
                    curNodeShapeLi,
                    TFNodesAST.getOperatorsIdx("mean"),
                    reductionAxesList,
                )
            },
        )

    def ArgMax(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.ArgMax(
                    extraNodeInfoDict[curNode.getName()][0],
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    extraNodeInfoDict[inputsRef[0]][0],
                )
            },
        )

    def NoOp(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        return (None, {curNode.getName(): None})

    def Square(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 1
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx(".*"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                )
            },
        )

    def Pad(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        # Mode refers to 'CONSTANT', 'REFLECT' or 'SYMMETRIC'
        mode = 0
        if "mode" in curNode.getAttrMapRef():
            mode = curNode.getAttrMapRef()["mode"].getI()

        constant_values = 0
        if "constant_values" in curNode.getAttrMapRef():
            constant_values = curNode.getAttrMapRef()["constant_values"].getI()

        assert (
            mode == 0 and constant_values == 0
        )  # For now to make life easy - deal with SYMMETRIC AND REFLECT when time comes
        inputsRef = curNode.getInputsRef()
        inputTensorShapeLi = extraNodeInfoDict[inputsRef[0]][0]
        return (
            None,
            {
                curNode.getName(): AST.UninterpFuncCall(
                    extraNodeInfoDict[curNode.getName()][0],
                    TFNodesAST.UninterpFuncCallNames.Pad.name,
                    [
                        AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                        AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    ],
                    outputDiffInpDims=1,
                )
            },
        )

    def FusedBatchNorm(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        return (
            None,
            {
                curNode.getName(): AST.FusedBatchNorm(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[2]]),
                )
            },
        )

    def FusedBatchNormV3(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        return (
            None,
            {
                curNode.getName(): AST.FusedBatchNorm(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[2]]),
                )
            },
        )

    def Transpose(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        permNodeName = inputsRef[1]
        # We need to fetch the tensor value of the perm Node
        permNode = graph.__getitem__(permNodeName)
        permTensor = permNode.getAttrVal("value").getTensor()
        permList = permTensor.getContentAsValArr()
        assert permTensor.getDType().kind == "i"
        assert permTensor.getShapeRef().getRank() == 1
        return (
            None,
            {
                curNode.getName(): AST.Transpose(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]), permList
                )
            },
        )

    def Split(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        axisNodeName = inputsRef[
            0
        ]  # split_dim input. Has to be a constant. We don't support dynamic codegen yet
        axisNode = graph.__getitem__(axisNodeName)
        axisTensor = axisNode.getAttrVal("value").getTensor()
        axis = axisTensor.getConstantVal()
        numSplits = curNode.getAttrVal("num_split").getI()
        inputTensorShape = extraNodeInfoDict[inputsRef[1]][0]
        assert axis < len(inputTensorShape)
        assert inputTensorShape[axis] % numSplits == 0  # Should perfectly split
        sizeAlongSplitDim = int(inputTensorShape[axis] / numSplits)
        outputAsts = {}
        for i in range(0, numSplits):
            output_name = curNode.getName()
            if i != 0:
                output_name += ":" + str(i)
            subscriptRanges = []
            for j in range(0, len(inputTensorShape)):
                start = 0
                end = inputTensorShape[j] - 1
                if j == axis:
                    start = i * sizeAlongSplitDim
                    end = start + sizeAlongSplitDim - 1
                subscriptRanges.append((start, end))
            outputAsts[output_name] = AST.Slice(
                AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]), subscriptRanges
            )
        return (None, outputAsts)

    def Squeeze(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        inputTensorShape = extraNodeInfoDict[inputsRef[0]][0]
        inputTensorRank = len(inputTensorShape)

        squeezeDims = curNode.getAttrMapRef()["squeeze_dims"].getList().getILi()
        squeezeDimsRank = len(squeezeDims)

        return (
            None,
            {
                curNode.getName(): AST.UninterpFuncCall(
                    extraNodeInfoDict[curNode.getName()][0],
                    TFNodesAST.UninterpFuncCallNames.Squeeze.name,
                    list(map(lambda x: AST.Int(x, 32, isSecret=False), squeezeDims))
                    + [AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])],
                )
            },
        )

    def BiasAdd(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        assert len(inputsRef) == 2
        return (
            None,
            {
                curNode.getName(): AST.BOp(
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                    TFNodesAST.getOperatorsIdx("+"),
                    AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                )
            },
        )

    def ReadVariableOp(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        return (
            None,
            {curNode.getName(): AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])},
        )

    def Softmax(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        return (
            None,
            {curNode.getName(): AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])},
        )

    def StopGradient(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        inputsRef = curNode.getInputsRef()
        return (
            None,
            {curNode.getName(): AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])},
        )

    def VarHandleOp(
        graph: Graph.Graph,
        curNode: Graph.Node,
        dictNodeNameToOutVarStr: dict,
        extraNodeInfoDict: dict,
    ):
        return TFNodesAST.VariableV2(
            graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict
        )
