from utils import logger
from utils.nodes import Input, Output


class OnnxNode:
    """
    Class having function for all Onnx Nodes to check it is supported by our backend through assertions.
    """

    @classmethod
    def input(cls, node):
        assert isinstance(node, Input)
        logger.debug("Input is OK!")

    @classmethod
    def output(cls, node):
        assert isinstance(node, Output)
        logger.debug("Output is OK!")

    @classmethod
    def Cast(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        logger.debug("Cast type is", node.attrs["to"])
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Cast is OK!")

    @classmethod
    def Pad(cls, node):
        assert node.attrs["mode"] == "constant"

        if node.opset_version >= 11:
            # input: data, pads, constant_value
            # attrs: mode

            # Skip constant_val input (last input)
            assert len(node.inputs) >= 2
            node.inputs = node.inputs[:2]

        else:
            # input: data
            # attrs: mode, pads, value
            assert node.attrs["value"] == 0
            assert len(node.inputs) == 1
            # todo: check attr pad
        logger.debug("Pad is OK! (with possible modifications)")

    @classmethod
    def Concat(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        logger.debug("Concat axis is", node.attrs["axis"])
        assert node.opset_version <= 10
        logger.debug("Concat is OK! (No assertions)")

    @classmethod
    def Sigmoid(cls, node):
        assert len(node.inputs) == 1
        assert node.opset_version <= 10
        logger.debug("Sigmoid is OK!")

    @classmethod
    def HardSigmoid(cls, node):
        assert len(node.inputs) == 1
        assert node.opset_version <= 10
        logger.debug("Hard Sigmoid is OK!")

    @classmethod
    def Relu(cls, node):
        assert len(node.inputs) == 1
        assert node.opset_version <= 10
        logger.debug("Relu is OK!")

    @classmethod
    def Div(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Div is OK!")

    @classmethod
    def Add(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Add is OK!")

    @classmethod
    def Sub(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Sub is OK!")

    @classmethod
    def Mul(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Mul is OK!")

    @classmethod
    def Gather(cls, node):
        # TODO: Support Gather for multiple axes and non-zero axis.
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        assert "axes" not in node.attrs
        if "axis" in node.attrs:
            assert node.attrs["axis"] == 0
            logger.debug("Gather axis is", node.attrs["axis"])
        logger.debug("Gather is OK!")

    @classmethod
    def Gemm(cls, node):
        assert node.opset_version <= 10
        # todo transpose done separately in gemm
        if "alpha" not in node.attrs:
            node.attrs["alpha"] = 1.0
        if "beta" not in node.attrs:
            node.attrs["beta"] = 1.0
        if "transA" not in node.attrs:
            node.attrs["transA"] = 0
        if "transB" not in node.attrs:
            node.attrs["transB"] = 0

    @classmethod
    def Constant(cls, node):
        assert node.opset_version <= 10
        logger.debug("Concat is OK!")

    @classmethod
    def Transpose(cls, node):
        # TODO; Add support for more permutations and input shapes as needed.
        logger.debug("In node", node.name, " of type", node.op_type)
        logger.debug("Transpose permutations are", node.attrs["perm"])
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Transpose is OK!")

    @classmethod
    def ReduceMean(cls, node):
        # TODO: Add support for non-keepdims and multiple axes
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert len(node.attrs["axes"]) == 1
        if "keepdims" in node.attrs:
            assert node.attrs["keepdims"] == 1
        assert node.opset_version <= 10
        logger.debug("ReduceMean is OK!")

    @classmethod
    def MatMul(cls, node):
        # TODO: Add support for transposing operands.
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("MatMul is OK!")

    @classmethod
    def BatchNormalization(cls, node):
        assert len(node.inputs) == 5
        assert node.opset_version <= 10
        node.inputs = node.inputs[:3]
        logger.debug("Batch Normalization is OK! (with possible modifications)")

    @classmethod
    def Unsqueeze(cls, node):
        # TODO: Make Unsqueeze more generalized.
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert len(node.attrs["axes"]) == 1
        assert node.opset_version <= 10
        logger.debug("Unsqueeze is OK!")

    @classmethod
    def Reshape(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Reshape is OK!")

    @classmethod
    def Flatten(cls, node):
        assert len(node.inputs) == 1
        assert node.opset_version <= 10

    @classmethod
    def Conv(cls, node):
        assert node.opset_version <= 10
        pass

    @classmethod
    def MaxPool(cls, node):
        assert node.opset_version <= 10
        pass

    @classmethod
    def AveragePool(cls, node):
        assert node.opset_version <= 10
        pass

    @classmethod
    def GlobalAveragePool(cls, node):
        assert node.opset_version <= 10
        pass

    @classmethod
    def ConvTranspose(cls, node):
        assert node.opset_version <= 10
        pass

    @classmethod
    def LeakyRelu(cls, node):
        assert node.opset_version <= 10
        if "alpha" not in node.attributes:
            node.attributes["alpha"] = 0.01

    @classmethod
    def Tanh(cls, node):
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        # we can print the node at this step and get info on all node parameters
        # additionaly based on your node implementation add assertions or modification on node attributes.
        logger.debug("Tanh is OK!")

    @classmethod
    def Sqrt(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Sqrt is OK!")

    @classmethod
    def Pow(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Pow is OK!")

    @classmethod
    def Split(cls, node):
        # TODO: Add support for multiple axes and outputs.
        logger.debug("Split axis is", node.attrs["axis"])
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 1
        assert len(node.outputs) == 3
        assert len(node.attrs["split"]) == 3
        assert node.attrs["axis"] == 2
        assert node.opset_version <= 10
        logger.debug("Split is OK!")

    @classmethod
    def Slice(cls, node):
        # TODO: Add support for more general slice.
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 5
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Slice is OK!")

    @classmethod
    def Shape(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Shape is OK!")

    @classmethod
    def Squeeze(cls, node):
        # TODO: Add support for multiple axes.
        logger.debug("In node", node.name, " of type", node.op_type)
        logger.debug("Squeeze axes are", node.attrs["axes"])
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert len(node.attrs["axes"]) == 1
        assert node.opset_version <= 10
        logger.debug("Squeeze is OK!")

    @classmethod
    def NonZero(cls, node):
        # TODO: Add support for non-keepdims and multiple axes
        logger.debug("In node", node.name, " of type", node.op_type)
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("NonZero is OK!")

    @classmethod
    def ConstantOfShape(cls, node):
        # TODO: Add support for additional datatypes and multiple axes.
        logger.debug("In node", node.name, " of type", node.op_type)
        logger.debug("ConstantOfShape values are", node.attrs["value"])
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.attrs['value'].data_type == 7
        assert node.opset_version <= 10
        logger.debug("ConstantOfShape is OK!")

    @classmethod
    def Softmax(cls, node):
        logger.debug("In node", node.name, " of type", node.op_type)
        logger.debug("Softmax axis is", node.attrs["axis"])
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.opset_version <= 10
        logger.debug("Softmax is OK!")
