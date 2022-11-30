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
        assert len(node.inputs) == 1
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
        # Nothing to assert yet
        logger.debug("Concat is OK! (No assertions)")

    @classmethod
    def Sigmoid(cls, node):
        assert len(node.inputs) == 1
        logger.debug("Sigmoid is OK!")

    @classmethod
    def HardSigmoid(cls, node):
        assert len(node.inputs) == 1
        logger.debug("Hard Sigmoid is OK!")

    @classmethod
    def Relu(cls, node):
        assert len(node.inputs) == 1
        logger.debug("Relu is OK!")

    @classmethod
    def Div(cls, node):
        # todo is div there or not? in athos it takes one input
        pass

    @classmethod
    def Add(cls, node):
        assert len(node.inputs) == 2
        logger.debug("Add is OK!")

    @classmethod
    def Sub(cls, node):
        assert len(node.inputs) == 2
        logger.debug("Sub is OK!")

    @classmethod
    def Mul(cls, node):
        assert len(node.inputs) == 2
        logger.debug("Mul is OK!")

    @classmethod
    def Gather(cls, node):
        # Nothing to assert yet
        logger.debug("Concat is OK! (No assertions)")

    @classmethod
    def Gemm(cls, node):
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
        # Nothing to assert yet
        logger.debug("Concat is OK! (No assertions)")

    @classmethod
    def Transpose(cls, node):
        assert len(node.inputs) == 1
        logger.debug("Transpose is OK!")

    @classmethod
    def Split(cls, node):
        node.inputs = node.inputs[:1]
        logger.debug("Split is OK! (with possible modifications)")

    @classmethod
    def ReduceMean(cls, node):
        keepdims = node.attrs["keepdims"]
        axes = node.attrs["axes"]

        # currently handling only this case
        # currently support only 0 case
        assert keepdims == 0
        assert len(axes) == 2
        assert len(node.inputs) == 1
        del node.attrs["keepdims"]
        logger.debug("ReduceMean is OK! (with possible modifications)")

    @classmethod
    def MatMul(cls, node):
        # todo transpose and mul
        assert len(node.inputs) == 2
        logger.debug("MatMul is OK!")

    @classmethod
    def BatchNormalization(cls, node):
        assert len(node.inputs) == 5
        node.inputs = node.inputs[:3]
        logger.debug("Batch Normalization is OK! (with possible modifications)")

    @classmethod
    def Unsqueeze(cls, node):
        pass

    @classmethod
    def Reshape(cls, node):
        pass

    @classmethod
    def Flatten(cls, node):
        assert len(node.inputs) == 1

    @classmethod
    def Conv(cls, node):
        pass

    @classmethod
    def MaxPool(cls, node):
        pass

    @classmethod
    def AveragePool(cls, node):
        pass

    @classmethod
    def GlobalAveragePool(cls, node):
        pass

    @classmethod
    def ConvTranspose(cls, node):
        pass

    @classmethod
    def LeakyRelu(cls, node):
        if "alpha" not in node.attributes:
            node.attributes["alpha"] = 0.01

    @classmethod
    def Tanh(cls, node):
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        # we can print the node at this step and get info on all node parameters
        # additionaly based on your node implementation add assertions or modification on node attributes.
        logger.debug("Tanh is OK!")
