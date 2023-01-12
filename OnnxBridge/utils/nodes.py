from onnx import ValueInfoProto, TensorProto

from utils import Party
from utils.onnx2IR_helper import (
    translate_onnx,
    convert_attribute_proto,
    proto_val_to_dimension_tuple,
    onnx2ir,
)


class Input:
    """
    Represents the Input Nodes from Onnx Model Graph.
    """

    def __init__(self, node):
        self.name = node.name
        self.is_secret = True
        self.op_type = "input"
        if isinstance(node, ValueInfoProto):  # input
            self.shape = list(proto_val_to_dimension_tuple(node))
            self.data_type = onnx2ir(node.type.tensor_type.elem_type)
            # When weights are stripped from the model by the server,
            # the doc_string field is set to this exact MPC_MODEL_WEIGHTS
            # magic keyword
            if node.doc_string == "MPC_MODEL_WEIGHTS":
                self.party = Party.BOB
            else:
                self.party = Party.ALICE
        elif isinstance(node, TensorProto):  # initializers
            self.shape = list(node.dims)
            self.data_type = onnx2ir(node.data_type)
            self.party = Party.BOB
        else:
            assert False, "Unexpected input type"

    def __str__(self):
        return f"Name: {self.name}\nShape: {self.shape}\nDataType: {self.data_type}\nParty: {self.party}"


class Node(object):
    """
    Represents the Operator Nodes from the Onnx Model Graph.
    """

    opset_version = -1

    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict(
            [
                (attr.name, translate_onnx(attr.name, convert_attribute_proto(attr)))
                for attr in node.attribute
            ]
        )
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node

    def __str__(self):
        return (
            f"Node Name: {self.name}\nOperator: {self.op_type}\nDomain: {self.domain}\n"
            f"Attributes: {self.attrs}\nInputs: {self.inputs}\nOutputs: {self.outputs}\n"
        )


class Output:
    """
    Represents the Output Nodes from the Onnx Model Graph.
    """

    def __init__(self, node):
        self.name = node.name
        self.op_type = "output"
        self.shape = list(proto_val_to_dimension_tuple(node))
        self.data_type = onnx2ir(node.type.tensor_type.elem_type)
        self.party = Party.ALICE

    def __str__(self):
        return f"Name: {self.name}\nShape: {self.shape}\nDataType: {self.data_type}\nParty: {self.party}"


def process_input_nodes(program, graph, var_dict):
    """
    Processes the Input Nodes from the graph and appends them to the program list of Nodes.
    :param program: Program list to append to.
    :param graph: Model Proto onnx graph.
    :param var_dict: Variable Dictionary.
    :return: Program List
    """
    input_nodes = [Input(i) for i in graph.input] + [
        Input(i) for i in graph.initializer
    ]
    if program is not None:
        program = program + input_nodes
    else:
        program = input_nodes
    return program


def process_func_nodes(program, graph, var_dict):
    """
    Processes the Operator Nodes from the graph and appends them to the program list of Nodes.
    :param program: Program list to append to.
    :param graph: Model Proto onnx graph.
    :param var_dict: Variable Dictionary.
    :return: Program List
    """
    func_nodes = [Node(i) for i in graph.node]
    program = program + func_nodes
    return program


def process_output_nodes(program, graph, var_dict):
    """
    Processes the Output Nodes from the graph and appends them to the program list of Nodes.
    :param program: Program list to append to.
    :param graph: Model Proto onnx graph.
    :param var_dict: Variable Dictionary.
    :return: Program List
    """
    output_nodes = [Output(i) for i in graph.output]
    program = program + output_nodes
    return program


def print_nodes(program):
    """
    Prints the Program List as function calls, useful for debugging.
    :param program: Program List
    :return: NA
    """
    for node in program:
        if isinstance(node, Input):
            print(f"input({node.name})")
        elif isinstance(node, Node):
            print(f"{node.op_type}({node.inputs})---->{node.outputs}")
        else:
            print(f"output({node.name})")
