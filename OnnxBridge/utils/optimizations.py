import os
import struct

import math
import numpy as np
from onnx import ValueInfoProto, TensorShapeProto, helper
from onnx import numpy_helper
from onnx import shape_inference
from onnx.helper import make_tensor_value_info
from onnxsim import simplify

from utils import logger
from utils.onnx2IR_helper import proto_val_to_dimension_tuple


def get_data_type(proto_val):
    return proto_val.type.tensor_type.elem_type


def numpy_float_array_to_float_val_str_nchw(input_array):
    chunk = ""
    for val in np.nditer(input_array):
        chunk += str(val) + "\n"
    return chunk


def numpy_float_array_to_float_val_str_nhwc(input_array):
    chunk = []
    if len(input_array.shape) == 4:
        co, ci, h, w = input_array.shape
        arr = np.zeros([co, h, w, ci])
        for i in range(co):
            for j in range(ci):
                for k in range(h):
                    for l in range(w):
                        arr[i][k][l][j] = input_array[i][j][k][l]
        input_array = arr
    elif len(input_array.shape) == 2:
        co, ci = input_array.shape
        arr = np.zeros([ci, co])
        for i in range(co):
            for j in range(ci):
                arr[j][i] = input_array[i][j]
        input_array = arr
    return input_array


def preprocess_batch_normalization(graph_def, model_name_to_val_dict):
    # set names to graph nodes if not present
    for node in graph_def.node:
        node.name = node.output[0]
        # Update the batch normalization scale and B
        # so that mean and var are not required
        if node.op_type == "BatchNormalization":
            # scale
            gamma = model_name_to_val_dict[node.input[1]]
            # B
            beta = model_name_to_val_dict[node.input[2]]
            mean = model_name_to_val_dict[node.input[3]]
            var = model_name_to_val_dict[node.input[4]]
            for i in range(len(gamma)):
                rsigma = 1 / math.sqrt(var[i] + 1e-5)
                gamma[i] = gamma[i] * rsigma
                beta[i] = beta[i] - gamma[i] * mean[i]
                mean[i] = 0
                var[i] = 1 - 1e-5

    # Just testing if the correct values are put
    model_name_to_val_dict2 = {}
    for init_vals in graph_def.initializer:
        # TODO: Remove float_data
        model_name_to_val_dict2[init_vals.name] = init_vals.float_data
    for node in graph_def.node:
        node.name = node.output[0]
        if node.op_type == "BatchNormalization":
            mean = model_name_to_val_dict[node.input[3]]
            for val in mean:
                assert val == 0


def dump_model_weights_as_inp(model, model_dir, model_name):
    """
    Dumps the Model Weights to a file.
    :param model: Onnx Model
    :param model_dir: Model Directory
    :param model_name: Model Name
    :return: Path to saved Model Weights
    """
    weights_path = ""
    weights_fname = model_name + "_input_weights_.inp"
    weights_path = os.path.join(model_dir, weights_fname)

    # needed because initializers are not in sequential order and we need to strip them and dump in file
    exclude = [
        val for node in model.graph.node for val in node.output
    ]  # list to store variables that are not initializers
    exclude.append(
        model.graph.input[0].name
    )  # because we want to exclude input in initializers
    initializers = [
        inp for node in model.graph.node for inp in node.input if inp not in exclude
    ]

    model_name_to_val_dict = {
        init_vals.name: numpy_helper.to_array(init_vals).tolist()
        for init_vals in model.graph.initializer
    }
    preprocess_batch_normalization(model.graph, model_name_to_val_dict)

    chunk_n = ""
    for init_name in initializers:
        chunk_1 = numpy_float_array_to_float_val_str_nchw(
            np.asarray(model_name_to_val_dict[init_name], dtype=np.float32)
        )
        chunk_n += chunk_1

    f = open(weights_path, "w")
    f.write(chunk_n)
    f.close()
    return weights_path


def dump_model_weights_as_dat(model, model_dir, model_name):
    """
    Dumps the Model Weights to a file.
    :param model: Onnx Model
    :param model_dir: Model Directory
    :param model_name: Model Name
    :return: Path to saved Model Weights
    """
    weights_path = ""
    weights_fname = model_name + "_input_weights.dat"
    weights_path = os.path.join(model_dir, weights_fname)
    f = open(weights_path, "wb")

    # needed because initializers are not in sequential order and we need to strip them and dump in file
    exclude = [
        val for node in model.graph.node for val in node.output
    ]  # list to store variables that are not initializers
    exclude.append(
        model.graph.input[0].name
    )  # because we want to exclude input in initializers
    initializers = [
        inp for node in model.graph.node for inp in node.input if inp not in exclude
    ]

    model_name_to_val_dict = {
        init_vals.name: numpy_helper.to_array(init_vals).tolist()
        for init_vals in model.graph.initializer
    }
    preprocess_batch_normalization(model.graph, model_name_to_val_dict)

    chunk_n = ""
    for init_name in initializers:
        chunk_1 = numpy_float_array_to_float_val_str_nhwc(
            np.asarray(model_name_to_val_dict[init_name], dtype=np.float32)
        )
        for val in np.nditer(chunk_1):
            f.write(struct.pack("f", float(val)))

    f.close()
    return weights_path


def strip_weights(model):
    """
    Makes all the initializers as inputs, and saves them to a new file.
    :param model: Model to be stripped.
    :return: Stripped Model.
    """
    graph = model.graph

    # Outputs remain same
    new_outputs = list(graph.output)
    # Nodes remain same
    new_nodes = list(graph.node)

    # We replace all initializers with input nodes.
    new_initializers = []
    new_inputs = list(graph.input)
    for node in graph.initializer:
        input = ValueInfoProto()
        input.name = node.name
        # Magic keyword for input nodes belonging to server
        input.doc_string = "MPC_MODEL_WEIGHTS"
        input.type.tensor_type.elem_type = node.data_type
        for size in node.dims:
            dim = TensorShapeProto.Dimension()
            dim.dim_value = size
            input.type.tensor_type.shape.dim.append(dim)
        new_inputs.append(input)

    new_graph = helper.make_graph(
        new_nodes,
        graph.name,
        new_inputs,
        new_outputs,
        initializer=new_initializers,
        doc_string=graph.doc_string,
        value_info=graph.value_info,
    )
    new_model = helper.make_model(
        new_graph,
        ir_version=model.ir_version,
        doc_string=model.doc_string,
        model_version=model.model_version,
        domain=model.domain,
        producer_name="MPCWeightStripper",
    )
    new_model.metadata_props.extend(model.metadata_props)
    new_model.opset_import.pop()
    new_model.opset_import.extend(model.opset_import)
    return new_model


def relu_maxpool_optimiser(program):
    """
    Optimises the Onnx Model by replacing the order where MaxPool appears after Relu.
    :param program: Onnx Model as a list of nodes
    :return: Optimised Program
    """
    for idx, node in enumerate(program):
        if node.op_type == "Relu" and program[idx + 1].op_type == "MaxPool":
            relu = program[idx]
            maxpool = program[idx + 1]

            relu.inputs, maxpool.inputs = maxpool.inputs, relu.inputs
            relu.outputs, maxpool.outputs = maxpool.outputs, relu.outputs

            program[idx] = maxpool
            program[idx + 1] = relu

    return program


def optimise(model):
    """
    Simplifies the Onnx Model, function provided by Onnx.
    :param model: Onnx Model
    :return: Optimized Simplified Model
    """
    optimized_model, check = simplify(model)
    assert check, "Optimised ONNX model failed validation"
    return optimized_model


def check_batch_size(model):
    """
    Returns the batch size .
    :param model: Onnx Model
    :return: batch size
    """
    return model.graph.input[0].type.tensor_type.shape.dim[0].dim_value


# This does constant folding and eliminates nodes like Shape.
# Also annotates each node with shape information.
def infer_shapes(model):
    logger.debug("Before Shape inference *****************")
    logger.debug(model.graph.value_info)

    for input in model.graph.input:
        model.graph.value_info.append(
            make_tensor_value_info(
                input.name,
                get_data_type(input),
                proto_val_to_dimension_tuple(input),
            )
        )

    for output in model.graph.output:
        model.graph.value_info.append(
            make_tensor_value_info(
                output.name,
                get_data_type(output),
                proto_val_to_dimension_tuple(output),
            )
        )

    for init_vals in model.graph.initializer:
        model.graph.value_info.append(
            make_tensor_value_info(
                init_vals.name, init_vals.data_type, tuple(init_vals.dims)
            )
        )

    inferred_model = shape_inference.infer_shapes(model)

    logger.debug("After Shape inference  ******************")
    logger.debug(inferred_model.graph.value_info)

    return inferred_model
