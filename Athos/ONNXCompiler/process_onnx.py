"""
Authors: Shubham Ugare.
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

# Add SeeDot directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "SeeDot"))
sys.path.append(os.path.dirname(__file__))

# For this warning: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import _pickle as pickle
import onnx
import onnx.shape_inference
from onnx import numpy_helper
from onnxsim import simplify

import AST.AST as AST

from ONNXNodesAST import ONNXNodesAST, OnnxNode
from onnx.helper import make_tensor_value_info
from onnx import TensorProto, ValueInfoProto
from AST.PrintAST import PrintAST
from AST.MtdAST import MtdAST
import numpy

import common
import math

import numpy as np

np.set_printoptions(threshold=np.inf)

DEBUG = False
out_var_prefix = "J"


def get_unsupported_ops(model):
    ops = set([i.op_type for i in model.graph.node])
    # ops.discard("Assign")
    unsupported_ops = []
    for op in ops:
        if not hasattr(ONNXNodesAST, op):
            unsupported_ops.append(op)
    return unsupported_ops


def exitIfUnsupportedOps(model):
    unsupported_ops = get_unsupported_ops(model)
    if len(unsupported_ops) != 0:
        msg = (
            "Exiting compilation...\nCurrently we do not support the following ops: \n"
        )
        for i in unsupported_ops:
            msg = msg + "    " + i + "\n"
        sys.exit(msg)


# This does constant folding and eliminates nodes like Shape.
# Also annotates each node with shape information.
def optimise(model):
    optimized_model, check = simplify(model)
    assert check, "Optimised ONNX model failed validation"
    return optimized_model


def inferShapes(model):
    if DEBUG:
        print(model.graph.value_info)

    for input in model.graph.input:
        model.graph.value_info.append(
            make_tensor_value_info(
                input.name,
                common.get_data_type(input),
                common.proto_val_to_dimension_tuple(input),
            )
        )

    for output in model.graph.output:
        model.graph.value_info.append(
            make_tensor_value_info(
                output.name,
                common.get_data_type(output),
                common.proto_val_to_dimension_tuple(output),
            )
        )

    if DEBUG:
        print(model.graph.value_info)

    for init_vals in model.graph.initializer:
        model.graph.value_info.append(
            make_tensor_value_info(
                init_vals.name, init_vals.data_type, tuple(init_vals.dims)
            )
        )

    if DEBUG:
        print("Shape inference *****************")
        print(model.graph.value_info)

    inferred_model = onnx.shape_inference.infer_shapes(model)

    if DEBUG:
        print("Printing shape ******************")
        print(inferred_model.graph.value_info)
        print("Done ******************")

    return inferred_model


def get_node_metadata(model):
    value_info = {}
    for val in model.graph.value_info:
        value_info[val.name] = (
            val.type.tensor_type.elem_type,
            common.proto_val_to_dimension_tuple(val),
        )
    return value_info


def generate_seedot_ast(model, value_info, model_dir):
    graph_def = model.graph
    # Iterate through the ONNX graph nodes and translate them to SeeDot AST nodes
    program = None
    innermost_let_ast_node = None
    node_name_to_out_var_dict = {}
    out_var_count = 0
    mtdAST = MtdAST()

    (program, innermost_let_ast_node, out_var_count) = process_input_variables(
        program,
        innermost_let_ast_node,
        node_name_to_out_var_dict,
        out_var_count,
        mtdAST,
        graph_def,
        value_info,
    )

    process_onnx_nodes(
        innermost_let_ast_node,
        node_name_to_out_var_dict,
        out_var_count,
        mtdAST,
        graph_def,
        value_info,
    )

    output_tensors = [i.name for i in graph_def.output]
    addOutputs(
        output_tensors,
        innermost_let_ast_node,
        node_name_to_out_var_dict,
        mtdAST,
        value_info,
    )

    if DEBUG:
        PrintAST().visit(program)
        common.write_debug_info(node_name_to_out_var_dict)

    with open(os.path.join(model_dir, "astOutput.pkl"), "wb") as f:
        pickle.dump(program, f)


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


def dump_model_weights(model, scaling_factor, model_dir, model_name):
    weights_path = ""
    weights_fname = (
        model_name + "_input_weights_fixedpt_scale_" + str(scaling_factor) + ".inp"
    )
    weights_path = os.path.join(model_dir, weights_fname)

    model_name_to_val_dict = {
        init_vals.name: numpy_helper.to_array(init_vals).tolist()
        for init_vals in model.graph.initializer
    }

    preprocess_batch_normalization(model.graph, model_name_to_val_dict)

    chunk_n = ""
    cnt_n = 0
    for init_vals in model.graph.initializer:
        (chunk_1, cnt_1) = common.numpy_float_array_to_fixed_point_val_str(
            np.asarray(model_name_to_val_dict[init_vals.name], dtype=np.float32),
            scaling_factor,
        )
        chunk_n += chunk_1
        cnt_n += cnt_1

    print(
        "\nDumping model weights in ",
        weights_path,
        ".\nThese are to be used as input for party which owns the model\n",
    )
    f = open(weights_path, "w")
    f.write(chunk_n)
    f.close()
    return weights_path


# Generates the computation graph and tensor size metadata and saves them in
# the model directory.
# Optionaly dumps model weights as fixedpt in specified scaling factor
def compile(model_fname, input_t_info, output_t_names, scaling_factor, save_weights):
    sys.setrecursionlimit(10000)
    if not model_fname.endswith(".onnx"):
        sys.exit("Please supply a valid ONNX model (.onnx extension)")

    model_name = os.path.basename(model_fname)[:-5]
    model_abs_dir = os.path.dirname(os.path.abspath(model_fname))
    print("Loading onnx graph: ", model_fname)
    model = onnx.load(model_fname)
    OnnxNode.opset_version = model.opset_import[0].version
    graph_def = model.graph

    model = optimise(model)
    model = inferShapes(model)

    # Check after optimisation of model removes nodes like Shape
    exitIfUnsupportedOps(model)

    if DEBUG:
        print("Printing shape ******************")
        print(model.graph.value_info)
        print("Done ******************")

    # value_info: { name : (type, dimension tuple) }
    value_info = get_node_metadata(model)

    generate_seedot_ast(model, value_info, model_abs_dir)

    if save_weights:
        return dump_model_weights(model, scaling_factor, model_abs_dir, model_name)
    return


def main():
    sys.setrecursionlimit(10000)
    # First read the ONNX file
    if len(sys.argv) < 2:
        print("TF python file unspecified.", file=sys.stderr)
        exit(1)
    file_name = sys.argv[1]
    file_path = "models/" + file_name
    model_name = file_name[:-5]  # name without the '.onnx' extension

    # load the model and extract the graph
    model = onnx.load(file_path)
    OnnxNode.opset_version = model.opset_import[0].version
    graph_def = model.graph

    print(model.graph.value_info)
    # Before shape inference (model.graph.value_info) should have shapes of all the variables and constants
    model.graph.value_info.append(
        make_tensor_value_info(
            model.graph.input[0].name,
            TensorProto.FLOAT,
            common.proto_val_to_dimension_tuple(model.graph.input[0]),
        )
    )
    model.graph.value_info.append(
        make_tensor_value_info(
            model.graph.output[0].name,
            TensorProto.FLOAT,
            common.proto_val_to_dimension_tuple(model.graph.output[0]),
        )
    )

    print(model.graph.value_info)

    for init_vals in model.graph.initializer:
        model.graph.value_info.append(
            make_tensor_value_info(
                init_vals.name, TensorProto.FLOAT, tuple(init_vals.dims)
            )
        )

    if DEBUG:
        print("Shape inference *****************")
        print(model.graph.value_info)

    inferred_model = onnx.shape_inference.infer_shapes(model)

    if DEBUG:
        print("Printing shape ******************")
        print(inferred_model.graph.value_info)
        print("Done ******************")

    # value_info: {name : (type, dimension tuple) }
    value_info = {}
    for val in inferred_model.graph.value_info:
        value_info[val.name] = (
            val.type.tensor_type.elem_type,
            common.proto_val_to_dimension_tuple(val),
        )

    # Iterate through the ONNX graph nodes and translate them to SeeDot AST nodes
    program = None
    innermost_let_ast_node = None
    node_name_to_out_var_dict = {}
    out_var_count = 0
    mtdAST = MtdAST()

    (program, innermost_let_ast_node, out_var_count) = process_input_variables(
        program,
        innermost_let_ast_node,
        node_name_to_out_var_dict,
        out_var_count,
        mtdAST,
        graph_def,
        value_info,
    )

    process_onnx_nodes(
        innermost_let_ast_node,
        node_name_to_out_var_dict,
        out_var_count,
        mtdAST,
        graph_def,
        value_info,
    )

    output_tensors = [i.name for i in graph_def.output]
    addOutputs(
        output_tensors,
        innermost_let_ast_node,
        node_name_to_out_var_dict,
        mtdAST,
        value_info,
    )

    PrintAST().visit(program)

    common.write_debug_info(node_name_to_out_var_dict)

    with open("debug/" + model_name + "/" + model_name + ".pkl", "wb") as f:
        pickle.dump(program, f)
        print("Dumped SeeDot AST")


def addOutputs(
    output_tensors,
    innermost_let_ast_node,
    node_name_to_out_var_dict,
    mtdAST,
    value_info,
):
    lastLetASTNode = innermost_let_ast_node
    while True:
        if type(lastLetASTNode.expr) is AST.Let:
            lastLetASTNode = lastLetASTNode.expr
        else:
            break
    assert lastLetASTNode is not None
    if output_tensors is None or len(output_tensors) == 0:
        assert False, "Onnx model has no outputs specified"
    else:
        outVarCt = 0
        outVarPrefix = "O"
        for i in range(0, len(output_tensors)):  # name, decl, expr
            t_name = output_tensors[i]
            if i == len(output_tensors) - 1:
                output_name = AST.ID(node_name_to_out_var_dict[t_name])
                output = AST.Output(output_name, AST.Party.CLIENT)
                newNode = output
            else:
                output_name = AST.ID(node_name_to_out_var_dict[t_name])
                output = AST.Output(output_name, AST.Party.CLIENT)
                let_name_id = AST.ID(outVarPrefix + str(outVarCt))
                newNode = AST.Let(let_name_id, output, AST.ASTNode())
                mtdForCurAST = {
                    AST.ASTNode.mtdKeyTFOpName: "Output",
                    AST.ASTNode.mtdKeyTFNodeName: t_name,
                }
                mtdAST.visit(newNode, mtdForCurAST)
            lastLetASTNode.expr = newNode
            lastLetASTNode = newNode
            outVarCt += 1


def process_input_variables(
    program,
    innermost_let_ast_node,
    node_name_to_out_var_dict,
    out_var_count,
    mtdAST,
    graph_def,
    value_info,
):
    class Input:
        def __init__(self, node):
            self.name = node.name
            if isinstance(node, ValueInfoProto):  # input
                self.shape = list(common.proto_val_to_dimension_tuple(node))
                self.data_type = node.type.tensor_type.elem_type
                self.party = AST.Party.CLIENT
            elif isinstance(node, TensorProto):  # initializers
                self.shape = list(node.dims)
                self.data_type = node.data_type
                self.party = AST.Party.SERVER
            else:
                assert False, "Unexpected input type"

        def __str__(self):
            return "Name: {n}, Shape: {s}, DataType: {dt}, Party: {p}".format(
                n=self.name, s=self.shape, dt=self.data_type, p=self.party
            )

    input_nodes = [Input(i) for i in graph_def.input] + [
        Input(i) for i in graph_def.initializer
    ]

    for node in input_nodes:
        if DEBUG:
            print("Node information")
            print(node)

        curAst = ONNXNodesAST.Input(node, value_info, node_name_to_out_var_dict)
        mtdForCurAST = {
            AST.ASTNode.mtdKeyTFOpName: "Input",
            AST.ASTNode.mtdKeyTFNodeName: node.name,
        }
        if curAst is None:
            continue

        cur_out_var_ast_node = AST.ID(node.name)

        if program:
            assert type(innermost_let_ast_node) is AST.Let
            newNode = AST.Let(cur_out_var_ast_node, curAst, cur_out_var_ast_node)
            mtdAST.visit(newNode, mtdForCurAST)
            # Updating the innermost Let AST node and the expression for previous Let Node
            innermost_let_ast_node.expr = newNode
            innermost_let_ast_node = newNode
        else:
            innermost_let_ast_node = AST.Let(
                cur_out_var_ast_node, curAst, cur_out_var_ast_node
            )
            mtdAST.visit(innermost_let_ast_node, mtdForCurAST)
            innermost_let_ast_node.depth = 0
            program = innermost_let_ast_node

        node_name_to_out_var_dict[node.name] = node.name
    return (program, innermost_let_ast_node, out_var_count)


def process_onnx_nodes(
    innermost_let_ast_node,
    node_name_to_out_var_dict,
    out_var_count,
    mtdAST,
    graph_def,
    value_info,
):
    for node in graph_def.node:
        if DEBUG:
            print("Node information")
            print(node)
            print("Processing " + node.name + "\n")
        mtdForCurAST = {
            AST.ASTNode.mtdKeyTFOpName: node.op_type,
            AST.ASTNode.mtdKeyTFNodeName: node.name,
        }

        func = getattr(ONNXNodesAST, node.op_type)
        (innermost_let_ast_node, out_var_count) = func(
            node,
            value_info,
            node_name_to_out_var_dict,
            innermost_let_ast_node,
            out_var_count,
            mtdAST,
        )

        assert type(innermost_let_ast_node) is AST.Let


if __name__ == "__main__":
    main()
