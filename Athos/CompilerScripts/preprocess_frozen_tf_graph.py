"""

Authors: Pratik Bhatu.

Copyright:
Copyright (c) 2021 Microsoft Research
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
from tf_graph_io import *
from tf_graph_trans import *
import sys
import time
import os

import argparse
import os.path

# Transpose nodes require perm as compile time constants for parametric codegen
# So we don't eliminate the constants we need dring compile time
def get_const_names(graph):
    transp_perm_ops = set(
        i.inputs[1].op.name for i in graph.get_operations() if i.type == "Transpose"
    )
    padding_ops = set(
        i.inputs[1].op.name for i in graph.get_operations() if i.type == "Pad"
    )
    slice_begin_ops = set(
        i.inputs[1].op.name for i in graph.get_operations() if i.type == "Slice"
    )
    slice_size_ops = set(
        i.inputs[2].op.name for i in graph.get_operations() if i.type == "Slice"
    )
    mean_axes_ops = set(
        i.inputs[1].op.name for i in graph.get_operations() if i.type == "Mean"
    )
    split_dim_ops = set(
        i.inputs[0].op.name for i in graph.get_operations() if i.type == "Split"
    )
    white_list = (
        transp_perm_ops
        | padding_ops
        | slice_begin_ops
        | slice_size_ops
        | mean_axes_ops
        | split_dim_ops
    )
    all_const_ops = set(i.name for i in graph.get_operations() if i.type == "Const")
    return list(all_const_ops - white_list)


def check_operation_exists(graph, tensor_name):
    op_list = [i.name for i in graph.get_operations()]
    return tensor_name in op_list


def optimize(input_fname, output_t_name):
    if not input_fname.endswith(".pb"):
        sys.exit("Please supply a valid tensorflow protobuf model (.pb extension)")

    actual_fname = os.path.basename(input_fname)
    dirname = os.path.dirname(input_fname)
    output_fname = os.path.join(dirname, "mpc_processed_" + actual_fname)
    print("Loading ", input_fname, "for processing.")
    graph = load_pb(input_fname)

    if not check_operation_exists(graph, output_t_name):
        sys.exit(output_t_name + " output does not exist in the graph")
    input_names = [i.name for i in graph.get_operations() if i.type == "Placeholder"]

    # graph = remove_dead_nodes(graph, input_names, [output_t_name])

    print(
        "\n\nThis process will take some time to run as we execute portions of the graph.\n\n"
    )
    time.sleep(1)
    # Fold away all static computations

    print("Running fold splits")
    graph = fold_splits(graph)
    print(graph.get_operations(), end="\n\n")
    print("Running constant folding")
    graph = fold_constants(graph)

    # Convert constants to variables so as to separate the data and the generated code
    # Otherwise huge arrays will show up as constants in the generated code, thereby
    # increasing binary size.
    print("Convert frozen constants to variables")
    graph = convert_consts_to_var(graph, get_const_names(graph))

    input_names = [i.name for i in graph.get_operations() if i.type == "Placeholder"]
    # graph = remove_dead_nodes(graph, input_names, [output_t_name])

    # At this stage the graph still has constants embedded in it
    # in the assign nodes for variables. We cannot execute the graph without
    # these constants. However after inferring the size, we can call remove_dead_nodes
    # to optimize away the constants and assign nodes and make the graph amenable
    # for codegen
    dump_pb(graph, output_fname)
    print("The processed graph is dumped in ", output_fname)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelName", required=True, type=str, help="Name of tensorflow model (*.pb)"
    )
    parser.add_argument(
        "--outputTensorName",
        required=True,
        type=str,
        help="Name of the output tensor for the model. (Op name, dont add '/:0' suffix)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    optimize(args.modelName, args.outputTensorName)
