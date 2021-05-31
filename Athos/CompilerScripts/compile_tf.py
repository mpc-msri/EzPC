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
import argparse
import os.path
import json
import sys

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import parse_config
import tf_graph_io
import grappler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "TFCompiler"))
import DumpTFMtData
from TFNodesAST import TFNodesAST


def save_graph_def(path_to_pb):
    if not os.path.exists(path_to_pb):
        sys.exit("Cannot find " + path_to_pb)
    gd = tf_graph_io.load_graph_def_pb(path_to_pb)
    DumpTFMtData.save_graphdef(gd)
    return


def get_graph_from(graph_def):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        return graph


def check_operation_exists(graph, op):
    op_list = [i.name for i in graph.get_operations()]
    return op in op_list


def tensors_exist(graph, tensor_names):
    op_list = [i.name for i in graph.get_operations()]
    for i in tensor_names:
        assert i in op_list, "input " + i + " does not exist in the graph"
    return True


def set_input_shapes(graph, input_t_info):
    tensor_names = input_t_info.keys()
    assert tensors_exist(graph, tensor_names)

    graph_def = graph.as_graph_def()
    inputs = [i for i in graph.get_operations() if i.type == "Placeholder"]

    input_map = {}
    with tf.Graph().as_default() as new_graph:
        for i in inputs:
            if i.name not in input_t_info:
                continue
            shape = input_t_info[i.name]
            input_map[i.name] = tf.compat.v1.placeholder(
                i.get_attr("dtype"), shape=shape, name=i.name
            )
        tf.import_graph_def(graph_def, input_map=input_map, name="")
        return new_graph


def get_tensor(graph, name):
    return graph.get_operation_by_name(name).outputs[0]


def infer_input_info(graph):
    input_t_info = {}
    inputs = [i for i in graph.get_operations() if i.type == "Placeholder"]
    for i in inputs:
        input_t = i.outputs[0]
        if input_t.shape.dims == None:
            inp_shape = []
        else:
            inp_shape = input_t.shape.as_list()
            assert None not in inp_shape, (
                "Placeholder node "
                + i.name
                + " has unknown shape. Please specify name and shape in config"
            )
        input_t_info[i.name] = inp_shape
    return input_t_info


def get_unsupported_ops(graph):
    ops = set([i.type for i in graph.get_operations()])
    ops.discard("Assign")
    unsupported_ops = []
    for op in ops:
        if not hasattr(TFNodesAST, op):
            unsupported_ops.append(op)
    return unsupported_ops


def get_op_names_from_tensors(tensor_names):
    op_names = []
    for name in tensor_names:
        if ":" in name:
            try:
                op_name, out_n = name.split(":")
                out_n = int(out_n)
            except:
                raise ValueError(
                    "The tensor name {} looks like a tensor name but is not a valid one".format(
                        name
                    )
                )
            op_names.append(op_name)
        else:
            op_names.append(name)
    return op_names


# Generates the computation graph and tensor size metadata and saves them in
# the model directory.
# Optionaly dumps model weights as fixedpt in specified scaling factor
def compile(model_fname, input_t_info, output_t_names, scaling_factor, save_weights):
    model_name = os.path.basename(model_fname)[:-3]
    print("Loading tf graph ", model_fname)
    graph = tf_graph_io.load_pb(model_fname)
    output_op_names = get_op_names_from_tensors(output_t_names)
    assert tensors_exist(graph, output_op_names)

    if input_t_info == {}:
        input_t_info = infer_input_info(graph)
    else:
        tensors_exist(graph, list(input_t_info.keys()))
        graph = set_input_shapes(graph, input_t_info)
    input_t_names = list(input_t_info.keys())
    graph_def = grappler.optimize(graph, input_t_names, output_op_names)
    graph_def = grappler.convert_consts_to_var(graph_def)
    graph = get_graph_from(graph_def)

    unsupported_ops = get_unsupported_ops(graph)
    if len(unsupported_ops) != 0:
        msg = (
            "Exiting compilation...\nCurrently we do not support the following ops: \n"
        )
        for i in unsupported_ops:
            msg = msg + "    " + i + "\n"
        sys.exit(msg)

    feed_dict = {}
    for name, shape in input_t_info.items():
        tensor = get_tensor(graph, name)
        zeros = np.zeros(shape)
        feed_dict[tensor] = zeros

    cwd = os.getcwd()
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Run initializers generated by preprocessing
            if check_operation_exists(graph, "init_constvars"):
                sess.run(graph.get_operation_by_name("init_constvars"))
            sess.run(tf.compat.v1.global_variables_initializer())
            model_dir = os.path.realpath(os.path.dirname(model_fname))
            os.chdir(model_dir)

            # At this stage the graph still has constants embedded in it
            # in the assign nodes for variables. We cannot execute the graph without
            # these constants. We strip them away in a new graph def which is amenable
            # to codegen but leave them in the graph.
            optimized_graph_def = DumpTFMtData.strip_variable_init_constants(
                graph_def, input_t_names, output_op_names
            )

            tf_graph_io.dump_graph_def_pb(
                optimized_graph_def, "optimised_" + model_name + ".pb"
            )
            DumpTFMtData.save_graphdef(optimized_graph_def)
            DumpTFMtData.save_sizeinfo(optimized_graph_def, sess, feed_dict)
            print("Model compilation done.")
            weights_path = ""
            if save_weights:
                weights_fname = (
                    model_name
                    + "_input_weights_fixedpt_scale_"
                    + str(scaling_factor)
                    + ".inp"
                )
                print(
                    "\nDumping model weights in ",
                    model_dir + "/" + weights_fname,
                    ".\nThese are to be used as input for party which owns the model\n",
                )
                DumpTFMtData.save_weights(
                    optimized_graph_def, sess, feed_dict, weights_fname, scaling_factor
                )
                weights_path = os.path.join(model_dir, weights_fname)
    os.chdir(cwd)
    return weights_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, type=str, help="Path to the config file"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = parse_config.get_params(args.config)
    compile(
        params["model_name"],
        params["input_tensors"],
        params["output_tensors"],
        params["scale"],
        params["save_weights"],
    )
