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
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.python.platform import gfile
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.grappler import cluster
from tensorflow.compat.v1 import GraphKeys
from tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


def get_graph_from(graph_def):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        return graph


def get_default_config():
    c = tf_optimizer.config_pb2.ConfigProto()
    optimizer_opts = c.graph_options.rewrite_options
    OFF = RewriterConfig.Toggle.Value("OFF")
    optimizer_opts.layout_optimizer = OFF
    optimizer_opts.implementation_selector = OFF
    optimizer_opts.min_graph_nodes = -1
    optimizer_opts.meta_optimizer_iterations = 2
    optimizer_opts.memory_optimization = RewriterConfig.MemOptType.Value("NO_MEM_OPT")
    return c


def get_only_prune_config():
    c = get_default_config()
    optimizer_opts = c.graph_options.rewrite_options
    OFF = RewriterConfig.Toggle.Value("OFF")
    optimizer_opts.constant_folding = OFF
    optimizer_opts.shape_optimization = OFF
    optimizer_opts.remapping = OFF
    optimizer_opts.arithmetic_optimization = OFF
    optimizer_opts.dependency_optimization = OFF
    optimizer_opts.loop_optimization = OFF
    optimizer_opts.function_optimization = OFF
    optimizer_opts.meta_optimizer_iterations = 1
    return c


def get_white_list(graph):
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
    sum_axes_ops = set(
        i.inputs[1].op.name for i in graph.get_operations() if i.type == "Sum"
    )
    split_dim_ops = set(
        i.inputs[0].op.name for i in graph.get_operations() if i.type == "Split"
    )
    concat_axes_ops = set(
        i.inputs[2].op.name
        for i in graph.get_operations()
        if i.type == "ConcatV2" or i.type == "Concat"
    )
    argmax_axes_ops = set(
        i.inputs[1].op.name for i in graph.get_operations() if i.type == "ArgMax"
    )
    divisor_ops = set(
        i.inputs[1].op.name
        for i in graph.get_operations()
        if i.type in ["FloorDiv", "RealDiv"]
    )

    white_list = (
        transp_perm_ops
        | padding_ops
        | slice_begin_ops
        | slice_size_ops
        | mean_axes_ops
        | sum_axes_ops
        | split_dim_ops
        | concat_axes_ops
        | argmax_axes_ops
        | divisor_ops
    )
    return list(white_list)


def optimize(g, inputs, outputs):
    sd = SignatureDef()
    for name in inputs:
        input_t = g.get_operation_by_name(name).outputs[0]
        sd.inputs[name].name = name
        sd.inputs[name].dtype = input_t.dtype.as_datatype_enum
        sd.inputs[name].tensor_shape.CopyFrom(input_t.shape.as_proto())
    for name in outputs:
        output_t = g.get_operation_by_name(name).outputs[0]
        sd.outputs[name].name = name
        sd.outputs[name].dtype = output_t.dtype.as_datatype_enum
        sd.outputs[name].tensor_shape.CopyFrom(output_t.shape.as_proto())

    tf.compat.v1.enable_resource_variables()
    cl = cluster.Cluster(disable_detailed_stats=True)

    # We have to run this twice to eliminate constants that are left after
    # optimising away split/pad/transpose nodes. They are const parameters like
    # axis, perm. They remain after 1 iter of optimization because we specify them
    # in the whitelist
    for i in range(2):
        if i == 0:
            graph = g
            c = get_default_config()
        else:
            graph = get_graph_from(optimized_graph_def)
            c = get_only_prune_config()

        white_list = get_white_list(graph)
        for name in white_list:
            graph.add_to_collection(
                GraphKeys.TRAIN_OP, graph.get_operation_by_name(name)
            )

        meta_graph = tf.compat.v1.train.export_meta_graph(
            graph_def=graph.as_graph_def(), graph=graph
        )
        meta_graph.signature_def["not_used_key"].CopyFrom(sd)

        optimized_graph_def = tf_optimizer.OptimizeGraph(
            config_proto=c, metagraph=meta_graph, cluster=cl
        )
    # Don't create VarHandleOp, ReadVariableOp, VarIsInitializedOp
    # Instead create VariableV2 ops in the future
    tf.disable_resource_variables()
    return optimized_graph_def


def delete_nodes(gd, ops):
    nodes_to_delete = set(op.name for op in ops)
    new_gd = tf.compat.v1.GraphDef()
    nodes_to_keep = []
    for n in gd.node:
        if not n.name in nodes_to_delete:
            nodes_to_keep.append(n)
    new_gd.node.extend(nodes_to_keep)
    return new_gd


def convert_consts_to_var(graph_def):
    graph = get_graph_from(graph_def)
    all_const_ops = set(i.name for i in graph.get_operations() if i.type == "Const")
    const_names_list = list(all_const_ops - set(get_white_list(graph)))
    const_var_names_pairs = []
    ops_to_delete = []
    with graph.as_default():
        preexisting_vars = [
            tf.get_variable(i.name, i.outputs[0].shape)
            for i in graph.get_operations()
            if i.type == "VariableV2" or i.type == "Variable"
        ]

        var_list = []
        for name in const_names_list:
            tensor = graph.get_operation_by_name(name).outputs[0]
            with tf.compat.v1.Session() as sess:
                t_value = sess.run(tensor)
            t_name = "{}_mpc_const_var".format(name)
            var = tf.compat.v1.Variable(t_value, name=t_name)
            var_read_op_name = var.to_proto().snapshot_name[:-2]
            const_var_names_pairs.append((name, var_read_op_name))
            var_list.append(var)

        for const_name, var_read_name in const_var_names_pairs:
            const_op = graph.get_operation_by_name(const_name)
            var_op = graph.get_operation_by_name(var_read_name)
            ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_op))
            ops_to_delete.append(const_op)

        tf.compat.v1.variables_initializer(
            var_list + preexisting_vars, "init_constvars"
        )
    return delete_nodes(graph.as_graph_def(), ops_to_delete)
