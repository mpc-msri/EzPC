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

import numpy
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph


def strip_variable_init_constants(graph_def, input_tensor_names, output_tensor_names):
    transforms = [
        "remove_nodes(op=Identity)",
        "strip_unused_nodes",
    ]
    # Sanity check if output/input nodes were constant and replaced with variables.
    all_node_names = set([i.name for i in graph_def.node])

    def get_true_names(tensor_names, all_nodes):
        real_names = []
        for i in tensor_names:
            if i not in all_nodes:
                var_name = i + "_mpc_const_var"
                if var_name in all_nodes:
                    real_names.append(var_name)
            else:
                real_names.append(i)
        return real_names

    real_input_names = get_true_names(input_tensor_names, all_node_names)
    real_output_names = get_true_names(output_tensor_names, all_node_names)
    optimized_graph_def = TransformGraph(
        graph_def, real_input_names, real_output_names, transforms
    )
    return optimized_graph_def


def save_graphdef(graph_def):
    with open("./graphDef.mtdata", "w") as f:
        f.write(str(graph_def))


def save_sizeinfo(optimized_graph_def, sess, feed_dict):
    # Save size information for tensors on which output depends
    tensors_to_evaluate = []
    tensors_to_evaluate_names = []
    graph = sess.graph
    for node in optimized_graph_def.node:
        output_number = 0
        for cur_output in graph.get_operation_by_name(node.name).outputs:
            tensors_to_evaluate.append(cur_output)
            if output_number == 0:
                tensor_name = node.name
            else:
                tensor_name = cur_output.name
            tensors_to_evaluate_names.append(tensor_name)
            output_number += 1
    tensors_evaluated = sess.run(tensors_to_evaluate, feed_dict)
    tensors_shape = list(map(lambda x: x.shape, tensors_evaluated))

    # Write size info in a file
    with open("./sizeInfo.mtdata", "w") as f:
        for ii, curr in enumerate(tensors_to_evaluate_names):
            curShape = tensors_shape[ii]
            f.write(tensors_to_evaluate_names[ii] + " ")
            for dim in curShape:
                f.write(str(dim) + " ")
            f.write("\n")


def save_graph_metadata(output_tensor, sess, feed_dict):
    graph_def = sess.graph_def
    transforms = [
        "remove_nodes(op=Identity)",
        "strip_unused_nodes",
        "fold_batch_norms",
        "fold_constants(ignore_errors=true)",
    ]
    optimized_graph_def = TransformGraph(
        graph_def, [], [output_tensor.name], transforms
    )
    with open("./graphDef.mtdata", "w") as f:
        f.write(str(optimized_graph_def))

    # Save size information for tensors on which output depends
    tensors_to_evaluate = []
    tensors_to_evaluate_names = []
    graph = sess.graph
    for node in optimized_graph_def.node:
        output_number = 0
        for cur_output in graph.get_operation_by_name(node.name).outputs:
            tensors_to_evaluate.append(cur_output)
            if output_number == 0:
                tensor_name = node.name
            else:
                tensor_name = cur_output.name
            tensors_to_evaluate_names.append(tensor_name)
            output_number += 1
    tensors_evaluated = sess.run(tensors_to_evaluate, feed_dict)
    tensors_shape = list(map(lambda x: x.shape, tensors_evaluated))

    # Write size info in a file
    with open("./sizeInfo.mtdata", "w") as f:
        for ii, curr in enumerate(tensors_to_evaluate_names):
            curShape = tensors_shape[ii]
            f.write(tensors_to_evaluate_names[ii] + " ")
            for dim in curShape:
                f.write(str(dim) + " ")
            f.write("\n")

    return optimized_graph_def


def updateWeightsForBN(optimized_graph_def, sess, feed_dict={}):
    graph = sess.graph
    graphDef = optimized_graph_def

    for node in graphDef.node:
        if node.op == "FusedBatchNorm" or node.op == "FusedBatchNormV3":
            gamma = graph.get_operation_by_name(node.input[1]).outputs[0]
            beta = graph.get_operation_by_name(node.input[2]).outputs[0]
            mu = graph.get_operation_by_name(node.input[3]).outputs[0]
            variance = graph.get_operation_by_name(node.input[4]).outputs[0]

            epsilon = node.attr["epsilon"].f
            rsigma = tf.rsqrt(variance + epsilon)

            sess.run(tf.assign(gamma, gamma * rsigma), feed_dict)
            sess.run(tf.assign(beta, beta - gamma * mu), feed_dict)
            sess.run(tf.assign(mu, tf.zeros(tf.shape(mu))), feed_dict)
            sess.run(
                tf.assign(variance, tf.fill(tf.shape(variance), 1 - epsilon)), feed_dict
            )


def dumpImageDataInt(imgData, filename, scalingFac, writeMode):
    print("Dumping image data...")
    with open(filename, writeMode) as ff:
        for xx in numpy.nditer(imgData, order="C"):
            ff.write(str(int(xx * (1 << scalingFac))) + " ")
        ff.write("\n\n")


def dumpTrainedWeightsInt(
    sess, evalTensors, filename, scalingFac, writeMode, alreadyEvaluated=False
):
    print("Dumping trained weights...")
    if alreadyEvaluated:
        finalParameters = evalTensors
    else:
        finalParameters = map(lambda x: sess.run(x), evalTensors)
    with open(filename, writeMode) as ff:
        for ii, curParameterVal in enumerate(finalParameters):
            for xx in numpy.nditer(curParameterVal, order="C"):
                ff.write(str(int(xx * (1 << scalingFac))) + " ")
            ff.write("\n\n")


def dumpTrainedWeightsFloat(
    sess, evalTensors, filename, writeMode, alreadyEvaluated=False
):
    print("Dumping trained weights float...")
    if alreadyEvaluated:
        finalParameters = evalTensors
    else:
        finalParameters = map(lambda x: sess.run(x), evalTensors)
    with open(filename, writeMode) as ff:
        for ii, curParameterVal in enumerate(finalParameters):
            for xx in numpy.nditer(curParameterVal, order="C"):
                ff.write((str(xx)) + " ")
            ff.write("\n\n")


def dumpImgAndWeightsData(
    sess, imgData, evalTensors, filename, scalingFac, alreadyEvaluated=False
):
    print("Starting to dump data...")
    dumpImageDataInt(imgData, filename, scalingFac, "w")
    dumpTrainedWeightsInt(
        sess, evalTensors, filename, scalingFac, "a", alreadyEvaluated=alreadyEvaluated
    )


def dumpImgAndWeightsDataSeparate(
    sess,
    imgData,
    evalTensors,
    imgFileName,
    weightFileName,
    scalingFac,
    alreadyEvaluated=False,
):
    print("Starting to dump data...")
    dumpImageDataInt(imgData, imgFileName, scalingFac, "w")
    dumpTrainedWeightsInt(
        sess,
        evalTensors,
        weightFileName,
        scalingFac,
        "w",
        alreadyEvaluated=alreadyEvaluated,
    )


def numpy_float_array_to_float_val_str(input_array):
    chunk = ""
    for val in numpy.nditer(input_array):
        chunk += str(val) + "\n"
    return chunk


def save_weights(optimized_graph_def, sess, feed_dict, filename, scaling_factor):
    graph = sess.graph
    varNames = [
        node.name
        for node in optimized_graph_def.node
        if node.op in ["VariableV2", "Variable"]
    ]
    graph_vars = [graph.get_operation_by_name(i).outputs[0] for i in varNames]
    updateWeightsForBN(optimized_graph_def, sess, feed_dict)
    values = sess.run(graph_vars, feed_dict)
    with open(filename, "w") as ff:
        for val in values:
            if val.shape.count(0) > 0:  # Empty array, nothing to dump.
                continue
            for xx in numpy.nditer(val, order="C"):
                ff.write(str(int(xx * (1 << scaling_factor))) + " ")
            ff.write("\n")
