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
import tensorflow as tf
from tensorflow.python.platform import gfile
import os.path
import sys


def display_graph(graph, tensorboard_log_dir):
    writer = tf.summary.FileWriter(tensorboard_log_dir, graph)
    writer.close()


def load_graph_def_pb(path_to_pb):
    if not os.path.isfile(path_to_pb):
        sys.exit(path_to_pb + " file does not exist")
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def load_pb(path_to_pb):
    if not os.path.isfile(path_to_pb):
        sys.exit(path_to_pb + " file does not exist")
    graph_def = load_graph_def_pb(path_to_pb)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        return graph


def dump_pb(graph, filename):
    with tf.io.gfile.GFile(filename, "wb") as f:
        graph_def = graph.as_graph_def()
        f.write(graph_def.SerializeToString())


def dump_graph_def_pb(graph_def, filename):
    with tf.io.gfile.GFile(filename, "wb") as f:
        f.write(graph_def.SerializeToString())


def dump_pb_without_vars(graph, output_names, filename):
    with tf.io.gfile.GFile(filename, "wb") as f:
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), output_names
            )
        f.write(graph_def.SerializeToString())


def save_model(graph, model_name):
    with graph.as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            save_path = tf.train.Saver().save(sess, model_name)
            print("Model saved in path: %s" % save_path)
