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
import numpy as np

import pytest

import sys
import os

# Athos DIR
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tests.utils import TFConfig, Compiler, assert_almost_equal


@pytest.mark.parametrize(
    "tfOp, a_shape, kernel_shape, strides, padding",
    [
        (tf.nn.conv2d, [1, 5, 5, 1], [2, 2, 1, 2], [1, 1, 1, 1], "SAME"),
        (tf.nn.conv2d, [1, 5, 5, 1], [2, 2, 1, 2], [1, 1, 1, 1], "VALID"),
        (tf.nn.conv3d, [1, 5, 5, 5, 1], [2, 2, 2, 1, 2], [1, 1, 1, 1, 1], "SAME"),
        (tf.nn.conv3d, [1, 5, 5, 5, 1], [2, 2, 2, 1, 2], [1, 1, 1, 1, 1], "VALID"),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_conv(test_dir, backend, tfOp, a_shape, kernel_shape, strides, padding, dtype):
    if tfOp == tf.nn.conv3d and backend in ["2PC_HE", "2PC_OT"]:
        pytest.skip("[conv3d] Missing Support in SCI")
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    kernel_inp = dtype(np.random.randn(*kernel_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        filters = tf.constant(kernel_inp, name="filter")
        output = tfOp(a, filters, strides, padding, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    config.config["scale"] = 12
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "tfOp, a_shape, kernel_shape, strides, padding",
    [
        (tf.nn.depthwise_conv2d, [1, 5, 5, 1], [2, 2, 1, 3], [1, 1, 1, 1], "VALID"),
        (tf.nn.depthwise_conv2d, [1, 5, 5, 1], [2, 2, 1, 3], [1, 1, 1, 1], "SAME"),
        (tf.nn.depthwise_conv2d, [1, 5, 5, 3], [2, 2, 3, 2], [1, 1, 1, 1], "VALID"),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_depthwise_conv(
    test_dir, backend, tfOp, a_shape, kernel_shape, strides, padding, dtype
):
    if backend in ["2PC_HE"]:
        pytest.skip("[SCI][grouped_conv] Missing Support in SCI")
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    kernel_inp = dtype(np.random.randn(*kernel_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        filters = tf.constant(kernel_inp, name="filter")
        output = tfOp(a, filters, strides, padding, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    config.config["scale"] = 12
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "tfOp, a_shape, kernel_shape, output_shape, strides, padding",
    [
        (
            tf.nn.conv3d_transpose,
            [1, 4, 4, 4, 2],
            [2, 2, 2, 1, 2],
            [1, 5, 5, 5, 1],
            [1, 1, 1, 1, 1],
            "VALID",
        ),
        pytest.param(
            tf.nn.conv3d_transpose,
            [1, 5, 5, 5, 2],
            [2, 2, 2, 1, 2],
            [1, 5, 5, 5, 1],
            [1, 1, 1, 1, 1],
            "SAME",
            marks=pytest.mark.skip(reason="[conv3d_transpose] SAME padding bug"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_conv_transpose(
    test_dir,
    backend,
    tfOp,
    a_shape,
    kernel_shape,
    output_shape,
    strides,
    padding,
    dtype,
):
    if backend in ["2PC_HE", "2PC_OT"]:
        pytest.skip("[conv3d] Missing Support in SCI")
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    kernel_inp = dtype(np.random.randn(*kernel_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        filters = tf.constant(kernel_inp, name="filter")
        output = tfOp(a, filters, output_shape, strides, padding, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    config.config["scale"] = 12
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return
