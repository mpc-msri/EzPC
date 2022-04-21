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


@pytest.mark.parametrize("a_shape", [[2, 2], []])
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize(
    "tfOp",
    [
        tf.math.square,
        tf.math.negative,
        pytest.param(
            tf.math.floor,
            marks=pytest.mark.skip(reason="[floor] Floor1 not implemented"),
        ),
        tf.shape,
        tf.identity,
        tf.zeros_like,
    ],
)
def test_uop(test_dir, backend, tfOp, a_shape, dtype):
    if backend.startswith("2PC") and tfOp == tf.math.square:
        pytest.skip("[SCI][square] Secret Secret mul not implemented")
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tfOp(a, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "a_shape, axis, keepdims",
    [
        ([3, 2], None, False),
        ([3, 2], [0, 1], False),
        ([3, 2], 0, False),
        ([3, 2], 1, False),
        ([3, 2, 4], 1, False),
        ([3, 2, 4], [1, 2], False),
        ([3, 2, 4], [2, 1], False),
        ([3, 2], 0, True),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize("tfOp", [tf.math.reduce_mean, tf.reduce_sum])
# @pytest.mark.skip(reason="[reduce] Reduce mean output mismatch and shape failure")
def test_reduce(test_dir, backend, tfOp, a_shape, axis, keepdims, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tfOp(a, axis=axis, keepdims=keepdims, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "a_shape, axis",
    [
        ([3, 2], None),
        ([3, 2], 0),
        ([3, 2], 1),
        ([3, 2, 3], 1),
        ([3, 2, 1, 1], 1),
        ([3, 2], 1),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.skip(reason="[argmax] Need support for argmax along arbitrary axis")
def test_argmax(test_dir, backend, a_shape, axis, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.math.argmax(a, axis=axis, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


# NHWC is the default format
@pytest.mark.parametrize(
    "a_shape, ksize, strides, padding, data_format",
    [
        ([1, 5, 5, 1], [1, 2, 2, 1], [1, 2, 2, 1], "VALID", "NHWC"),
        pytest.param(
            [1, 5, 5, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            "SAME",
            "NHWC",
            marks=pytest.mark.skip(reason="[max/avg_pool] Pooling SAME pad bug"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize("tfOp", [tf.nn.max_pool, tf.nn.avg_pool])
def test_pool(
    test_dir, backend, tfOp, a_shape, ksize, strides, padding, data_format, dtype
):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tfOp(
            a,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name="output",
        )
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


# x = tf.constant([1.8, 2.2], dtype=tf.float32)
# tf.dtypes.cast(x, tf.int32)
# Currently cast acts as an identity operation.
@pytest.mark.parametrize("a_shape", [[2, 2]])
@pytest.mark.parametrize(
    "from_dtype, to_dtype",
    [
        (np.single, np.single),
        pytest.param(
            np.double,
            np.single,
            marks=pytest.mark.skip(reason="[cast] Support for parsing DOUBLES"),
        ),
        pytest.param(
            np.single,
            np.int32,
            marks=pytest.mark.skip(reason="[cast] Only support identity cast"),
        ),
    ],
)
def test_cast(test_dir, backend, a_shape, from_dtype, to_dtype):
    graph = tf.Graph()
    a_inp = from_dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(
            tf.as_dtype(from_dtype), shape=a_inp.shape, name="a"
        )
        output = tf.cast(a, to_dtype, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize("a_shape, value", [([2, 2], 9.2), ([], 9.2), ([2, 2], 1)])
def test_fill(test_dir, backend, a_shape, value):
    graph = tf.Graph()
    with graph.as_default():
        output = tf.fill(a_shape, value)
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output)

    config = TFConfig(backend).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([], timeoutSeconds=60)
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return
