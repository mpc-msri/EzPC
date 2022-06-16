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


@pytest.mark.parametrize("a_shape", [[4, 4], [1], []])
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize(
    "tfOp",
    [
        tf.math.sqrt,
        tf.math.rsqrt,
        tf.math.sigmoid,
        tf.math.tanh,
        tf.nn.relu,
    ],
)
def test_non_linear(test_dir, backend, tfOp, a_shape, dtype):
    if backend not in ["2PC_OT", "CPP"] and tfOp in [
        tf.math.sqrt,
        tf.math.rsqrt,
        tf.math.sigmoid,
        tf.math.tanh,
    ]:
        pytest.skip(
            "Operation {op} not supported for backend {backend}".format(
                op=tfOp.__name__, backend=backend
            )
        )
    if a_shape == []:
        pytest.skip(
            "[Athos] Missing Support for tan/sig/sqrt/relu of scalar (0-d) variables"
        )

    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    if tfOp in [tf.math.sqrt, tf.math.rsqrt]:
        a_inp = np.abs(a_inp)

    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tfOp(a, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})
    assert expected_output is not None
    config = TFConfig(backend).add_input(a).add_output(output)
    config.config["scale"] = 12
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.skip(reason="[softmax] Haven't implemented softmax")
@pytest.mark.parametrize("a_shape, axis", [([2, 3], 1), ([1], 0)])
@pytest.mark.parametrize("dtype", [np.single])
def test_softmax(test_dir, backend, a_shape, axis, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.nn.softmax(a, axis=axis, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})
    assert expected_output is not None
    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return
