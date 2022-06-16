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
    "a_shape,b_shape,dtype",
    [
        ((4, 4), (4, 4), np.single),  # Normal
        ((2, 2), (1,), np.single),  # Broadcasting
        ((3, 1, 2, 1), (2, 1, 4), np.single),  # Broadcasting
        ((2, 2), (), np.single),  # Constant
    ],
)
@pytest.mark.parametrize(
    "tfOp", [tf.math.add, tf.math.subtract, tf.math.multiply, tf.raw_ops.AddV2]
)
def test_arith_binop(test_dir, backend, tfOp, a_shape, b_shape, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    b_inp = dtype(np.random.randn(*b_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        b = tf.constant(b_inp, name="b")
        output = tfOp(x=a, y=b, name="output")
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
    "a_shape, b_shape, data_format, dtype",
    [
        ([4, 1, 4], [4], None, np.single),  # Normal
        ([4, 1, 4], [4], "N..C", np.single),  # Same as above
        pytest.param(
            [4, 4, 1],
            [4],
            "NC..",
            np.single,
            marks=pytest.mark.skip(reason="[bias_add] NC.. not supported"),
        ),  # Normal
    ],
)
def test_bias_add(test_dir, backend, a_shape, b_shape, data_format, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    b_inp = dtype(np.random.randn(*b_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        b = tf.constant(b_inp, name="b")
        output = tf.nn.bias_add(value=a, bias=b, data_format=data_format, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize(
    "tfOp, a_val, divisor",
    [
        pytest.param(
            tf.divide,
            [7, -7],
            5,
            marks=pytest.mark.skip(reason="[divide] Support for parsing DOUBLES"),
        ),  # [1, -2]
        (tf.divide, [7.0, -7.0], 5.0),  # [1.4, -1.4]
        pytest.param(
            tf.truediv,
            [7, -7],
            5,
            marks=pytest.mark.skip(reason="[divide] Support for parsing DOUBLES"),
        ),  # [1.4, -1.4]
        (tf.truediv, [7.0], 5.0),  # [1.4]
        (tf.divide, 7.0, 5.0),  # 1.4
        pytest.param(
            tf.floordiv,
            [7, -7],
            5,
            marks=pytest.mark.skip(
                reason="[divide] Add support for converting div by constant into a mul"
            ),
        ),  # [1, -2]
        pytest.param(
            tf.floordiv,
            [7.0, -7.0],
            5.0,
            marks=pytest.mark.skip(
                reason="[divide] Add support for converting div by constant into a mul"
            ),
        ),  # [1.0, -2.0]
        pytest.param(
            tf.truncatediv,
            -7,
            5,
            marks=pytest.mark.skip(reason="[divide] Truncated div not supported"),
        ),  # -1
    ],
)
def test_div(test_dir, backend, tfOp, a_val, divisor, dtype):
    graph = tf.Graph()
    a_inp = np.array(a_val)
    with graph.as_default():
        b = tf.constant(divisor, name="b")
        a = tf.compat.v1.placeholder(tf.as_dtype(b.dtype), shape=a_inp.shape, name="a")
        output = tfOp(a, b, name="output")
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
    "a_shape, b_shape, transpose_a, transpose_b, bisModel",
    [
        ([3, 2], [2, 3], False, False, True),
        pytest.param(
            [3, 2],
            [2, 3],
            False,
            False,
            False,
            marks=pytest.mark.skip(
                reason="[matmul] expect atleast one param to belong to model"
            ),
        ),
        ([1, 2], [2, 3], False, False, True),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_matmul(
    test_dir, backend, a_shape, b_shape, transpose_a, transpose_b, bisModel, dtype
):
    if backend == "2PC_HE" and a_shape[0] != 1:
        pytest.skip("HE only supports vector matrix multiplication")
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    b_inp = dtype(np.random.randn(*b_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        if bisModel:
            b = tf.constant(b_inp, name="b")
        else:
            b = tf.compat.v1.placeholder(
                tf.as_dtype(dtype), shape=b_inp.shape, name="b"
            )
        output = tf.matmul(a, b, transpose_a, transpose_b, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        feed_dict = {a: a_inp}
        if not bisModel:
            feed_dict[b] = b_inp
        expected_output = sess.run(output, feed_dict=feed_dict)
    config = TFConfig(backend).add_input(a).add_output(output)
    if not bisModel:
        config.add_input(b)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "a, b",
    [
        ([1.2, 1.3], [1.2, 1.3]),
        ([1.2, 1.3], [1.2, 1.2]),
        ([1.2, 1.3], [1.2]),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.skip(reason="[equal] Not able to cast boolean to int ezpc")
def test_equal(test_dir, backend, a, b, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.array(a))
    b_inp = dtype(np.array(b))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        b = tf.constant(b_inp, name="b")
        output = tf.math.equal(a, b, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return
