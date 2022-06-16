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
    "a_shape, out_shape",
    [
        ([2, 3], [6]),
        ([6], [2, 3]),
        ([2, 3], [3, 2]),
        ([2, 3], [-1]),  # Flatten 1-D,
        ([1], []),  # convert to scalar,
        ([3, 2, 3], [2, -1]),  # infer -1 as 9,
        ([3, 2, 3], [-1, 9]),  # infer -1 as 2
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_reshape(test_dir, backend, a_shape, out_shape, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.reshape(a, out_shape, name="output")
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


@pytest.mark.parametrize(
    "a_shape, perm",
    [([2, 3], [1, 0]), ([2, 4, 3], [0, 2, 1])],  # normal transpose, with perm
)
@pytest.mark.parametrize("dtype", [np.single])
def test_transpose(test_dir, backend, a_shape, perm, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.transpose(a, perm, name="output")
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
    "a_shape, num_or_size_splits, axis",
    [
        ([2, 10], 5, 1),
        pytest.param(
            [5, 7],
            [1, 4, 2],
            1,
            marks=pytest.mark.skip(
                reason="[split] don't support split into specific sizes (SplitV)"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_split(test_dir, backend, a_shape, num_or_size_splits, axis, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.split(a, num_or_size_splits, axis, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    if type(output) == list:
        tf_output = output[-1]
        tf_expected_output = expected_output[-1]
    else:
        tf_output = output
        tf_expected_output = expected_output
    config = TFConfig(backend).add_input(a).add_output(tf_output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=tf_expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


# Squeeze
@pytest.mark.parametrize(
    "a_shape, axis",
    [
        pytest.param(
            [1, 2, 1, 3, 1, 1],
            None,
            marks=pytest.mark.skip(reason="[squeeze] Parametric squeeze not supported"),
        ),
        pytest.param(
            [1, 2, 1, 3, 1, 1],
            [2, 4],
            marks=pytest.mark.skip(reason="[squeeze] Parametric squeeze not supported"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_squeeze(test_dir, backend, a_shape, axis, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.squeeze(a, axis=axis, name="output")
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
    "a_shape, begin, size",
    [
        ([3, 2, 3], [1, 0, 0], [1, 1, 3]),
        ([3, 2, 3], [1, 0, 0], [1, 2, 3]),
        ([3, 2, 3], [1, 0, 0], [2, 1, 3]),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_slice(test_dir, backend, a_shape, begin, size, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.slice(a, begin, size, name="output")
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
    "a_shape, b_shape, axis",
    [
        ([2, 3], [3, 3], 0),
        ([2, 3, 2, 1], [2, 6, 2, 1], 1),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_concat(test_dir, backend, a_shape, b_shape, axis, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    b_inp = dtype(np.random.randn(*b_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        b = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=b_inp.shape, name="b")
        output = tf.concat([a, b], axis, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp, b: b_inp})

    config = TFConfig(backend).add_input(a).add_input(b).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp, b_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


# ExpandDims
@pytest.mark.parametrize(
    "a_shape, axis",
    [
        pytest.param(
            [3, 2, 3], 1, marks=pytest.mark.skip(reason="[expand_dims] not supported")
        ),
        pytest.param(
            [2, 5], 0, marks=pytest.mark.skip(reason="[expand_dims] not supported")
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_expand_dims(test_dir, backend, a_shape, axis, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.expand_dims(a, axis, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


# Pad
@pytest.mark.parametrize(
    "a_shape, paddings, mode, constant_values",
    [
        ([1, 2, 2, 1], [[1, 1], [1, 2], [1, 1], [1, 3]], "CONSTANT", 0),
        pytest.param(
            [1, 2, 2, 1],
            [[1, 1], [1, 2], [1, 1], [1, 3]],
            "REFLECT",
            0,
            marks=pytest.mark.skip(reason="[pad] REFLECT not supported"),
        ),
        pytest.param(
            [1, 2, 2, 1],
            [[1, 1], [1, 2], [1, 1], [1, 3]],
            "SYMMETRIC",
            0,
            marks=pytest.mark.skip(reason="[pad] SYMMETRIC not supported"),
        ),
        pytest.param(
            [2, 3],
            [
                [1, 1],
                [2, 2],
            ],
            "CONSTANT",
            0,
            marks=pytest.mark.skip(reason="[pad] Generic pad not supported"),
        ),
        pytest.param(
            [1, 2, 2, 1],
            [[1, 1], [1, 2], [1, 1], [1, 3]],
            "CONSTANT",
            1.2,
            marks=pytest.mark.skip(reason="[pad] non-zero padding not supported"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_pad(test_dir, backend, a_shape, paddings, mode, constant_values, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        pad = tf.constant(paddings, name="paddings")
        output = tf.pad(
            a, pad, mode=mode, constant_values=constant_values, name="output"
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


# Tile
@pytest.mark.parametrize(
    "a_shape, multiples", [([2, 3], [1, 2]), ([2, 3], [2, 1]), ([2, 3], [2, 2])]
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.skip(reason="[tile] Not supported")
def test_tile(test_dir, backend, a_shape, multiples, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        mults = tf.constant(multiples, name="multiples")
        output = tf.tile(a, mults, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = TFConfig(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return
