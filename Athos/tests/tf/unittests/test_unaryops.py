import tensorflow as tf
import numpy as np

import pytest

import sys
import os

# Athos DIR
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tests.utils import Config, Compiler, assert_almost_equal


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
        pytest.param(
            tf.zeros_like, marks=pytest.mark.skip(reason="[zeros_like] EzPC issue for inp=[2,2]")
        ),
    ],
)
def test_uop(test_dir, backend, tfOp, a_shape, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tfOp(a, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = Config(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(tf_output=expected_output, mpc_tensor=mpc_output, precision=2)
    return


@pytest.mark.parametrize(
    "a_shape, axis, keepdims",
    [
        ([3, 2], None, False),
        ([3, 2], [0, 1], False),
        ([3, 2], 0, False),
        ([3, 2], 1, False),
        ([3, 2], 0, True),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize("tfOp", [tf.math.reduce_mean, tf.reduce_sum])
@pytest.mark.skip(reason="[reduce] Reduce mean assert shape failure")
def test_reduce(test_dir, backend, tfOp, a_shape, axis, keepdims, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tfOp(a, axis=axis, keepdims=keepdims, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = Config(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(tf_output=expected_output, mpc_tensor=mpc_output, precision=2)
    return


@pytest.mark.parametrize(
    "a_shape, axis",
    [
        ([3, 2], None),
        ([3, 2], 0),
        ([3, 2], 1),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.skip(reason="[argmax] Generic argmax not implemented")
def test_argmax(test_dir, backend, a_shape, axis, dtype):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tf.math.argmax(a, axis=axis, name="output")
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})

    config = Config(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(tf_output=expected_output, mpc_tensor=mpc_output, precision=2)
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

    config = Config(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(tf_output=expected_output, mpc_tensor=mpc_output, precision=2)
    return


# x = tf.constant([1.8, 2.2], dtype=tf.float32)
# tf.dtypes.cast(x, tf.int32)
# Currently cast acts as an identity operation.
@pytest.mark.parametrize("a_shape", [[2, 2]])
@pytest.mark.parametrize(
    "from_dtype, to_dtype",
    [
        (np.single, np.single),
        (
            np.double,
            np.single,
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

    config = Config(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(tf_output=expected_output, mpc_tensor=mpc_output, precision=2)
    return


@pytest.mark.parametrize("a_shape, value", [([2, 2], 9.2), ([], 9.2), ([2, 2], 1)])
def test_fill(test_dir, backend, a_shape, value):
    graph = tf.Graph()
    with graph.as_default():
        output = tf.fill(a_shape, value)
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output)

    config = Config(backend).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([])
    assert_almost_equal(tf_output=expected_output, mpc_tensor=mpc_output, precision=2)
    return