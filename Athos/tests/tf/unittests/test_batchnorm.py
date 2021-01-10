import tensorflow as tf
import numpy as np

import pytest

import sys
import os

# Athos DIR
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tests.utils import Config, Compiler, assert_almost_equal


@pytest.mark.parametrize(
    "a_shape, scale, offset, mean, variance",
    [([1, 2, 2, 1], [1.5], [2.3], [0.5], [0.2]), 
    #([1], 1.5, 2.3, 0.5, 0.2), ([], 1.5, 2.3, 0.5, 0.2)
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize(
    "tfOp", [tf.raw_ops.FusedBatchNorm]
)
@pytest.mark.skip(reason="[batch_norm] Test not complete")
def test_fused_batch_norm(
    test_dir, backend, tfOp, a_shape, scale, offset, mean, variance, dtype
):
    graph = tf.Graph()
    a_inp = dtype(np.random.randn(*a_shape))
    with graph.as_default():
        a = tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=a_inp.shape, name="a")
        output = tfOp(
            x=a,
            scale=scale,
            offset=offset,
            mean=mean,
            variance=variance,
            is_training=False,
            name="output",
        )
    with tf.compat.v1.Session(graph=graph) as sess:
        expected_output = sess.run(output, feed_dict={a: a_inp})
    assert expected_output is not None
    config = Config(backend).add_input(a).add_output(output)
    compiler = Compiler(graph, config, test_dir)
    mpc_output = compiler.compile_and_run([a_inp])
    assert_almost_equal(tf_output=expected_output, mpc_tensor=mpc_output, precision=2)
    return