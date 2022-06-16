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
import numpy as np
import onnx
from onnx import helper

import pytest

# Athos DIR
import sys, os
import optparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tests.utils import (
    ONNXConfig,
    Compiler,
    assert_almost_equal,
    make_onnx_graph,
    run_onnx,
    Frontend,
)


def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias


@pytest.mark.parametrize(
    "a_shape, scale_val, bias_val, mean_val, var_val",
    [
        ([1, 2, 2, 1], [1.5, 1.5], [2.3, 2.3], [0.5, 0.5], [0.2, 0.2]),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_batch_norm(
    test_dir, backend, a_shape, scale_val, bias_val, mean_val, var_val, dtype
):
    Op = "BatchNormalization"
    a = np.random.randn(*a_shape).astype(dtype)
    scale = np.array(scale_val).astype(dtype)
    bias = np.array(bias_val).astype(dtype)
    mean = np.array(mean_val).astype(dtype)
    var = np.array(var_val).astype(dtype)

    out = _batchnorm_test_mode(a, scale, bias, mean, var).astype(dtype)

    node = onnx.helper.make_node(
        Op,
        inputs=["a", "scale", "bias", "mean", "var"],
        outputs=["out"],
    )

    graph = make_onnx_graph(
        node,
        inputs=[a],
        outputs=[out],
        tensors=[scale, bias, mean, var],
        tensor_names=["scale", "bias", "mean", "var"],
        name=Op + "_test",
    )
    expected_output = run_onnx(graph, [a])
    config = ONNXConfig(backend).parse_io(graph)
    compiler = Compiler(graph, config, test_dir, Frontend.ONNX)
    mpc_output = compiler.compile_and_run([a])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return
