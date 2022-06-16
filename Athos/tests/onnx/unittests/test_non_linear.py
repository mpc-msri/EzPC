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


@pytest.mark.parametrize(
    "a_shape",
    [
        ((4, 4, 4, 4)),  # Normal
        pytest.param(
            (4, 4),
            marks=pytest.mark.skip(reason="non 4/5D input not handled"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize(
    "Op",
    [
        pytest.param("Relu", id="relu"),
        pytest.param("Sqrt", marks=pytest.mark.skip(reason="Sqrt not implemented")),
        pytest.param(
            "Sigmoid", marks=pytest.mark.skip(reason="Sigmoid not implemented")
        ),
        pytest.param("Tanh", marks=pytest.mark.skip(reason="Tanh not implemented")),
    ],
)
def test_non_linear(test_dir, backend, Op, a_shape, dtype):
    a = np.random.randn(*a_shape).astype(dtype)
    if Op == "Relu":
        out = np.clip(a, 0, np.inf)
    elif Op == "Sigmoid":
        out = 1.0 / (1.0 + np.exp(np.negative(a)))
    elif Op == "Tanh":
        out = np.tanh(a)
    elif Op == "Sqrt":
        out = np.sqrt(a)

    node = helper.make_node(
        Op,
        inputs=[
            "a",
        ],
        outputs=["out"],
    )
    graph = make_onnx_graph(
        node,
        inputs=[a],
        outputs=[out],
        tensors=[],
        tensor_names=[],
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
