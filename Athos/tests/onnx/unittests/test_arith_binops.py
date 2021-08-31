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
from onnx.backend.test.case.node.gemm import gemm_reference_implementation

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
    "a_shape,b_shape,dtype",
    [
        ((4, 4, 4, 4), (4, 4, 4, 4), np.single),  # Normal
        pytest.param(
            (4, 4),
            (4, 4),
            np.single,
            marks=pytest.mark.skip(reason="non 4/5D input not handled"),
        ),  # Normal
        pytest.param(
            (2, 2),
            (1,),
            np.single,
            marks=pytest.mark.skip(reason="non 4/5D input not handled"),
        ),  # Broadcasting
        pytest.param(
            (3, 1, 2, 1),
            (2, 1, 4),
            np.single,
            marks=pytest.mark.skip(reason="non 4/5D input not handled"),
        ),  # Broadcasting
        pytest.param(
            (2, 2),
            (),
            np.single,
            marks=pytest.mark.skip(reason="non 4/5D input not handled"),
        ),  # Constant
    ],
)
@pytest.mark.parametrize(
    "Op",
    [
        ("Add"),
        pytest.param("Sub", marks=pytest.mark.skip(reason="Sub not implemented")),
        pytest.param("Mul", marks=pytest.mark.skip(reason="Mul not implemented")),
    ],
)
def test_arith_binop(test_dir, backend, Op, a_shape, b_shape, dtype):
    onnx_to_np_op = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply}
    a = np.random.randn(*a_shape).astype(dtype)
    b = np.random.randn(*b_shape).astype(dtype)
    out = onnx_to_np_op[Op](a, b)
    node = helper.make_node(
        Op,
        inputs=["a", "b"],
        outputs=["out"],
    )
    graph = make_onnx_graph(
        node,
        inputs=[a],
        outputs=[out],
        tensors=[b],
        tensor_names=["b"],
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


@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize(
    "a_val, divisor",
    [
        pytest.param(
            [7.0, -7.0], 5.0, marks=pytest.mark.skip(reason="Div not implemented")
        ),  # [1.4, -1.4]
        pytest.param(
            7.0, 5.0, marks=pytest.mark.skip(reason="Div not implemented")
        ),  # 1.4
        pytest.param(
            [3.0, 4.0], [1.0, 2.0], marks=pytest.mark.skip(reason="Div not implemented")
        ),
    ],
)
def test_div(test_dir, backend, a_val, divisor, dtype):
    Op = "Div"
    a = np.array(a_val).astype(dtype)
    b = np.array(divisor).astype(dtype)
    out = np.divide(a, b)
    node = helper.make_node(
        Op,
        inputs=["a", "b"],
        outputs=["out"],
    )
    graph = make_onnx_graph(
        node,
        inputs=[a],
        outputs=[out],
        tensors=[b],
        tensor_names=["b"],
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


@pytest.mark.parametrize(
    "a_shape, b_shape, bisModel",
    [
        ([3, 2], [2, 3], True),
        pytest.param(
            [3, 2],
            [2, 3],
            False,
            marks=pytest.mark.skip(
                reason="[matmul] expect atleast one param to belong to model"
            ),
        ),
        ([1, 2], [2, 3], True),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.skip(reason="matmul not supported for now. only gemm is.")
def test_matmul(test_dir, backend, a_shape, b_shape, bisModel, dtype):
    if backend == "2PC_HE" and a_shape[0] != 1:
        pytest.skip("HE only supports vector matrix multiplication")
    Op = "MatMul"
    a = np.random.randn(*a_shape).astype(dtype)
    b = np.random.randn(*b_shape).astype(dtype)
    out = np.matmul(a, b)
    node = onnx.helper.make_node(
        Op,
        inputs=["a", "b"],
        outputs=["out"],
    )
    if not bisModel:
        graph = make_onnx_graph(
            node,
            inputs=[a, b],
            outputs=[out],
            name=Op + "_test",
        )
        expected_output = run_onnx(graph, [a, b])
    else:
        graph = make_onnx_graph(
            node,
            inputs=[a],
            outputs=[out],
            tensors=[b],
            tensor_names=["b"],
            name=Op + "_test",
        )
        expected_output = run_onnx(graph, [a])
    config = ONNXConfig(backend).parse_io(graph)
    compiler = Compiler(graph, config, test_dir, Frontend.ONNX)
    if not bisModel:
        mpc_output = compiler.compile_and_run([a, b])
    else:
        mpc_output = compiler.compile_and_run([a])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape, alpha, beta, transA, transB",
    [
        pytest.param(
            (3, 2),
            (2, 3),
            (1, 5),
            0.25,
            0.35,
            1,
            1,
            id="gemm_all_attributes",
            marks=pytest.mark.skip(reason="[gemm] alpha,beta not handled"),
        ),
        pytest.param(
            (3, 5),
            (5, 4),
            (1, 4),
            0.5,
            None,
            0,
            0,
            id="gemm_alpha",
            marks=pytest.mark.skip(reason="[gemm] alpha not handled"),
        ),
        pytest.param(
            (2, 7),
            (7, 4),
            (1, 4),
            None,
            0.5,
            0,
            0,
            id="gemm_beta",
            marks=pytest.mark.skip(reason="[gemm] beta not handled"),
        ),
        pytest.param(
            (3, 6), (6, 4), (3, 4), None, None, 0, 0, id="gemm_default_matrix_bias"
        ),
        pytest.param(
            (2, 10),
            (10, 3),
            None,
            None,
            None,
            0,
            0,
            id="gemm_default_no_bias",
            marks=pytest.mark.skip(reason="[gemm] bias is mandatory"),
        ),
        pytest.param(
            (2, 3),
            (3, 4),
            (),
            None,
            None,
            0,
            0,
            id="gemm_default_scalar_bias",
            marks=pytest.mark.skip(reason="[gemm] scaleup0 not found"),
        ),
        pytest.param(
            (3, 7),
            (7, 3),
            (1,),
            None,
            None,
            0,
            0,
            id="gemm_default_single_elem_vector_bias",
        ),
        pytest.param(
            (2, 7), (7, 4), (1, 4), None, None, 0, 0, id="gemm_default_vector_bias"
        ),
        pytest.param((6, 3), (6, 4), (1, 4), None, None, 1, 0, id="gemm_transposeA"),
        pytest.param((3, 6), (4, 6), (1, 4), None, None, 0, 1, id="gemm_transposeB"),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_gemm(
    test_dir, backend, a_shape, b_shape, c_shape, alpha, beta, transA, transB, dtype
):
    Op = "Gemm"
    a = np.random.randn(*a_shape).astype(dtype)
    b = np.random.randn(*b_shape).astype(dtype)

    kwargs = {"inputs": ["a", "b"], "outputs": ["out"]}
    npkwargs = {}

    if c_shape is not None:
        kwargs["inputs"].append("c")
        c = dtype(np.random.randn(*c_shape))
        npkwargs["C"] = c

    if alpha is not None:
        kwargs["alpha"] = alpha
        npkwargs["alpha"] = alpha
    if beta is not None:
        kwargs["beta"] = beta
        npkwargs["beta"] = beta
    if transA == 1:
        kwargs["transA"] = 1
        npkwargs["transA"] = 1
    if transB == 1:
        kwargs["transB"] = 1
        npkwargs["transB"] = 1

    out = gemm_reference_implementation(a, b, **npkwargs)
    node = onnx.helper.make_node(Op, **kwargs)

    kwargs = {
        "inputs": [a],
        "outputs": [out],
        "tensors": [b],
        "tensor_names": ["b"],
        "name": Op + "_test",
    }

    if c_shape is not None:
        kwargs["tensors"].append(c)
        kwargs["tensor_names"].append("c")

    graph = make_onnx_graph(node, **kwargs)
    expected_output = run_onnx(graph, [a])
    config = ONNXConfig(backend).parse_io(graph)
    compiler = Compiler(graph, config, test_dir, Frontend.ONNX)
    mpc_output = compiler.compile_and_run([a])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "a_val, b_val",
    [
        ([1.2, 1.3], [1.2, 1.3]),
        ([1.2, 1.3], [1.2, 1.2]),
        ([1.2, 1.3], [1.2]),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.skip(reason="[equal] Not able to cast boolean to int ezpc")
def test_equal(test_dir, backend, a_val, b_val, dtype):
    Op = "Equal"
    a = np.array(a_val).astype(dtype)
    b = np.array(b_val).astype(dtype)
    out = np.equal(a, b)
    node = helper.make_node(
        Op,
        inputs=["a", "b"],
        outputs=["out"],
    )
    graph = make_onnx_graph(
        node,
        inputs=[a],
        outputs=[out],
        tensors=[b],
        tensor_names=["b"],
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
