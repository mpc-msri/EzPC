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

# TODO: add conv autopad, convtranspose autopad, convtranspose_dilations, fix grouped conv dims


@pytest.mark.parametrize(
    "a_shape, kernel_shape, pads, strides, output_shape, group",
    [
        pytest.param(
            [1, 1, 5, 5],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [1, 1],
            [1, 1, 5, 5],
            1,
            id="conv2d_pad",
        ),
        pytest.param(
            [1, 1, 5, 5],
            [1, 1, 3, 3],
            [0, 0, 0, 0],
            [1, 1],
            [1, 1, 3, 3],
            1,
            id="conv2d_nopad",
        ),
        pytest.param(
            [1, 1, 7, 5],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [2, 2],
            [1, 1, 4, 3],
            1,
            id="conv2d_strides_pad",
        ),
        pytest.param(
            [1, 1, 7, 5],
            [1, 1, 3, 3],
            [0, 0, 0, 0],
            [2, 2],
            [1, 1, 3, 2],
            1,
            id="conv2d_strides_nopad",
        ),
        pytest.param(
            [1, 1, 7, 5],
            [1, 1, 3, 3],
            [1, 0, 1, 0],
            [2, 2],
            [1, 1, 4, 2],
            1,
            marks=pytest.mark.skip(reason="Seedot reshape typecheck assertion"),
            id="conv2d_strides_assymetric_pad",
        ),  # padding only along H dimension
        # a_shape, kernel_shape, pads, strides, output_shape",
        pytest.param(
            [1, 2, 4, 16, 16],
            [2, 2, 3, 3, 3],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1],
            [1, 2, 4, 16, 16],
            1,
            id="conv3d_pad",
        ),
        pytest.param(
            [1, 2, 4, 16, 16],
            [2, 2, 3, 3, 3],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1],
            [1, 2, 2, 14, 14],
            1,
            id="conv3d_nopad",
        ),
        pytest.param(
            [1, 2, 4, 16, 16],
            [2, 2, 3, 3, 3],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2],
            [1, 2, 2, 8, 8],
            1,
            id="conv3d_strides_pad",
        ),
        pytest.param(
            [1, 2, 4, 16, 16],
            [2, 2, 3, 3, 3],
            [0, 0, 0, 0, 0, 0],
            [2, 2, 2],
            [1, 2, 1, 7, 7],
            1,
            id="conv3d_strides_nopad",
        ),
        pytest.param(
            [1, 4, 5, 5],
            [1, 1, 3, 3],
            [0, 0, 0, 0],
            [1, 1],
            [1, 1, 3, 3],
            4,
            id="conv2d_grouped",
            marks=pytest.mark.skip(reason="fix test dims"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_conv(
    test_dir, backend, a_shape, kernel_shape, pads, strides, output_shape, group, dtype
):
    Op = "Conv"
    if len(a_shape) == 4:
        version = 2  # 2d
    elif len(a_shape) == 5:
        version = 3  # 3d

    if version == 3 and backend in ["2PC_HE", "2PC_OT"]:
        pytest.skip("[conv3d] Missing Support in SCI")

    a = np.random.randn(*a_shape).astype(dtype)
    kernel = np.random.randn(*kernel_shape).astype(dtype)

    # Only need this for its shape
    out = np.zeros(output_shape).astype(dtype)

    hw_kernel_shape = kernel_shape[-version:]
    node = onnx.helper.make_node(
        Op,
        inputs=["a", "kernel"],
        outputs=["output"],
        kernel_shape=hw_kernel_shape,
        pads=pads,
        strides=strides,
        group=group,
        # Default values for other attributes: dilations=[1, 1], groups=1
    )

    graph = make_onnx_graph(
        node,
        inputs=[a],
        outputs=[out],
        tensors=[kernel],
        tensor_names=["kernel"],
        name=Op + "_test",
    )
    expected_output = run_onnx(graph, [a])
    config = ONNXConfig(backend).parse_io(graph)
    config.config["scale"] = 12
    compiler = Compiler(graph, config, test_dir, Frontend.ONNX)
    mpc_output = compiler.compile_and_run([a])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "a_shape, kernel_shape, pads, strides, output_shape, output_padding",
    [
        pytest.param(
            [1, 1, 3, 3],
            [1, 2, 3, 3],
            [0, 0, 0, 0],
            [1, 1],
            [1, 2, 5, 5],
            False,
            id="convtranspose2d_nopad",
        ),
        pytest.param(
            [1, 1, 3, 3],
            [1, 2, 3, 3],
            [1, 1, 1, 1],
            [1, 1],
            [1, 2, 3, 3],
            False,
            id="convtranspose2d_pad",
        ),
        pytest.param(
            [1, 1, 3, 3],
            [1, 2, 3, 3],
            [0, 0, 0, 0],
            [3, 2],
            [1, 2, 10, 8],
            True,
            id="convtranspose2d_output_padding",
        ),
        pytest.param(
            [1, 1, 3, 4, 5],
            [1, 2, 3, 3, 3],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1],
            [1, 2, 5, 6, 7],
            False,
            id="convtranspose3d_nopad",
        ),
        pytest.param(
            [1, 1, 3, 4, 5],
            [1, 2, 3, 3, 3],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1],
            [1, 2, 3, 4, 5],
            False,
            id="convtranspose3d_pad",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_convtranspose(
    test_dir,
    backend,
    a_shape,
    kernel_shape,
    pads,
    strides,
    output_shape,
    output_padding,
    dtype,
):
    Op = "ConvTranspose"
    if len(a_shape) == 4:
        version = 2  # 2d
    elif len(a_shape) == 5:
        version = 3  # 3d

    if version == 3 and backend in ["2PC_HE", "2PC_OT"]:
        pytest.skip("[conv3dtranspose] Missing Support in SCI")

    a = np.random.randn(*a_shape).astype(dtype)
    kernel = np.random.randn(*kernel_shape).astype(dtype)

    # Only need this for its shape
    out = np.zeros(output_shape).astype(dtype)

    hw_kernel_shape = kernel_shape[-version:]
    if not output_padding:
        node = onnx.helper.make_node(
            Op,
            inputs=["a", "kernel"],
            outputs=["output"],
            kernel_shape=hw_kernel_shape,
            pads=pads,
            strides=strides
            # Default values for other attributes: dilations=[1, 1], groups=1
        )
    else:
        node = onnx.helper.make_node(
            Op,
            inputs=["a", "kernel"],
            outputs=["output"],
            kernel_shape=hw_kernel_shape,
            pads=pads,
            strides=strides,
            output_padding=[1, 1]
            # Default values for other attributes: dilations=[1, 1], groups=1
        )

    graph = make_onnx_graph(
        node,
        inputs=[a],
        outputs=[out],
        tensors=[kernel],
        tensor_names=["kernel"],
        name=Op + "_test",
    )
    expected_output = run_onnx(graph, [a])
    config = ONNXConfig(backend).parse_io(graph)
    config.config["scale"] = 12
    compiler = Compiler(graph, config, test_dir, Frontend.ONNX)
    mpc_output = compiler.compile_and_run([a])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return
