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
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx import TensorProto

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
        ((4, 4, 4, 4)),  # Normal[[2, 2], []]
        pytest.param(
            (2, 2),
            marks=pytest.mark.skip(reason="non 4/5D input not handled"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
@pytest.mark.parametrize(
    "Op",
    [
        pytest.param("Neg", marks=pytest.mark.skip(reason="Neg not implemented")),
        pytest.param("Floor", marks=pytest.mark.skip(reason="Floor not implemented")),
        pytest.param(
            "Identity", marks=pytest.mark.skip(reason="Identity not implemented")
        ),
    ],
)
def test_uop(test_dir, backend, Op, a_shape, dtype):
    a = dtype(np.random.randn(*a_shape))
    if Op == "Neg":
        out = np.negative(a)
    elif Op == "Floor":
        out = np.floor(a)
    elif Op == "Identity":
        out = np.negative(a)
    node = helper.make_node(
        Op,
        inputs=["a"],
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


@pytest.mark.parametrize(
    "a_shape, axes, keepdims",
    [
        pytest.param(
            (3, 2, 2),
            None,
            1,
            marks=pytest.mark.skip(reason="axes can't be none. keepdims has to be 0"),
            id="default_axes_keepdims",
        ),
        pytest.param(
            (3, 2, 2),
            [1],
            0,
            marks=pytest.mark.skip(reason="axes length has to be 2"),
            id="do_not_keepdims",
        ),
        pytest.param(
            (3, 2, 2),
            [1],
            1,
            marks=pytest.mark.skip(reason="keepdims has to be 0"),
            id="keepdims",
        ),
        pytest.param(
            (3, 2, 2, 4),
            [1, 2],
            0,
            marks=pytest.mark.skip(reason="segfault"),
            id="reduce_nc",
        ),
        pytest.param((3, 2, 2, 4), [2, 3], 0, id="reduce_hw"),
        pytest.param(
            (3, 2, 2),
            [-2],
            1,
            marks=pytest.mark.skip(reason="don't support negative axes"),
            id="negative_axes_keepdims",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_reducemean(test_dir, backend, a_shape, axes, keepdims, dtype):
    Op = "ReduceMean"
    a = dtype(np.random.randn(*a_shape))
    out = np.mean(
        a, axis=(None if axes is None else tuple(axes)), keepdims=keepdims == 1
    )

    kwargs = {"name": Op, "inputs": ["a"], "outputs": ["out"], "keepdims": keepdims}

    if axes is not None:
        kwargs["axes"] = axes

    node = helper.make_node(Op, **kwargs)
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


@pytest.mark.parametrize(
    "a_shape, start, end",
    [
        pytest.param(
            (4, 4, 4, 4),
            None,
            None,
            marks=pytest.mark.skip(reason="bug in addOutputs"),
        ),
        pytest.param(
            (2, 2),
            None,
            None,
            marks=pytest.mark.skip(reason="bug in addOutputs"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_shape(test_dir, backend, a_shape, start, end, dtype):
    Op = "Shape"
    a = dtype(np.random.randn(*a_shape))
    out = np.array(a.shape[start:end]).astype(np.int64)
    kwargs = {}
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end
    node = onnx.helper.make_node(Op, inputs=["a"], outputs=["out"], **kwargs)
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


@pytest.mark.parametrize(
    "a_shape, kernel_shape, pads, strides, auto_pad, output_shape",
    [
        pytest.param(
            [1, 3, 32],
            [2],
            [0, 0],
            [1],
            "NOTSET",
            [1, 3, 31],
            id="averagepool_1d_default",
            marks=pytest.mark.skip(
                reason="bug helper_processPool: list index out of range"
            ),
        ),
        pytest.param(
            [1, 3, 32, 32],
            [2, 2],
            [0, 0, 0, 0],
            [1, 1],
            "NOTSET",
            [1, 3, 31, 31],
            id="averagepool_2d_default",
        ),
        pytest.param(
            [1, 3, 28, 28],
            [3, 3],
            [2, 2, 2, 2],
            [1, 1],
            "NOTSET",
            [1, 3, 30, 30],
            id="averagepool_2d_pads1",
            marks=pytest.mark.skip(reason="bug correctness issue. 23% mismatch"),
        ),
        pytest.param(
            [1, 1, 5, 5],
            [5, 5],
            [2, 2, 2, 2],
            [1, 1],
            "NOTSET",
            [1, 1, 5, 5],
            id="averagepool_2d_pads2",
            marks=pytest.mark.skip(reason="bug correctness issue. 80-90% mismatch"),
        ),
        pytest.param(
            [1, 1, 5, 5],
            [3, 3],
            None,
            [2, 2],
            "SAME_UPPER",
            [1, 1, 3, 3],
            id="averagepool_2d_same_upper",
            marks=pytest.mark.skip(reason="non explicit padding not supported"),
        ),
        pytest.param(
            [1, 3, 32, 32],
            [2, 2],
            None,
            [1, 1],
            "SAME_LOWER",
            [1, 3, 32, 32],
            id="averagepool_2d_same_lower",
            marks=pytest.mark.skip(reason="non explicit padding not supported"),
        ),
        pytest.param(
            [1, 3, 32, 32],
            [5, 5],
            [0, 0, 0, 0],
            [3, 3],
            "NOTSET",
            [1, 3, 10, 10],
            id="averagepool_2d_strides",
        ),
        pytest.param(
            [1, 3, 32, 32, 32],
            [2, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1],
            "NOTSET",
            [1, 3, 31, 31, 31],
            id="averagepool_3d_default",
            marks=pytest.mark.skip(reason="averagepool_3d not supported"),
        ),
    ],
)
# we dont support ceil_mode, count_include_pad
@pytest.mark.parametrize("dtype", [np.single])
def test_avgpool(
    test_dir,
    backend,
    a_shape,
    kernel_shape,
    pads,
    strides,
    auto_pad,
    output_shape,
    dtype,
):
    Op = "AveragePool"
    a = np.random.randn(*a_shape).astype(dtype)
    # Only need this for its shape
    out = np.zeros(output_shape).astype(dtype)

    kwargs = {
        "inputs": ["a"],
        "outputs": ["output"],
        "kernel_shape": kernel_shape,
        "strides": strides,
    }
    if auto_pad is "NOTSET":
        kwargs["pads"] = pads
    else:
        kwargs["auto_pad"] = auto_pad

    node = onnx.helper.make_node(Op, **kwargs)

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


@pytest.mark.parametrize(
    "a_shape, kernel_shape, pads, strides, auto_pad, output_shape",
    [
        pytest.param(
            [1, 3, 32],
            [2],
            [0, 0],
            [1],
            "NOTSET",
            [1, 3, 31],
            id="maxpool_1d_default",
            marks=pytest.mark.skip(
                reason="bug helper_processPool: list index out of range"
            ),
        ),
        pytest.param(
            [1, 3, 32, 32],
            [2, 2],
            [0, 0, 0, 0],
            [1, 1],
            "NOTSET",
            [1, 3, 31, 31],
            id="maxpool_2d_default",
        ),
        pytest.param(
            [1, 3, 28, 28],
            [3, 3],
            [2, 2, 2, 2],
            [1, 1],
            "NOTSET",
            [1, 3, 30, 30],
            id="maxpool_2d_pads1",
            marks=pytest.mark.skip(reason="bug correctness issue. 1.8% mismatch"),
        ),
        pytest.param(
            [1, 1, 5, 5],
            [5, 5],
            [2, 2, 2, 2],
            [1, 1],
            "NOTSET",
            [1, 1, 5, 5],
            id="maxpool_2d_pads2",
        ),
        pytest.param(
            [1, 1, 5, 5],
            [3, 3],
            None,
            [2, 2],
            "SAME_UPPER",
            [1, 1, 3, 3],
            id="maxpool_2d_same_upper",
            marks=pytest.mark.skip(reason="non explicit padding not supported"),
        ),
        pytest.param(
            [1, 3, 32, 32],
            [2, 2],
            None,
            [1, 1],
            "SAME_LOWER",
            [1, 3, 32, 32],
            id="maxpool_2d_same_lower",
            marks=pytest.mark.skip(reason="non explicit padding not supported"),
        ),
        pytest.param(
            [1, 3, 32, 32],
            [5, 5],
            [0, 0, 0, 0],
            [3, 3],
            "NOTSET",
            [1, 3, 10, 10],
            id="maxpool_2d_strides",
        ),
        pytest.param(
            [1, 3, 32, 32, 32],
            [2, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1],
            "NOTSET",
            [1, 3, 31, 31, 31],
            id="maxpool_3d_default",
            marks=pytest.mark.skip(reason="maxpool_3d not supported"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_maxpool(
    test_dir,
    backend,
    a_shape,
    kernel_shape,
    pads,
    strides,
    auto_pad,
    output_shape,
    dtype,
):
    Op = "MaxPool"
    a = np.random.randn(*a_shape).astype(dtype)
    # Only need this for its shape
    out = np.zeros(output_shape).astype(dtype)

    kwargs = {
        "inputs": ["a"],
        "outputs": ["output"],
        "kernel_shape": kernel_shape,
        "strides": strides,
    }
    if auto_pad is "NOTSET":
        kwargs["pads"] = pads
    else:
        kwargs["auto_pad"] = auto_pad

    node = onnx.helper.make_node(Op, **kwargs)

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


@pytest.mark.parametrize(
    "a_shape",
    [
        ((1, 3, 5, 5)),
    ],
)
@pytest.mark.parametrize("dtype", [np.single])
def test_global_avgpool(test_dir, backend, a_shape, dtype):
    a = dtype(np.random.randn(*a_shape))
    out = np.mean(a, axis=tuple(range(2, np.ndim(a))), keepdims=True)
    Op = "GlobalAveragePool"
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


@pytest.mark.parametrize(
    "from_type, to_type",
    [
        pytest.param("FLOAT", "FLOAT", id="cast_identity"),
        pytest.param("FLOAT", "FLOAT16", id="cast_f32_f16"),
        pytest.param("FLOAT", "DOUBLE", id="cast_f32_d"),
        pytest.param("FLOAT16", "FLOAT", id="cast_f16_f32"),
        pytest.param("FLOAT16", "DOUBLE", id="cast_f16_d"),
        pytest.param("DOUBLE", "FLOAT", id="cast_d_f32"),
        pytest.param("DOUBLE", "FLOAT16", id="cast_d_f16"),
        pytest.param("FLOAT", "STRING", id="cast_f32_string"),
        pytest.param("STRING", "FLOAT", id="cast_string_f32"),
    ],
)
@pytest.mark.parametrize(
    "compile_time",
    [
        pytest.param(True),
        pytest.param(
            False,
            marks=pytest.mark.skip(
                reason="""we don't support runtime casting. 
                          Only casts of constants at compile time 
                          are supported and no-ops casts (Identity)"""
            ),
        ),
    ],
)
@pytest.mark.skip(reason="[cast] Bug in add_outputs() - KeyError: 'output'")
def test_cast(test_dir, backend, from_type, to_type, compile_time):
    Op = "Cast"
    shape = (3, 4)
    if "STRING" != from_type:
        input = np.random.random_sample(shape).astype(
            TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)]
        )
        if "STRING" == to_type:
            # Converting input to str, then give it object dtype for generating script
            ss = []
            for i in input.flatten():
                s = str(i).encode("utf-8")
                su = s.decode("utf-8")
                ss.append(su)

            output = np.array(ss).astype(object).reshape([3, 4])
        else:
            output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
    else:
        input = np.array(
            [
                "0.47892547",
                "0.48033667",
                "0.49968487",
                "0.81910545",
                "0.47031248",
                "0.816468",
                "0.21087195",
                "0.7229038",
                "NaN",
                "INF",
                "+INF",
                "-INF",
            ],
            dtype=np.dtype(object),
        ).reshape([3, 4])
        output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
    node = onnx.helper.make_node(
        Op,
        inputs=["input"],
        outputs=["output"],
        to=getattr(TensorProto, to_type),
    )
    if compile_time == True:
        graph = make_onnx_graph(
            node,
            inputs=[],
            outputs=[output],
            tensors=[input],
            tensor_names=["input"],
            name=Op + "_test",
        )
        expected_output = run_onnx(graph, [])
    else:
        graph = make_onnx_graph(
            node,
            inputs=[input],
            outputs=[output],
            tensors=[],
            tensor_names=[],
            name=Op + "_test",
        )
        expected_output = run_onnx(graph, [input])
    config = ONNXConfig(backend).parse_io(graph)
    compiler = Compiler(graph, config, test_dir, Frontend.ONNX)
    if compile_time == True:
        mpc_output = compiler.compile_and_run([])
    else:
        mpc_output = compiler.compile_and_run([input])

    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return


@pytest.mark.parametrize(
    "shape, attribute",
    [
        pytest.param((3, 4), "value", id="constant_tensor"),
        pytest.param((1), "value_float", id="constant_float_scalar"),
        pytest.param((20), "value_floats", id="constant_floats"),
        pytest.param((1), "value_int", id="constant_int_scalar"),
        pytest.param((20), "value_ints", id="constant_ints"),
        pytest.param(
            (3, 4),
            "sparse_value",
            marks=pytest.mark.skip(reason="We don't support sparse tensors"),
        ),
        pytest.param(
            (1),
            "value_string",
            marks=pytest.mark.skip(reason="We don't support string tensors"),
        ),
        pytest.param(
            (20),
            "value_strings",
            marks=pytest.mark.skip(reason="We don't support string tensors"),
        ),
    ],
)
@pytest.mark.skip(
    reason="""[constant] onnxsim gives runtime error. 
Issue is it doesn't support opset version 13 of this node.
Need to fix onnxoptimize upstream"""
)
def test_constant(test_dir, backend, shape, attribute):
    Op = "Constant"
    kwargs = {}
    print("Shape = ", shape)
    if attribute == "value":
        values = np.random.randn(*shape).astype(np.float32)
        kwargs[attribute] = onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.FLOAT,
            dims=values.shape,
            vals=values.flatten().astype(float),
        )
    elif attribute == "value_float":
        values = np.random.randn(1).astype(np.float32)
        kwargs[attribute] = values[0]
    elif attribute == "value_floats":
        values = np.random.randn(*shape).astype(np.float32)
        kwargs[attribute] = values.flatten().astype(float)
    elif attribute == "value_int":
        values = np.array(np.random.randint(-(2 ** 32 - 1), 2 ** 32 - 1)).astype(
            np.int64
        )
        kwargs[attribute] = int(values)
    elif attribute == "value_ints":
        values = np.random.randint(-(2 ** 32 - 1), 2 ** 32 - 1, shape).astype(np.int32)
        print(values)
        kwargs[attribute] = values.flatten().astype(int)

    kwargs["inputs"] = []
    kwargs["outputs"] = ["values"]

    node = helper.make_node(Op, **kwargs)
    graph = make_onnx_graph(
        node,
        inputs=[],
        outputs=[values],
        tensors=[],
        tensor_names=[],
        name=Op + "_test",
    )
    expected_output = run_onnx(graph, [])
    config = ONNXConfig(backend).parse_io(graph)
    compiler = Compiler(graph, config, test_dir, Frontend.ONNX)
    mpc_output = compiler.compile_and_run([])
    assert_almost_equal(
        model_output=expected_output, mpc_tensor=mpc_output, precision=2
    )
    return
