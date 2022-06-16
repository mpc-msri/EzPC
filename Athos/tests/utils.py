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
import tempfile
import sys
import os
import shutil
import re
from enum import Enum, auto

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import CompileTFGraph
import CompileONNXGraph
import CompilerScripts.parse_config as parse_config
from CompilerScripts.get_output import convert_raw_output_to_np

import onnx
from onnx import helper
from onnx.backend.test.case import node
from onnx import numpy_helper
import onnxruntime as ort

from io import BytesIO

import numpy as np
import subprocess
import threading


class Frontend(Enum):
    Tensorflow = auto()
    ONNX = auto()


class BaseConfig:
    def __init__(self, mode):
        self.config = {
            "scale": 23,
            "bitlength": 64,
            "save_weights": True,
        }
        if mode == "CPP":
            self.config["target"] = "CPP"
        elif mode == "3PC":
            self.config["target"] = "PORTHOS"
        elif mode == "2PC_OT":
            self.config["target"] = "SCI"
            self.config["bitlength"] = 41
            self.config["scale"] = 15
            self.config["backend"] = "OT"

        elif mode == "2PC_HE":
            self.config["target"] = "SCI"
            self.config["bitlength"] = 41
            self.config["scale"] = 12
            self.config["backend"] = "HE"
        else:
            assert False, "Mode has to be one of CPP/3PC/2PC_OT/2PC_HE"


class TFConfig(BaseConfig):
    def __init__(self, mode):
        super().__init__(mode)
        self.config["model_name"] = "model.pb"

    def add_input(self, tensor_op):
        input_name = tensor_op.op.name
        shape = tensor_op.shape.as_list()
        shape_string = ",".join(map(str, shape))
        inputs = self.config.get("input_tensors")
        if inputs == None:
            self.config["input_tensors"] = {input_name: shape_string}
        else:
            self.config["input_tensors"][input_name] = shape_string
        return self

    def add_output(self, tensor_op):
        output_name = tensor_op.name
        outputs = self.config.get("output_tensors")
        if outputs == None:
            self.config["output_tensors"] = [output_name]
        else:
            self.config["output_tensors"].append(output_name)
        return self


class ONNXConfig(BaseConfig):
    def __init__(self, mode):
        super().__init__(mode)
        self.config["model_name"] = "model.onnx"

    def parse_io(self, graph):
        for inp in graph.input:
            input_name = inp.name
            shape = [i.dim_value for i in inp.type.tensor_type.shape.dim]
            shape_string = ",".join(map(str, shape))
            inputs = self.config.get("input_tensors")
            if inputs == None:
                self.config["input_tensors"] = {input_name: shape_string}
            else:
                self.config["input_tensors"][input_name] = shape_string

        for out in graph.output:
            output_name = out.name
            outputs = self.config.get("output_tensors")
            if outputs == None:
                self.config["output_tensors"] = [output_name]
            else:
                self.config["output_tensors"].append(output_name)
        return self


def get_params(config):
    return parse_config.parse_config(config)


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.mkdir(path)
    return


def save_graph(graph_def, config, test_dir, frontend):
    fname = config["model_name"]
    fpath = os.path.join(test_dir, fname)
    if frontend == Frontend.Tensorflow:
        with open(fpath, "wb") as f:
            f.write(graph_def.SerializeToString())
    elif frontend == Frontend.ONNX:
        model = onnx.helper.make_model(graph_def, producer_name="onnx-test")
        model.opset_import[0].version = 15
        onnx.save(model, fpath)
    config["model_name"] = fpath
    return


class Program:
    def __init__(self, program_path, model_weight_path, params, test_dir):
        self.program_path = program_path
        self.model_weight_path = model_weight_path
        self.scale = params["scale"]
        self.bitlength = params["bitlength"]
        self.target = params["target"]
        self.test_dir = test_dir

    def run(self, inputs, timeoutSeconds):
        # scale input and dump to file
        inputs_scaled = os.path.join(
            self.test_dir, "input_fixedpt_scale_" + str(self.scale) + ".inp"
        )
        with open(inputs_scaled, "w") as ff:
            for i in inputs:
                for xx in np.nditer(i, order="C"):
                    ff.write(str(int(xx * (1 << self.scale))) + " ")
                ff.write("\n")
        raw_output = os.path.join(self.test_dir, "raw_output")
        if self.target == "CPP":
            os.system(
                "cat {inputs} {weights} | {program} > {output}".format(
                    program=self.program_path,
                    inputs=inputs_scaled,
                    weights=self.model_weight_path,
                    output=raw_output,
                )
            )
        elif self.target == "PORTHOS":
            util_dir = os.path.dirname(os.path.abspath(__file__))
            porthos_dir = os.path.join(util_dir, "..", "..", "Porthos")
            ip_addr = os.path.join(porthos_dir, "files", "addresses")
            keys_dir = os.path.join(porthos_dir, "files", "keys")
            client_cmd = (
                "{program} 0 {ip_addr_file} {keys_dir} < {input} > {output}".format(
                    program=self.program_path,
                    ip_addr_file=ip_addr,
                    input=inputs_scaled,
                    output=raw_output,
                    keys_dir=keys_dir,
                )
            )
            server_cmd = "{program} 1 {ip_addr_file} {keys_dir} < {input}".format(
                program=self.program_path,
                ip_addr_file=ip_addr,
                input=self.model_weight_path,
                keys_dir=keys_dir,
            )
            party2_cmd = "{program} 2 {ip_addr_file} {keys_dir}".format(
                program=self.program_path, ip_addr_file=ip_addr, keys_dir=keys_dir
            )
            commands = [client_cmd, server_cmd, party2_cmd]
            procs = [subprocess.Popen(i, shell=True) for i in commands]
            for p in procs:
                try:
                    p.wait(timeoutSeconds)
                except subprocess.TimeoutExpired:
                    p.kill()
        elif self.target == "SCI":
            util_dir = os.path.dirname(os.path.abspath(__file__))
            sci_dir = os.path.join(util_dir, "..", "..", "SCI")
            port = 1234
            client_cmd = "{program} r=2 port={port} < {input} > {output}".format(
                program=self.program_path,
                port=port,
                input=inputs_scaled,
                output=raw_output,
            )
            server_cmd = "{program} r=1 port={port} < {input} > /dev/null".format(
                program=self.program_path,
                port=port,
                input=self.model_weight_path,
                output=raw_output,
            )
            commands = [client_cmd, server_cmd]
            procs = [subprocess.Popen(i, shell=True) for i in commands]
            for p in procs:
                try:
                    p.wait(timeoutSeconds)
                except subprocess.TimeoutExpired:
                    p.kill()
        return convert_raw_output_to_np(raw_output, self.bitlength, self.scale)


# Compiler(graph, config, test_dir, frontend=Frontend.ONNX)
class Compiler:
    def __init__(self, graph, config, test_dir, frontend=Frontend.Tensorflow):
        if frontend == Frontend.Tensorflow:
            self.graph_def = graph.as_graph_def()
        else:
            self.graph_def = graph
        self.config = config.config
        self.test_dir = test_dir
        self.frontend = frontend

    def compile_and_run(self, inputs, timeoutSeconds=40):
        save_graph(self.graph_def, self.config, self.test_dir, self.frontend)
        params = get_params(self.config)
        if self.frontend == Frontend.Tensorflow:
            (output_program, model_weight_file) = CompileTFGraph.generate_code(
                params, role="server", debug=False
            )
        else:
            (output_program, model_weight_file) = CompileONNXGraph.generate_code(
                params, role="server", debug=False
            )
        prog = Program(output_program, model_weight_file, params, self.test_dir)
        output = prog.run(inputs, timeoutSeconds)
        return output


def assert_almost_equal(model_output, mpc_tensor, precision):
    if model_output.shape == (0,):
        return
    np.testing.assert_almost_equal(
        model_output.flatten(), mpc_tensor, decimal=precision
    )
    return


def make_onnx_graph(
    inp_node,  # type: onnx.NodeProto
    inputs,  # type: Sequence[np.ndarray]
    outputs,  # type: Sequence[np.ndarray]
    tensors,  # type: Sequence[np.ndarray] The other tensors that appear in the graph
    tensor_names,
    name,
):
    present_inputs = [x for x in inp_node.input if (x != "")]
    present_outputs = [x for x in inp_node.output if (x != "")]
    inputs_vi = [
        node._extract_value_info(arr, arr_name)
        for arr, arr_name in zip(inputs, present_inputs)
    ]
    outputs_vi = [
        node._extract_value_info(arr, arr_name)
        for arr, arr_name in zip(outputs, present_outputs)
    ]
    initializer = [
        numpy_helper.from_array(arr, arr_name)
        for arr, arr_name in zip(tensors, tensor_names)
    ]
    value_info = [
        node._extract_value_info(arr, arr_name)
        for arr, arr_name in zip(tensors, tensor_names)
    ]
    graph = onnx.helper.make_graph(
        nodes=[inp_node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi,
        initializer=initializer,
        value_info=value_info,
    )
    return graph


def run_onnx(graph, inputs):
    model = onnx.helper.make_model(graph, producer_name="onnx-test")
    print(model.opset_import)
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    model_file = BytesIO()
    onnx.save_model(model, model_file)
    sess = ort.InferenceSession(model_file.getvalue())
    feed_dict = {}
    for i in range(0, len(inputs)):
        input_name = sess.get_inputs()[i].name
        feed_dict[input_name] = inputs[i]
    output_names = [i.name for i in graph.output]
    output = sess.run(output_names, feed_dict)
    model_file.close()
    if len(output) == 1:
        return output[0]
    return output
