import tempfile
import sys
import os
import shutil
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import CompilerScripts.parse_config as parse_config
import CompileTFGraph

import numpy as np
import subprocess
import threading


class Config:
    def __init__(self, mode):
        self.config = {
            "model_name": "model.pb",
            "scale": 23,
            "bitlength": 64,
            "save_weights": True,
        }
        if mode == "CPP":
            self.config["target"] = "CPP"
        elif mode == "3PC":
            self.config["target"] = "PORTHOS"
        elif mode == "2PC_OT":
            self.config["target"] = "PORTHOS2PC"
            self.config["bitlength"] = 41
            self.config["backend"] = "OT"

        elif mode == "2PC_HE":
            self.config["target"] = "PORTHOS2PC"
            self.config["bitlength"] = 41
            self.config["backend"] = "HE"
        else:
            assert False, "Mode has to be one of CPP/3PC/2PC_OT/2PC_HE"

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
        output_name = tensor_op.op.name
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


def save_graph(graph_def, config, test_dir):
    fname = config["model_name"]
    fpath = os.path.join(test_dir, fname)
    with open(fpath, "wb") as f:
        f.write(graph_def.SerializeToString())
        print("\n\nfile  name: ", f.name, "\n\n\n")
    config["model_name"] = fpath
    return


def convert_raw_output_to_np(filename, bitlength, scale):
    matcher = re.compile(r"[-]?[0-9]+")
    scaled_array = []
    with open(filename, "r") as f:
        for line in f:
            match = matcher.fullmatch(line.rstrip())
            if match:
                unsigned_number = int(match.group(0))
                number = (
                    unsigned_number
                    if (unsigned_number < 2 ** (bitlength - 1))
                    else unsigned_number - 2 ** bitlength
                )
                scaled_array.append(float(number) / (2 ** scale))
    return np.array(scaled_array)


class Program:
    def __init__(self, program_path, model_weight_path, params, test_dir):
        self.program_path = program_path
        self.model_weight_path = model_weight_path
        self.scale = params["scale"]
        self.bitlength = params["bitlength"]
        self.target = params["target"]
        self.test_dir = test_dir

    def run(self, inputs):
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
                p.wait()
        elif self.target == "PORTHOS2PC":
            util_dir = os.path.dirname(os.path.abspath(__file__))
            sci_dir = os.path.join(util_dir, "..", "..", "SCI")
            port = 1234
            client_cmd = "{program} r=2 p={port} < {input} > {output}".format(
                program=self.program_path,
                port=port,
                input=inputs_scaled,
                output=raw_output,
            )
            server_cmd = "{program} r=1 p={port} < {input} > /dev/null".format(
                program=self.program_path,
                port=port,
                input=self.model_weight_path,
                output=raw_output,
            )
            commands = [client_cmd, server_cmd]
            procs = [subprocess.Popen(i, shell=True) for i in commands]
            for p in procs:
                p.wait()
        return convert_raw_output_to_np(raw_output, self.bitlength, self.scale)


class Compiler:
    def __init__(self, graph, config, test_dir):
        self.graph_def = graph.as_graph_def()
        self.config = config.config
        self.test_dir = test_dir

    def compile_and_run(self, inputs):
        save_graph(self.graph_def, self.config, self.test_dir)
        params = get_params(self.config)
        print(params)
        (output_program, model_weight_file) = CompileTFGraph.generate_code(params)
        prog = Program(output_program, model_weight_file, params, self.test_dir)
        output = prog.run(inputs)
        return output


def assert_almost_equal(tf_output, mpc_tensor, precision):
    if tf_output.shape == (0,):
        return
    np.testing.assert_almost_equal(tf_output.flatten(), mpc_tensor, decimal=precision)
    return
