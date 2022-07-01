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
import argparse
from argparse import RawTextHelpFormatter

import os
import os.path
import json
import sys
from zipfile import ZipFile

# import TFCompiler.ProcessTFGraph as Athos
import CompilerScripts.parse_config as parse_config
import ONNXCompiler.process_onnx as compile_onnx


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--role",
        required=True,
        type=str,
        choices=["server", "client"],
        help="""
Choose server if you are the model owner.
Choose client if you are the data owner.
""",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="""Path to the config json file
Config file should be a json in the following format:
{
  //--------------------------- Mandatory options ---------------------------

  "model_name":"model.onnx",  // ONNX file to compile.
  "output_tensors":[
    "output1",
    "output2"
  ],
  "target":"SCI",  // Compilation target. ABY/CPP/CPPRING/PORTHOS/SCI


  
  //--------------------------- Optional options ---------------------------
  "scale":10,             // Scaling factor to compile for. DEFAULT=12.
  "bitlength":64,         // Bit length to compile for. DEFAULT=64.
  "save_weights" : true,  // Save model scaled weights in fixed point. DEFAULT=true.

  "input_tensors":{               // Name and shape of the input tensors
    "actual_input_1":"224,244,3", // for the model. Not required if the
    "input2":"2,245,234,3"        // placeholder nodes have shape info in the .pb file.
  },
  "modulo" : 32,      // Modulo to be used for shares. Applicable for 
                      // CPPRING/SCI backend. For 
                      // SCI + backend=OT => Power of 2 
                      // SCI + backend=HE => Prime value."

  "backend" : "OT",   // Backend to be used - OT/HE (default OT). 
                      // Only applicable for SCI backend

  "disable_all_hlil_opts" : false,      // Disable all optimizations in HLIL. DEFAULT=false
  "disable_relu_maxpool_opts" : false,  // Disable Relu-Maxpool optimization. DEFAULT=false
  "disable_garbage_collection" : false, // Disable Garbage Collection optimization. DEFAULT=false
  "disable_trunc_opts" : false          // Disable truncation placement optimization. DEFAULT=false
}
""",
    )
    args = parser.parse_args()
    return args


def generate_code(params, role, debug=False):
    model_path = params["model_name"]
    input_tensor_info = params["input_tensors"]
    output_tensors = params["output_tensors"]
    scale = 12 if params["scale"] is None else params["scale"]
    target = params["target"]
    if params["bitlength"] is None:
        if target == "SCI":
            bitlength = 41
        else:
            bitlength = 64
    else:
        bitlength = params["bitlength"]
        # CPP currently only supports 32 and 64 bitwdith
        if target == "CPP":
            bitlength = 64 if bitlength > 32 else 32
    save_weights = True if params["save_weights"] is None else params["save_weights"]
    disable_all_hlil_opts = (
        False
        if params["disable_all_hlil_opts"] is None
        else params["disable_all_hlil_opts"]
    )
    disable_relu_maxpool_opts = (
        False
        if params["disable_relu_maxpool_opts"] is None
        else params["disable_relu_maxpool_opts"]
    )
    disable_garbage_collection = (
        False
        if params["disable_garbage_collection"] is None
        else params["disable_garbage_collection"]
    )
    disable_trunc_opts = (
        False if params["disable_trunc_opts"] is None else params["disable_trunc_opts"]
    )
    modulo = params["modulo"]
    backend = "OT" if params["backend"] is None else params["backend"]
    assert bitlength <= 64 and bitlength >= 1, "Bitlen must be >= 1 and <= 64"
    assert target in [
        "PORTHOS",
        "SCI",
        "ABY",
        "CPP",
        "CPPRING",
        "FSS",
    ], "Target must be any of ABY/CPP/FSS/CPPRING/PORTHOS/SCI"

    cwd = os.getcwd()
    athos_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.basename(model_path)
    model_abs_path = os.path.abspath(model_path)
    model_abs_dir = os.path.dirname(model_abs_path)

    pruned_model_path = os.path.join(model_abs_dir, "optimised_" + model_name)

    if role == "server":
        # Compile to seedot. Generate AST in model directory
        weights_path = compile_onnx.compile(
            model_path, input_tensor_info, output_tensors, scale, save_weights, role
        )
        # Zip the pruned model, sizeInfo to send to client
        file_list = [pruned_model_path]
        if "config_name" in params:
            file_list.append(params["config_name"])
        zip_path = os.path.join(model_abs_dir, "client.zip")
        with ZipFile(zip_path, "w") as zip:
            for file in file_list:
                zip.write(file, os.path.basename(file))
    else:
        weights_path = compile_onnx.compile(
            pruned_model_path,
            input_tensor_info,
            output_tensors,
            scale,
            save_weights,
            role,
        )

    # Compile to ezpc
    model_base_name = model_name[:-5]
    ezpc_file_name = "{mname}_{bl}_{target}.ezpc".format(
        mname=model_base_name, bl=bitlength, target=target.lower()
    )
    ezpc_abs_path = os.path.join(model_abs_dir, ezpc_file_name)

    seedot_args = ""
    seedot_args += '--astFile "{}/astOutput.pkl" --consSF {} '.format(
        model_abs_dir, scale
    )
    seedot_args += '--bitlen {} --outputFileName "{}" '.format(bitlength, ezpc_abs_path)
    seedot_args += "--disableAllOpti {} ".format(disable_all_hlil_opts)
    seedot_args += "--disableRMO {} ".format(disable_relu_maxpool_opts)
    seedot_args += "--disableLivenessOpti {} ".format(disable_garbage_collection)
    seedot_args += "--disableTruncOpti {} ".format(disable_trunc_opts)

    seedot_script = os.path.join(athos_dir, "SeeDot", "SeeDot.py")
    print("python3 {} ".format(seedot_script) + seedot_args)
    os.system("python3 {} ".format(seedot_script) + seedot_args)

    # Add library functions
    if target in ["ABY", "CPPRING"]:
        library = "cpp"
    else:
        library = target.lower()

    lib_bitlength = 64 if bitlength > 32 else 32
    library_dir = os.path.join(athos_dir, "TFEzPCLibrary")
    common = os.path.join(library_dir, "Library{}_common.ezpc".format(lib_bitlength))
    if library == "cpp":
        pre = os.path.join(
            library_dir, "Library{}_{}_pre.ezpc".format(lib_bitlength, library)
        )
        post = os.path.join(
            library_dir, "Library{}_{}_post.ezpc".format(lib_bitlength, library)
        )
    else:
        pre = os.path.join(
            library_dir, "Library{}_{}.ezpc".format(lib_bitlength, library)
        )
        post = ""
    temp = os.path.join(model_abs_dir, "temp.ezpc")
    if library == "fss":
        os.system(
            'cat "{pre}" "{ezpc}"> "{temp}"'.format(
                pre=pre, common=common, post=post, ezpc=ezpc_abs_path, temp=temp
            )
        )
    else:
        os.system(
            'cat "{pre}" "{common}" "{post}" "{ezpc}"> "{temp}"'.format(
                pre=pre, common=common, post=post, ezpc=ezpc_abs_path, temp=temp
            )
        )
    os.system('mv "{temp}" "{ezpc}"'.format(temp=temp, ezpc=ezpc_abs_path))
    if library == "fss":
        os.system("fssc --bitlen {bl} --disable-tac {ezpc}".format(bl=bitlength, ezpc=ezpc_abs_path))
        print("\n\nGenerated binary: {mb}.out".format(mb=model_base_name))
        program_name = model_base_name + "_" + target + ".out"
        program_path = os.path.join(model_abs_dir, program_name)
    else:
        ezpc_dir = os.path.join(athos_dir, "../EzPC/EzPC/")
        # Copy generated code to the ezpc directory
        os.system('cp "{ezpc}" "{ezpc_dir}"'.format(ezpc=ezpc_abs_path, ezpc_dir=ezpc_dir))
        os.chdir(ezpc_dir)
        ezpc_args = ""
        ezpc_args += "--bitlen {bl} --codegen {target} --disable-tac ".format(
            bl=bitlength, target=target
        )
        output_name = ezpc_file_name[:-5] + "0.cpp"
        if modulo is not None:
            ezpc_args += "--modulo {} ".format(modulo)
        if target == "SCI":
            ezpc_args += "--backend {} ".format(backend.upper())
            output_name = ezpc_file_name[:-5] + "_{}0.cpp".format(backend.upper())
        if target in ["PORTHOS"]:
            ezpc_args += "--sf {} ".format(scale)

        os.system(
            'eval `opam config env`; ./ezpc.sh "{}" '.format(ezpc_file_name) + ezpc_args
        )
        os.system(
            'mv "{output}" "{model_dir}" '.format(
                output=output_name, model_dir=model_abs_dir
            )
        )
        os.system('rm "{}"'.format(ezpc_file_name))
        output_file = os.path.join(model_abs_dir, output_name)

        print("Compiling generated code to {target} target".format(target=target))
        if target == "SCI":
            program_name = model_base_name + "_" + target + "_" + backend + ".out"
        else:
            program_name = model_base_name + "_" + target + ".out"
        program_path = os.path.join(model_abs_dir, program_name)
        os.chdir(model_abs_dir)
        if debug:
            opt_flag = "-O0 -g"
        else:
            opt_flag = "-O3"
        if target in ["CPP", "CPPRING"]:
            os.system(
                'g++ {opt_flag} -w "{file}" -o "{output}"'.format(
                    file=output_file, output=program_path, opt_flag=opt_flag
                )
            )
        elif target == "PORTHOS":
            porthos_src = os.path.join(athos_dir, "..", "Porthos", "src")
            porthos_lib = os.path.join(porthos_src, "build", "lib")
            if os.path.exists(porthos_lib):
                os.system(
                    """g++ {opt_flag} -fopenmp -pthread -w -march=native -msse4.1 -maes -mpclmul \
            -mrdseed -fpermissive -fpic -std=c++17 -L \"{porthos_lib}\" -I \"{porthos_headers}\" \"{file}\" \
            -lPorthos-Protocols -lssl -lcrypto -lrt -lboost_system \
            -o \"{output}\"""".format(
                        porthos_lib=porthos_lib,
                        porthos_headers=porthos_src,
                        file=output_file,
                        output=program_path,
                        opt_flag=opt_flag,
                    )
                )
            else:
                print(
                    "Not compiling generated code. Please follow the readme and build Porthos."
                )
        elif target == "SCI":
            sci_install = os.path.join(athos_dir, "..", "SCI", "build", "install")
            build_dir = "build_dir"
            os.system("rm -r {build_dir}".format(build_dir=build_dir))
            os.mkdir(build_dir)
            os.chdir(build_dir)
            cmake_file = """
                cmake_minimum_required (VERSION 3.0)
                project (BUILD_IT)
                find_package(SCI REQUIRED PATHS \"{sci_install}\")
                add_executable({prog_name} {src_file})
                target_link_libraries({prog_name} SCI::SCI-{backend})
            """.format(
                sci_install=sci_install,
                prog_name=program_name,
                src_file=output_file,
                backend=backend.upper(),
            )
            with open("CMakeLists.txt", "w") as f:
                f.write(cmake_file)

            if os.path.exists(sci_install):
                ret = os.system("cmake --log-level=ERROR .")
                if ret != 0:
                    sys.exit("Compilation of generated code failed. Exiting...")
                ret = os.system("cmake --build . --parallel")
                if ret != 0:
                    sys.exit("Compilation of generated code failed. Exiting...")
                os.system(
                    "mv {tmp_prog} {prog_path}".format(
                        tmp_prog=program_name, prog_path=program_path
                    )
                )
                os.chdir("..")
                os.system("rm -r {build_dir}".format(build_dir=build_dir))
            else:
                print(
                    "Not compiling generated code. Please follow the readme and build and install SCI."
                )

        os.chdir(cwd)
        print("\n\nGenerated binary: {}".format(program_path))
    if role == "server":
        print("\n\nUse as input to server (model weights): {}".format(weights_path))
        print("Share {} file with the client".format(zip_path))
    return (program_path, weights_path)


if __name__ == "__main__":
    args = parse_args()
    params = parse_config.get_params(args.config)
    params["config_name"] = args.config

    generate_code(params, args.role)