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

import TFCompiler.ProcessTFGraph as Athos
import CompilerScripts.parse_config as parse_config


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="""Path to the config json file
Config file should be a json in the following format:
{
  //--------------------------- Mandatory options ---------------------------

  "network_name":"ResNet",  // Any network name from Athos/Networks directory (ResNet/DenseNet/ChestXRay/..)
  "target":"SCI",    // Compilation target. ABY/CPP/CPPRING/PORTHOS/SCI


  
  //--------------------------- Optional options ---------------------------
  "scale":10,           // Scaling factor to compile for. DEFAULT=12.
  "bitlength":64,       // Bit length to compile for. DEFAULT=64.

  "modulo" : 32,      // Modulo to be used for shares. Applicable for 
                      // CPPRING/SCI backend. For 
                      // SCI + backend=OT => Power of 2 
                      // SCI + backend=HE => Prime value."

  "backend" : "OT",   // Backend to be used - OT/HE (DEFAULT=OT). 
                      // Only applicable for SCI backend

  "disable_all_hlil_opts" : false,      // Disable all optimizations in HLIL. DEFAULT=false
  "disable_relu_maxpool_opts" : false,  // Disable Relu-Maxpool optimization. DEFAULT=false
  "disable_garbage_collection" : false, // Disable Garbage Collection optimization. DEFAULT=false
  "disable_trunc_opts" : false          // Disable truncation placement optimization. DEFAULT=false
  "run_in_tmux" : false                 // Also run the compiled program in a new tmux session
}
""",
    )
    args = parser.parse_args()
    return args


def generate_code(params, debug=False):
    network_name = params["network_name"]
    assert network_name in [
        "ResNet",
        "DenseNet",
        "SqueezeNetImgNet",
        "SqueezeNetCIFAR10",
    ], "Network must be any of ResNet/DenseNet/SqueezeNetImgNet/SqueezeNetCIFAR10"
    scale = 12 if params["scale"] is None else params["scale"]
    target = params["target"]
    if params["bitlength"] is None:
        if target == "SCI":
            bitlength = 41
        else:
            bitlength = 64
    else:
        bitlength = params["bitlength"]
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
    run_in_tmux = False if params["run_in_tmux"] is None else params["run_in_tmux"]

    assert bitlength <= 64 and bitlength >= 1, "Bitlen must be >= 1 and <= 64"
    assert target in [
        "PORTHOS",
        "SCI",
        "ABY",
        "CPP",
        "CPPRING",
    ], "Target must be any of ABY/CPP/CPPRING/PORTHOS/SCI"

    cwd = os.getcwd()
    athos_dir = os.path.dirname(os.path.abspath(__file__))
    model_abs_dir = os.path.join(athos_dir, "Networks", network_name)
    if not os.path.exists(model_abs_dir):
        sys.exit("Model directory {} does not exist".format(model_abs_dir))

    # Generate graphdef and sizeInfo metadata, and dump model weights
    os.chdir(model_abs_dir)
    os.system("./setup_and_run.sh {scale}".format(scale=scale))
    os.chdir(cwd)
    print(
        "--------------------------------------------------------------------------------"
    )
    print("Compiling model to EzPC")
    print(
        "--------------------------------------------------------------------------------"
    )

    # Compile to seedot. Generate AST in model directory
    print("model_dir = ", model_abs_dir)
    Athos.process_tf_graph(model_abs_dir)

    # Compile to ezpc
    model_base_name = network_name
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
    os.system(
        'cat "{pre}" "{common}" "{post}" "{ezpc}"> "{temp}"'.format(
            pre=pre, common=common, post=post, ezpc=ezpc_abs_path, temp=temp
        )
    )
    os.system('mv "{temp}" "{ezpc}"'.format(temp=temp, ezpc=ezpc_abs_path))

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

    print(
        "--------------------------------------------------------------------------------"
    )
    print("Compiling generated {} code".format(target))
    print(
        "--------------------------------------------------------------------------------"
    )
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
        os.system("rm -rf {build_dir}".format(build_dir=build_dir))
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

    input_path = os.path.join(model_abs_dir, "model_input_scale_{}.inp".format(scale))
    weights_path = os.path.join(
        model_abs_dir, "model_weights_scale_{}.inp".format(scale)
    )
    # program_path
    if run_in_tmux:
        is_tmux_installed = os.system("type tmux > /dev/null")
        if is_tmux_installed != 0:
            print(
                "Not running the program. Tmux is not installed. Please install tmux and run or do the following manually to run."
            )
            return

        print(
            "--------------------------------------------------------------------------------"
        )
        mode = target + " - " + backend if target == "SCI" else target
        print("Running program securely in {} mode".format(mode))
        print(
            "--------------------------------------------------------------------------------"
        )

        sample_networks_dir = os.path.join(
            athos_dir, "CompilerScripts", "sample_networks"
        )
        if target in ["CPP", "CPPRING"]:
            run_script_path = os.path.join(sample_networks_dir, "run_demo_cpp.sh")
        elif target == "PORTHOS":
            run_script_path = os.path.join(sample_networks_dir, "run_demo_3pc.sh")
        elif target == "SCI":
            run_script_path = os.path.join(sample_networks_dir, "run_demo_2pc.sh")
        os.system(
            "{script} {model_dir} {model_binary} {model_input} {model_weight}".format(
                script=run_script_path,
                model_dir=model_abs_dir,
                model_binary=program_path,
                model_input=input_path,
                model_weight=weights_path,
            )
        )
        print(
            "\nAttach to tmux session named {model} to see results (tmux a -t {model})".format(
                model=network_name
            )
        )
    return


if __name__ == "__main__":
    args = parse_args()
    params = parse_config.get_params(args.config, sample_network=True)
    generate_code(params)
