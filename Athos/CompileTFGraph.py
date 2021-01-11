'''

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

'''
import argparse
from argparse import RawTextHelpFormatter

import os
import os.path
import json
import sys

import TFCompiler.ProcessTFGraph as Athos
import CompilerScripts.parse_config as parse_config
import CompilerScripts.compile_tf as compile_tf


def parse_args():
  parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
  parser.add_argument(
    "--config",
    required=True,
    type=str,
    help="""Path to the config json file
Config file should be a json in the following format:
{
  // Mandatory options

  "model_name":"model.pb",  // Tensorflow protobuf file to compile.
  "output_tensors":[
  "output1",
  "output2"
  ],
  "target":"PORTHOS2PC",  // Compilation target. ABY/CPP/CPPRING/PORTHOS/PORTHOS2PC
  
  // Optional options
  "scale":10,         // Scaling factor to compile for. Defaults to 12.
  "bitlength":64,       // Bit length to compile for. Defaults to 64.
  "save_weights" : true,  // Save model scaled weights in fixed point. Defaults to false.

  "input_tensors":{           // Name and shape of the input tensors
  "actual_input_1":"224,244,3",     // for the model. Not required if the
  "input2":"2,245,234,3"        // placeholder nodes have shape info.
  },
  "modulo" : 32,      // Modulo to be used for shares. Applicable for 
              // CPPRING/PORTHOS2PC backend. For 
              // PORTHOS2PC + backend=OT => Power of 2 
              // PORTHOS2PC + backend=HE => Prime value."

  "backend" : "OT",     // Backend to be used - OT/HE (default OT). 
              // Only applicable for PORTHOS2PC backend

  "disable_all_hlil_opts" : false,    // Disable all optimizations in HLIL
  "disable_relu_maxpool_opts" : false,  // Disable Relu-Maxpool optimization
  "disable_garbage_collection" : false,   // Disable Garbage Collection optimization
  "disable_trunc_opts" : false      // Disable truncation placement optimization
}
""",
  )
  args = parser.parse_args()
  return args

def generate_code(params):
  # Mandatory
  model_name = params["model_name"]
  input_tensor_info = params["input_tensors"]
  output_tensors = params["output_tensors"]
  scale = 12 if params["scale"] is None else params["scale"]
  bitlength = 64 if params["bitlength"] is None else params["bitlength"]
  target = params["target"]
  save_weights = params["save_weights"]
  save_weights = False if save_weights is None else save_weights

  assert bitlength <= 64 and bitlength >= 1, "Bitlen must be >= 1 and <= 64"
  assert target in [
    "PORTHOS",
    "PORTHOS2PC",
    "ABY",
    "CPP",
    "CPPRING",
  ], "Target must be any of ABY/CPP/CPPRING/PORTHOS/PORTHOS2PC"

  cwd = os.getcwd()
  athos_dir = os.path.dirname(os.path.abspath(__file__))
  model_abs_path = os.path.abspath(model_name)
  model_abs_dir = os.path.dirname(model_abs_path)
  # Generate graphdef and sizeInfo metadata
  weights_path = compile_tf.compile(
    model_name, input_tensor_info, output_tensors, scale, save_weights
  )

  # Compile to seedot. Generate AST in model directory
  Athos.process_tf_graph(model_abs_path)

  # Compile to ezpc
  model_base_name = os.path.basename(model_abs_path)[:-3]
  ezpc_file_name = "{mname}_{bl}_{target}.ezpc".format(
    mname=model_base_name, bl=bitlength, target=target.lower()
  )
  ezpc_abs_path = os.path.join(model_abs_dir, ezpc_file_name)
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
  seedot_args = ""
  seedot_args += "--astFile {}/astOutput.pkl --consSF {} ".format(
    model_abs_dir, scale
  )
  seedot_args += "--bitlen {} --outputFileName {} ".format(bitlength, ezpc_abs_path)
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
    "cat {pre} {common} {post} {ezpc}> {temp}".format(
      pre=pre, common=common, post=post, ezpc=ezpc_abs_path, temp=temp
    )
  )
  os.system("mv {temp} {ezpc}".format(temp=temp, ezpc=ezpc_abs_path))

  modulo = params["modulo"]
  backend = "OT" if params["backend"] is None else params["backend"]
  ezpc_dir = os.path.join(athos_dir, "../EzPC/EzPC/")
  # Copy generated code to the ezpc directory
  os.system("cp {ezpc} {ezpc_dir}".format(ezpc=ezpc_abs_path, ezpc_dir=ezpc_dir))
  os.chdir(ezpc_dir)
  ezpc_args = ""
  ezpc_args += "--bitlen {bl} --codegen {target} --disable-tac ".format(
    bl=bitlength, target=target
  )
  output_name = ezpc_file_name[:-5] + "0.cpp"
  if modulo is not None:
    ezpc_args += "--modulo {} ".format(modulo)
  if target == "PORTHOS2PC":
    ezpc_args += "--backend {} ".format(backend.upper())
    output_name = ezpc_file_name[:-5] + "_{}0.cpp".format(backend.upper())
  if target in ["PORTHOS"]:
    ezpc_args += "--sf {} ".format(scale)

  os.system(
    "eval `opam config env`; ./ezpc.sh {} ".format(ezpc_file_name) + ezpc_args
  )
  os.system(
    "cp {output} {model_dir} ".format(output=output_name, model_dir=model_abs_dir)
  )
  output_file = os.path.join(model_abs_dir, output_name)

  if target == "PORTHOS2PC":
    program_name = model_base_name + "_" + target + "_" + backend + ".out"
  else:
    program_name = model_base_name + "_" + target + ".out"
  program_path = os.path.join(model_abs_dir, program_name)
  os.chdir(model_abs_dir)
  if target in [ "CPP", "CPPRING"]:
    os.system(
      "g++ -O3 -w {file} -o {output}".format(file=output_file, output=program_path)
    )
  elif target == "PORTHOS":
    porthos_src = os.path.join(athos_dir, "..", "Porthos", "src")
    porthos_lib = os.path.join(porthos_src, "build", "lib")
    if os.path.exists(porthos_lib):
      os.system(
        """g++ -O3 -fopenmp -pthread -w -march=native -msse4.1 -maes -mpclmul \
        -mrdseed -fpermissive -fpic -std=c++17 -L {porthos_lib} -I {porthos_headers} {file} \
        -lPorthos-Protocols -lssl -lcrypto -lrt -lboost_system \
        -o {output}""".format(porthos_lib=porthos_lib, porthos_headers=porthos_src,
          file=output_file, output=program_path)
      )
    else:
      print("Not compiling generated code. Please follow the readme and build Porthos.")
  elif target == "PORTHOS2PC":
    sci = os.path.join(athos_dir, "..", "SCI")
    sci_src = os.path.join(sci, "src")
    sci_lib = os.path.join(sci, "build", "lib")
    eigen_path = os.path.join(sci, "extern", "eigen")
    seal_lib_path = os.path.join(sci, "extern", "SEAL", "native", "lib")
    if os.path.exists(sci_lib):
      os.system(
        """g++ -O3 -fpermissive -pthread -w -maes -msse4.1 -mavx -mavx2 -mrdseed \
        -faligned-new -std=c++17 -fopenmp -I {eigen} -I {sci_src} {file} \
        -L {sci_lib} -lSCI-LinearHE -L {seal} -lseal -lssl -lcrypto \
        -o {output}""".format(eigen=eigen_path, sci_src=sci_src,
          file=output_file,sci_lib=sci_lib,seal=seal_lib_path, output=program_path)
      )
    else:
      print("Not compiling generated code. Please follow the readme and build SCI.")

  os.chdir(cwd)
  return (program_path, weights_path)

if __name__ == "__main__":
  args = parse_args()
  params = parse_config.get_params(args.config)
  generate_code(params)