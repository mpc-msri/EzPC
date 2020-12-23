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


if __name__ == "__main__":
  args = parse_args()
  params = parse_config.get_params(args.config)
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

  athos_dir = os.path.dirname(os.path.abspath(__file__))
  model_abs_path = os.path.abspath(model_name)
  model_abs_dir = os.path.dirname(model_abs_path)
  # Generate graphdef and sizeInfo metadata
  compile_tf.compile(
    model_name, input_tensor_info, output_tensors, scale, save_weights
  )

  # Compile to seedot. Generate AST in model directory
  Athos.process_tf_graph(model_abs_path)

  # Compile to ezpc
  model_base_name = model_name[:-3]
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
  os.system("cp {ezpc} {ezpc_dir}".format(ezpc=ezpc_abs_path, ezpc_dir=ezpc_dir))
  os.chdir(ezpc_dir)
  ezpc_args = ""
  ezpc_args += "--bitlen {bl} --codegen {target} --disable-tac".format(
    bl=lib_bitlength, target=target
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
