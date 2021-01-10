import argparse
import os.path
import json
import sys

"""
Sample config:
{
// Mandatory
  "model_name":"model.pb",
  "output_tensors":[
  "output1",
  "output2"
  ],
  "scale":10,
  "bitlength":63,
  "target":"SCI",             // ABY/CPP/CPPRING/PORTHOS/SCI
  "save_weights" : true
// Optional
  "input_tensors":{
    "actual_input_1":"2,245,234,3",
    "input2":"2,245,234,3"
  },
  "modulo"
  "backend"
  "disable_all_hlil_opts"
  "disable_relu_maxpool_opts"
  "disable_garbage_collection"
  "disable_trunc_opts"
}
"""


def get_config(config_path):
  if not os.path.isfile(config_path):
    sys.exit("Config file specified does not exist")
  with open(config_path) as f:
    try:
      config = json.load(f)
    except JSONDecodeError as e:
      sys.exit(
        "Error while parsing the config json:\n"
        + e.msg
        + " at line no. "
        + str(e.lineno)
      )
  return config


def get_str_param(config, p_name):
  p = config.get(p_name)
  if p is None:
    sys.exit(p_name + " not specified in config.")
  assert type(p) == str, p_name + " is not a string"
  return p


def get_opt_str_param(config, p_name):
  p = config.get(p_name)
  if p is None:
    return p
  assert type(p) == str, p_name + " is not a string"
  return p


def get_bool_param(config, p_name):
  p = config.get(p_name)
  if p is None:
    sys.exit(p_name + " not specified in config.")
  assert type(p) == bool, p_name + " is not a boolean"
  return p


def get_opt_bool_param(config, p_name):
  p = config.get(p_name)
  if p is None:
    return p
  assert type(p) == bool, p_name + " is not a boolean"
  return p


def get_int_param(config, p_name):
  p = config.get(p_name)
  if p is None:
    sys.exit(p_name + " not specified in config.")
  assert type(p) == int, p_name + " is not an integer"
  return p


def get_opt_int_param(config, p_name):
  p = config.get(p_name)
  if p is None:
    return p
  assert type(p) == int, p_name + " is not an integer"
  return p


def get_str_list_param(config, p_name):
  p = config.get(p_name)
  if p is None:
    sys.exit(p_name + " not specified in config.")
  assert type(p) == list, p_name + "is not a list of strings"
  for i in p:
    assert type(i) == str, p_name + "is not a list of strings"
  return p


def get_opt_param(config, p_name):
  p = config.get(p_name)
  return p


def get_shape_list(shape_string):
  shape = []
  if shape_string == "":
    return shape
  for i in shape_string.split(","):
    assert i.isnumeric(), "Given input shape has non-integer value : {}".format(i)
    shape.append(int(i))
  return shape


def parse_input_tensors(config):
  input_t_info = {}
  p = config.get("input_tensors")
  if p is None:
    return input_t_info
  assert type(p) == dict, "Input tensors should be a dict of name=>shape"
  for name, shape_str in p.items():
    input_t_info[name] = get_shape_list(shape_str)
  return input_t_info


def parse_config(config):
  model_fname = get_str_param(config, "model_name")
  if not model_fname.endswith(".pb"):
    sys.exit(
      model_fname
      + " is not a tensorflow protobuf file. Please supply "
      + "a valid tensorflow protobuf model (.pb extension)"
    )
  if not os.path.isfile(model_fname):
    sys.exit(model_fname + " file does not exist")
  target = get_str_param(config, "target").upper()
  output_tensors = get_str_list_param(config, "output_tensors")
  input_t_info = parse_input_tensors(config)

  save_weights = get_opt_bool_param(config, "save_weights")
  scale = get_opt_int_param(config, "scale")
  bitlen = get_opt_int_param(config, "bitlength")
  modulo = get_opt_int_param(config, "modulo")
  backend = get_opt_str_param(config, "backend")
  disable_hlil_opts = get_opt_bool_param(config, "disable_all_hlil_opts")
  disable_rmo = get_opt_bool_param(config, "disable_relu_maxpool_opts")
  disable_garbage_collection = get_opt_bool_param(
    config, "disable_garbage_collection"
  )
  disable_trunc = get_opt_bool_param(config, "disable_trunc_opts")

  params = {
    "model_name": model_fname,
    "input_tensors": input_t_info,
    "output_tensors": output_tensors,
    "scale": scale,
    "bitlength": bitlen,
    "target": target,
    "save_weights": save_weights,
    "modulo": modulo,
    "backend": backend,
    "disable_all_hlil_opts": disable_hlil_opts,
    "disable_relu_maxpool_opts": disable_rmo,
    "disable_garbage_collection": disable_garbage_collection,
    "disable_trunc_opts": disable_trunc,
  }
  return params


def get_params(config_fname):
  config = get_config(config_fname)
  params = parse_config(config)
  return params
