import argparse
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ONNXCompiler'))

from tf_graph_io import *
from tf_graph_trans import *

import common
import numpy as np

def check_operation_exists(graph, tensor_name):
  op_list = [i.name for i in graph.get_operations()]
  return tensor_name in op_list

def gen_random_input(model_fname, input_t_name, scaling_factor, input_shape, dump_numpy):
  if not model_fname.endswith('.pb'):
    sys.exit("Please supply a valid tensorflow protobuf model (.pb extension)")
  else:
    model_name = os.path.basename(model_fname)[:-3]
  print("Loading processed tf graph ", model_fname)
  graph = load_pb(model_fname)

  if not check_operation_exists(graph, input_t_name):
    sys.exit(input_t_name + " input does not exist in the graph")

  input_t = graph.get_operation_by_name(input_t_name).outputs[0]

  # Generate random tensor as input
  inp_shape = input_t.shape.as_list()
  if None in inp_shape:
    if input_shape == []:
      sys.exit("Please supply shape for the input tensor as it is parametric (? dim) for this model. See --help.")
    else:
      inp_shape = input_shape
  rand_inp_t = np.random.rand(*inp_shape)
  (chunk_inp, cnt) = common.numpy_float_array_to_fixed_point_val_str(rand_inp_t, scaling_factor)

  model_dir = os.path.realpath(os.path.dirname(model_fname))
  os.chdir(model_dir)
  f = open(model_name + '_input_fixedpt_scale_' + str(scaling_factor) + '.inp', 'w') 
  f.write(chunk_inp)
  f.close()
  if dump_numpy:
    rand_inp_t.dump(model_name + '_input_fixedpt_scale_' + str(scaling_factor) + '.npy')
  return

def boolean_string(s):
  if s not in {'False', 'True'}:
    raise ValueError('Not a valid boolean string')
  return s == 'True'

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--modelName", required=True, type=str, help="Name of processed tensorflow model (mpc_processed*.pb)")
  parser.add_argument("--inputTensorName", required=True, type=str, help="Name of the input tensor for the model. (Op name, dont add '/:0' suffix)")
  parser.add_argument("--sf", default=12, type=int, help="scaling factor (int)")
  parser.add_argument("--inputTensorShape", type=str, default='', help="Comma separated list of shape for input tensor. eg: \"2,245,234,3\"")
  parser.add_argument("--dumpNumpy", type=boolean_string, default=False, help="Dump model weights in fixedpt {True/False}")
  args = parser.parse_args()
  return args

def get_shape_list(shape_string):
  if shape_string == '':
    return []
  return [int(i) for i in shape_string.split(",")]

if __name__ == '__main__':
  args = parse_args()
  shape_list = get_shape_list(args.inputTensorShape)
  gen_random_input(args.modelName, args.inputTensorName, args.sf, shape_list, args.dumpNumpy)
