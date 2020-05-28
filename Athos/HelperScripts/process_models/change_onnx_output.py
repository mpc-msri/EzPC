'''

Authors: Pratik Bhatu.

Copyright:
Copyright (c) 2020 Microsoft Research
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

import onnx
import onnxruntime
import numpy as np
from onnx import helper, shape_inference, checker
from onnx import ValueInfoProto, ModelProto, TensorProto
import os

model_name = "shufflenet_may17.onnx"
output_model_name = "processed_" + model_name
inputs = ['data']
nodes_to_remove = ['LabelSelector', 'LabelIndexExtractor', 'ZipMap',
                   'activation37']
new_output_names = ['fc']
batch_size = 1

def fix_shape(shape_list, batch_size):
  if 'None' not in shape_list:
    return shape_list
  else:
    shape_list[0] = batch_size
    assert ('None' not in shape_list) , """Other than batch size there are input
                                        params with unkown dimension"""
    return shape_list

def fix_inp_shape(inp, batch_size):
  if inp.type.tensor_type.shape.dim[0].dim_param == 'None':
    inp.type.tensor_type.shape.dim[0].dim_value = batch_size
  return

def get_np_type_from_onnxruntime(typ_str):
  np_types = {
              'tensor(float)' : np.float32,
              'tensor(float64)' : np.float64,
              'tensor(int)' : np.int32,
              'tensor(int64)' : np.int64
             }
  return np_types[typ_str]

def get_onnx_type(arr):
  onnx_types = {
                np.float32 : TensorProto.FLOAT,
                np.float64 : TensorProto.DOUBLE,
                np.int32 : TensorProto.INT32,
                np.int64 : TensorProto.INT64
               }
  return onnx_types[arr.dtype.type]
  

model = onnx.load(model_name)
# 1. Inputs to remove
# Inputs to dead nodes should not show up as inputs for the model
# and also not in the initialization list.
inputs_to_remove = [ inp for i in model.graph.node 
                     if i.name in nodes_to_remove for inp in i.input ]
new_inputs = [ i for i in model.graph.input if i.name not in inputs_to_remove ]

# Fix batch size
fix_inp_shape(new_inputs[0], batch_size)

# 2. Remove their initializers
new_initializers = [ init for init in model.graph.initializer
                     if init.name not in nodes_to_remove
                     and init.name not in inputs_to_remove ]

# 3. Remove nodes
new_nodes = [ n for n in model.graph.node if n.name not in nodes_to_remove ]


# Get Ouput Tensor Types to create ValueInfo for output info
# by running model on dummy input
temp_model = ModelProto()
temp_model.CopyFrom(model)
for i in new_output_names:
  op = ValueInfoProto()
  op.name = i
  temp_model.graph.output.append(op)
onnx.save(temp_model, '__temp.onnx')
sess = onnxruntime.InferenceSession('__temp.onnx')
sess_inps = sess.get_inputs()
input_dict = {}
for i in sess_inps:
  shape = fix_shape(i.shape, batch_size)
  typ = get_np_type_from_onnxruntime(i.type)
  input_dict[i.name] = np.random.rand(*shape).astype(typ)

output_tensors = sess.run(new_output_names, input_dict)
if os.path.exists("__temp.onnx"):
  os.remove("__temp.onnx")

# 4. Create new output list
new_outputs = [] 
for i in range(0,len(new_output_names)):
  name = new_output_names[i]  
  typ = get_onnx_type(output_tensors[i]) 
  shape = output_tensors[i].shape 
  val_info = helper.make_tensor_value_info(name, typ, shape) 
  new_outputs.append(val_info)

new_graph = helper.make_graph(new_nodes,
                              model.graph.name,
                              new_inputs,
                              new_outputs,
                              initializer=new_initializers,
                              doc_string=model.graph.doc_string,
                              value_info=model.graph.value_info)
new_model = helper.make_model(new_graph,
                              ir_version=model.ir_version,
                              doc_string=model.doc_string, 
                              model_version=model.model_version,
                              domain=model.domain,
                              producer_name='MPCOpRemover')
new_model.metadata_props.extend(model.metadata_props)
new_model.opset_import.pop()
new_model.opset_import.extend(model.opset_import)
onnx.save(new_model, 'processed_'+model_name)
