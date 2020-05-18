import tensorflow as tf
import onnx
from onnx import shape_inference
import keras2onnx

model_filename = 'chest_xray_covid19_model.h5'
output_filename = 'covid_resnet.onnx'
input_h = 224
input_w = 224

tf.keras.backend.set_learning_phase(0)
keras_model = tf.keras.models.load_model(model_filename)
onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

def set_input_dim(onnx_model, idx, val):
  onnx_model.graph.input[0].type.tensor_type.shape.dim[idx].dim_value = val

def get_input_dim(onnx_model, idx):
  return onnx_model.graph.input[0].type.tensor_type.shape.dim[idx].dim_value 

#If input dims are parametric we need to materialize the dims to constants
# N H W C
dims = { "n" : 0, "h" : 1, "w" : 2, "c" : 3}
n = get_input_dim(onnx_model, dims["n"])
h = get_input_dim(onnx_model, dims["h"])
w = get_input_dim(onnx_model, dims["w"])
c = get_input_dim(onnx_model, dims["c"])

if 0 in [n,h,w,c]:
  set_input_dim(onnx_model, dims["n"], 1)
  set_input_dim(onnx_model, dims["h"], input_h)  
  set_input_dim(onnx_model, dims["w"], input_w)  

fixed_model = onnx.shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(fixed_model)
onnx.save_model(fixed_model, output_filename) 
