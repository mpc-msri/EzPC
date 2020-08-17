'''

Authors: Nishant Kumar.

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

import numpy
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

def get_optimized_graph_def(output_tensor):
  #First save the graph def
  graph_def = tf.get_default_graph().as_graph_def()
  transforms = [
   'remove_nodes(op=Identity)', 
   'strip_unused_nodes',
   'fold_batch_norms',
   'fold_constants(ignore_errors=true)'
  ]
  optimized_graph_def = TransformGraph(graph_def, [], [output_tensor.name], transforms)
  return optimized_graph_def

def save_graph_metadata(output_tensor, sess, feed_dict):
  #First save the graph def
  graph_def = tf.get_default_graph().as_graph_def()
  transforms = [
   'remove_nodes(op=Identity)', 
   'strip_unused_nodes',
   'fold_batch_norms',
   'fold_constants(ignore_errors=true)'
  ]
  optimized_graph_def = TransformGraph(graph_def, [], [output_tensor.name], transforms)
  with open('./graphDef.mtdata', 'w') as f:
    f.write(str(optimized_graph_def))

  # Save size information for tensors on which output depends
  tensors_to_evaluate = []
  tensors_to_evaluate_names = []
  graph = tf.get_default_graph()
  for node in optimized_graph_def.node:
    cur_output = graph.get_operation_by_name(node.name).outputs[0]
    tensors_to_evaluate.append(cur_output)
    tensors_to_evaluate_names.append(node.name)
  tensors_evaluated = sess.run(tensors_to_evaluate, feed_dict)
  tensors_shape = list(map(lambda x : x.shape, tensors_evaluated))

  # Write size info in a file
  with open('./sizeInfo.mtdata','w') as f:
    for ii, curr in enumerate(tensors_to_evaluate_names):
      curShape = tensors_shape[ii]
      f.write(tensors_to_evaluate_names[ii] + ' ')
      for dim in curShape:
        f.write(str(dim)+' ')
      f.write('\n')

  return optimized_graph_def

def updateWeightsForBN(optimized_graph_def, sess, feed_dict={}):
  def findNodeInGraphDefWithName(graphDef, curName):
    for curNode in graphDef.node:
      if curNode.name == curName:
        return curNode
    return None

  print("Updating weights for BN...")

  graph = sess.graph 
  graphDef = optimized_graph_def

  for node in graphDef.node:
      if (node.op == 'FusedBatchNorm' or node.op == 'FusedBatchNormV3'):
        gamma = graph.get_operation_by_name(node.input[1]).outputs[0]
        beta = graph.get_operation_by_name(node.input[2]).outputs[0]
        mu = graph.get_operation_by_name(node.input[3]).outputs[0]
        variance = graph.get_operation_by_name(node.input[4]).outputs[0]

        epsilon = node.attr['epsilon'].f
        rsigma = tf.rsqrt(variance + epsilon)

        sess.run(tf.assign(gamma, gamma*rsigma))
        sess.run(tf.assign(beta, beta - gamma*mu))
        sess.run(tf.assign(mu, tf.zeros(tf.shape(mu))))
        sess.run(tf.assign(variance, tf.fill(tf.shape(variance), 1-epsilon)))

  print("BN weight updation done. Continuing...")

def dumpImageDataInt(imgData, filename, scalingFac, writeMode):
  print("Dumping image data...")
  with open(filename, writeMode) as ff:
    for xx in numpy.nditer(imgData, order='C'):
      ff.write(str(int(xx * (1<<scalingFac))) + ' ')
    ff.write('\n\n')

def dumpTrainedWeightsInt(sess, evalTensors, filename, scalingFac, writeMode, alreadyEvaluated=False):
  print("Dumping trained weights...")
  if alreadyEvaluated: finalParameters = evalTensors
  else: finalParameters = map(lambda x : sess.run(x), evalTensors)
  with open(filename, writeMode) as ff:
    for ii, curParameterVal in enumerate(finalParameters):
      for xx in numpy.nditer(curParameterVal, order='C'):
        ff.write(str(int(xx * (1<<scalingFac))) + ' ')
      ff.write('\n\n')

def dumpTrainedWeightsFloat(sess, evalTensors, filename, writeMode, alreadyEvaluated=False):
  print("Dumping trained weights float...")
  if alreadyEvaluated: finalParameters = evalTensors
  else: finalParameters = map(lambda x : sess.run(x), evalTensors)
  with open(filename, writeMode) as ff:
    for ii, curParameterVal in enumerate(finalParameters):
      for xx in numpy.nditer(curParameterVal, order='C'):
        ff.write((str(xx)) + ' ')
      ff.write('\n\n')

def dumpImgAndWeightsData(sess, imgData, evalTensors, filename, scalingFac, alreadyEvaluated=False):
  print("Starting to dump data...")
  dumpImageDataInt(imgData, filename, scalingFac, 'w')
  dumpTrainedWeightsInt(sess, evalTensors, filename, scalingFac, 'a', alreadyEvaluated=alreadyEvaluated)

def dumpImgAndWeightsDataSeparate(sess, imgData, evalTensors, imgFileName, weightFileName, scalingFac, alreadyEvaluated=False):
  print("Starting to dump data...")
  dumpImageDataInt(imgData, imgFileName, scalingFac, 'w')
  dumpTrainedWeightsInt(sess, evalTensors, weightFileName, scalingFac, 'w', alreadyEvaluated=alreadyEvaluated)

def numpy_float_array_to_float_val_str(input_array):
  chunk = ''
  for val in numpy.nditer(input_array):
    chunk += str(val) + '\n'
  return chunk
