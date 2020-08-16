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

#
# This is the network A from SecureNN written in Tensorflow.
# Also, first presented in SecureML [https://eprint.iacr.org/2017/396.pdf].
# The network description is taken from fig 10 of MiniONN paper [https://eprint.iacr.org/2017/452.pdf].
#

import os, sys
import numpy as np
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

useRELUActivation = False

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  # initial = tf.truncated_normal(shape, stddev=0.1)
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 784])

#fc1
with tf.name_scope('fc1'):
  w_fc1 = weight_variable([784,128])
  b_fc1 = bias_variable([128])
  outp1 = tf.matmul(x, w_fc1) + b_fc1

if useRELUActivation:
  actv1 = tf.nn.relu(outp1)
else:
  actv1 = tf.square(outp1)

with tf.name_scope('fc2'):
  w_fc2 = weight_variable([128,128])
  b_fc2 = bias_variable([128])
  outp2 = tf.matmul(actv1, w_fc2) + b_fc2

if useRELUActivation:
  actv2 = tf.nn.relu(outp2)
else:
  actv2 = tf.square(outp2)

with tf.name_scope('fc3'):
  w_fc3 = weight_variable([128,10])
  b_fc3 = bias_variable([10])
  outp3 = tf.matmul(actv2, w_fc3) + b_fc3

finalOut = tf.argmax(outp3,1)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  imgData = [[0.02]*784]
  feed_dict = {x: imgData}
  
  pred = sess.run(finalOut, feed_dict=feed_dict)
  print(pred)
  
  output_tensor = None
  gg = tf.get_default_graph()
  for node in gg.as_graph_def().node:
    if node.name == 'ArgMax':
      output_tensor = gg.get_operation_by_name(node.name).outputs[0]
  optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)

  
