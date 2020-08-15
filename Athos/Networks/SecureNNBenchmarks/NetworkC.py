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
# This is the network C from SecureNN written in Tensorflow.
# The network description is taken from the public implementation of SecureNN.
#   [https://github.com/snwagh/securenn-public/blob/master/src/main.cpp]
#

import os, sys
import numpy as np
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  # initial = tf.truncated_normal(shape, stddev=0.1)
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None,784])

#conv1
w_conv1 = weight_variable([5,5,1,20])
conv1 = tf.nn.conv2d(tf.reshape(x, [-1,28,28,1]), w_conv1, strides=[1,1,1,1], padding='VALID')
relu1 = tf.nn.relu(conv1)
maxpool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

#conv2
w_conv2 = weight_variable([5,5,20,50])
conv2 = tf.nn.conv2d(maxpool1, w_conv2, strides=[1,1,1,1], padding='VALID')
relu2 = tf.nn.relu(conv2)
maxpool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

#fc1
w_fc1 = weight_variable([800,500])
b_fc1 = bias_variable([500])
fc1 = tf.matmul(tf.reshape(maxpool2, [-1,800]), w_fc1) + b_fc1
relu3 = tf.nn.relu(fc1)

#fc2
w_fc2 = weight_variable([500,10])
b_fc2 = bias_variable([10])
fc2 = tf.matmul(relu3, w_fc2) + b_fc2

finalOut = tf.argmax(fc2, 1)

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
