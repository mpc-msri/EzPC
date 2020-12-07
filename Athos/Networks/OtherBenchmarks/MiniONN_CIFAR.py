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
# This is a NN over the CIFAR-10 dataset, used in the MiniONN paper.
# Its taken from figure 13 of MiniONN paper [https://eprint.iacr.org/2017/452.pdf].
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

x = tf.placeholder(tf.float32, [None, 32, 32, 3])

# 1 
w_conv1 = weight_variable([3,3,3,64])
conv1 = tf.nn.conv2d(x, w_conv1, strides=[1,1,1,1],padding='SAME')
relu2 = tf.nn.relu(conv1)

# 3
w_conv3 = weight_variable([3,3,64,64])
conv3 = tf.nn.conv2d(relu2, w_conv3, strides=[1,1,1,1],padding='SAME')
relu4 = tf.nn.relu(conv3)

# 5
avgpool5 = tf.nn.avg_pool(relu4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 6
w_conv6 = weight_variable([3,3,64,64])
conv6 = tf.nn.conv2d(avgpool5, w_conv6, strides=[1,1,1,1], padding='SAME')
relu7 = tf.nn.relu(conv6)

# 8
w_conv8 = weight_variable([3,3,64,64])
conv8 = tf.nn.conv2d(relu7, w_conv8, strides=[1,1,1,1], padding='SAME')
relu9 = tf.nn.relu(conv8)

# 10
avgpool10 = tf.nn.avg_pool(relu9, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

# 11
w_conv11 = weight_variable([3,3,64,64])
conv11 = tf.nn.conv2d(avgpool10, w_conv11, strides=[1,2,2,1], padding='SAME')
relu12 = tf.nn.relu(conv11)

# 13
w_conv13 = weight_variable([1,1,64,64])
conv13 = tf.nn.conv2d(relu12, w_conv13, strides=[1,1,1,1], padding='SAME')
relu14 = tf.nn.relu(conv13)

# 15
w_conv15 = weight_variable([1,1,64,16])
conv15 = tf.nn.conv2d(relu14, w_conv15, strides=[1,1,1,1], padding='SAME')
relu16 = tf.nn.relu(conv15)

# # 17
w_fc17 = weight_variable([1024,10])
b_fc17 = bias_variable([10])
inpfc17 = tf.reshape(relu16, [-1,1024])
fc17 = tf.matmul(inpfc17, w_fc17) + b_fc17

finalOut = tf.argmax(fc17,1)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  imgData = np.full((1,32,32,3), 0.1)
  feed_dict = {x: imgData}
  
  pred = sess.run(finalOut, feed_dict=feed_dict)
  print(pred)
  
  output_tensor = None
  gg = tf.get_default_graph()
  for node in gg.as_graph_def().node:
    if node.name == 'ArgMax':
      output_tensor = gg.get_operation_by_name(node.name).outputs[0]
  optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)

  
