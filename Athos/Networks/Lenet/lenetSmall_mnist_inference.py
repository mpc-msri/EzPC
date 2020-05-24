# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import tempfile

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

FLAGS = None

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 16, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([256, 100])
    b_fc1 = bias_variable([100])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  # with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([100, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv, keep_prob, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.constant(0.25, shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.25, shape=shape)
  return tf.Variable(initial)

def findLabel(oneHotAns):
  for i in range(10):
    if oneHotAns[i] == 1.0:
      return i
  return -1

def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True,seed=1)
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  y_conv, keep_prob, modelWeights = deepnn(x)
  pred = tf.argmax(y_conv, 1)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if len(sys.argv)!=2:
      print("Need the mnist image number to run inference on.")
      exit(-1)

    curImageNum = int(sys.argv[1])
    imagex = mnist.test.images[curImageNum:curImageNum+1,:]
    imagey = mnist.test.labels[curImageNum:curImageNum+1,:]
    keep_prob_value = 1.0
    feed_dict = {x:imagex, y_:imagey, keep_prob:keep_prob_value}
    
    output_tensor = None
    gg = tf.get_default_graph()
    for node in gg.as_graph_def().node:
      # if node.name == 'fc2/add':
      if node.name == 'ArgMax':
        output_tensor = gg.get_operation_by_name(node.name).outputs[0]
    optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)

    saver = tf.train.Saver(modelWeights)
    saver.restore(sess, './TrainedModel/lenetSmallModel')

    start_time = time.time()
    prediction = sess.run([y_conv, keep_prob],feed_dict=feed_dict)
    duration = time.time() - start_time

    print("Duration of execution : ", duration)
    print('Result ::::::::: \n', prediction[0])

    print("Prediction: ", np.argmax(prediction[0]))
    print("Actual label: ", findLabel(imagey[0]))

    trainVarsName = []
    for node in optimized_graph_def.node:
      if node.op=="VariableV2":
        trainVarsName.append(node.name)
    trainVars = list(map(lambda x : tf.get_default_graph().get_operation_by_name(x).outputs[0] , trainVarsName))
    DumpTFMtData.dumpImgAndWeightsDataSeparate(sess, imagex[0], trainVars, 'LenetSmall_img_{0}.inp'.format(curImageNum), 'LenetSmall_weights_{0}.inp'.format(curImageNum), 15)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)