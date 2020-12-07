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

Parts of the code in this file is taken from https://github.com/aymericdamien/TensorFlow-Examples/.

'''

from __future__ import print_function
import os, sys
import time
import numpy
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784

# Model weights
W = tf.Variable(tf.constant(0.1, shape=[784,10]))
b = tf.Variable(tf.constant(0.2, shape=[10]))

# Construct model
pred = tf.argmax(tf.matmul(x, W) + b, 1)
init = tf.global_variables_initializer()

def findLabel(oneHotAns):
	for i in range(10):
		if oneHotAns[i] == 1.0:
			return i
	return -1

def saveImg(seqNum, imaged):
	temp = imaged.reshape([28,28])
	# plt.gray()
	plt.imshow(temp)
	plt.savefig("MNIST_test_image" + str(seqNum) + ".png")

with tf.Session() as sess:
	sess.run(init)

	if len(sys.argv)!=2:
		print("Need the mnist image number to run inference on.")
		exit(-1)

	curImageNum = int(sys.argv[1])

	imagex = mnist.test.images[curImageNum:curImageNum+1,:]
	imagey = mnist.test.labels[curImageNum:curImageNum+1,:]
	# saveImg(curImageNum, imagex[0]) # save the image so that so pic viewer can render it
	feed_dict = {x:imagex}

	output_tensor = None
	gg = tf.get_default_graph()
	for node in gg.as_graph_def().node:
		if node.name == 'ArgMax':
			output_tensor = gg.get_operation_by_name(node.name).outputs[0]
	optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)

	evalTensors = [W,b]
	saver = tf.train.Saver(evalTensors)
	saver.restore(sess, './TrainedModel/model')
	
	start_time = time.time()
	outp = sess.run(pred, feed_dict)
	end_time = time.time()

	print("Duration of execution = ", (end_time-start_time))
	print(outp)
	correctLabel = findLabel(imagey[0])
	print("Correct label = " + str(correctLabel))
	print("Dumping of values into .inp file...")
	trainVarsName = []
	for node in optimized_graph_def.node:
		if node.op=="VariableV2":
			trainVarsName.append(node.name)
	trainVars = list(map(lambda x : tf.get_default_graph().get_operation_by_name(x).outputs[0] , trainVarsName))
	DumpTFMtData.dumpImgAndWeightsDataSeparate(sess, imagex[0], trainVars, 'LR_img_{0}.inp'.format(curImageNum), 'LR_weights_{0}.inp'.format(curImageNum), 15)

