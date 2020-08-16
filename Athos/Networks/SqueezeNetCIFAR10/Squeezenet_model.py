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

** 
Parts of this code including the model itself, the training code and some other parts
were taken from https://github.com/kaizouman/tensorsandbox/tree/master/cifar10/models/squeeze
**

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import Util
import time
import numpy
import matplotlib
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData
from argparse import ArgumentParser

class SqueezeNet1Orig:
	def __init__(self):
		self.all_weights = []

	def inference(self, images):
		# conv1
		conv1 = self.conv_layer(images,
								size=3,
								filters=64,
								stride=1,
								decay=False,
								name='conv1')

		# pool1
		pool1 = self.pool_layer(conv1,
								size=3,
								stride=2,
								name='pool1')

		# fire2
		fire2 = self.fire_layer(pool1, 32, 64, 64, decay=False, name='fire2')

		# fire3
		fire3 = self.fire_layer(fire2, 32, 64, 64, decay=False, name='fire3')

		# pool2
		pool2 = self.pool_layer(fire3,
								size=3,
								stride=2,
								name='pool2')

		# fire4
		fire4 = self.fire_layer(pool2, 32, 128, 128, decay=False, name='fire4')

		# fire5
		fire5 = self.fire_layer(fire4, 32, 128, 128, decay=False, name='fire5')

		# Final squeeze to get ten classes
		conv2 = self.conv_layer(fire5,
								size=1,
								filters=10,
								stride=1,
								decay=False,
								name='squeeze')

		# Average pooling on spatial dimensions
		predictions = self.avg_layer(conv2, name='avg_pool')

		return predictions

	def pool_layer(self, inputs, size, stride, name):
		with tf.variable_scope(name) as scope:
			outputs = tf.nn.max_pool(inputs,
									ksize=[1,size,size,1],
									strides=[1,stride,stride,1],
									padding='SAME',
									name=name)
		return outputs

	def fire_layer(self, inputs, s1x1, e1x1, e3x3, name, decay=False):
		with tf.variable_scope(name) as scope:
			# Squeeze sub-layer
			squeezed_inputs = self.conv_layer(inputs,
											  size=1,
											  filters=s1x1,
											  stride=1,
											  decay=decay,
											  name='s1x1')

			# Expand 1x1 sub-layer
			e1x1_outputs = self.conv_layer(squeezed_inputs,
										   size=1,
										   filters=e1x1,
										   stride=1,
										   decay=decay,
										   name='e1x1')

			# Expand 3x3 sub-layer
			e3x3_outputs = self.conv_layer(squeezed_inputs,
										   size=3,
										   filters=e3x3,
										   stride=1,
										   decay=decay,
										   name='e3x3')

			# Concatenate outputs along the last dimension (channel)
			return tf.concat([e1x1_outputs, e3x3_outputs], 3)

	def avg_layer(self, inputs, name):
		w = inputs.get_shape().as_list()[1]
		h = inputs.get_shape().as_list()[2]
		c = inputs.get_shape().as_list()[3]
		with tf.variable_scope(name) as scope:
			# Use current spatial dimensions as Kernel size to produce a scalar
			avg = tf.nn.avg_pool(inputs,
					 ksize=[1,w,h,1],
					 strides=[1,1,1,1],
					 padding='VALID',
					 name=scope.name)
		# Reshape output to remove spatial dimensions reduced to one
		return tf.reshape(avg, shape=[-1,c])

	def conv_layer(self, inputs, size, filters, stride, decay, name):
		channels = inputs.shape[3]
		shape = [size, size, channels, filters]
		with tf.variable_scope(name + '/conv') as scope:
			weights = self._get_weights_var('weights', shape=shape, decay=decay)
			biases = self.get_cons_variable([filters], 0.0)
			conv = tf.nn.conv2d(inputs,
					weights,
					strides=[1,stride,stride,1],
			padding='SAME')

			pre_activation = tf.nn.bias_add(conv, biases)

			outputs= tf.nn.relu(pre_activation, name=scope.name)

		return outputs

	def get_cons_variable(self, shape, val):
		initial = tf.constant(val, shape=shape)
		temp = tf.Variable(initial)
		self.all_weights.append(temp)
		return temp

	def _get_weights_var(self, name, shape, decay=False):
		"""Helper to create an initialized Variable with weight decay.

		The Variable is initialized using a normal distribution whose variance
		is provided by the xavier formula (ie inversely proportional to the number
		of inputs)

		Args:
			name: name of the tensor variable
			shape: the tensor shape
			decay: a boolean indicating if we apply decay to the tensor weights
			using a regularization loss

		Returns:
			Variable Tensor
		"""
		# Declare an initializer for this variable
		initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
		# Declare variable (it is trainable by default)
		var = tf.get_variable(name=name,
							  shape=shape,
							  initializer=initializer,
							  dtype=tf.float32)
		if decay:
			# We apply a weight decay to this tensor var that is equal to the
			# model weight decay divided by the tensor size
			weight_decay = self.wd
			for x in shape:
				weight_decay /= x
			# Weight loss is L2 loss multiplied by weight decay
			weight_loss = tf.multiply(tf.nn.l2_loss(var),
									  weight_decay,
									  name='weight_loss')
			# Add weight loss for this variable to the global losses collection
			tf.add_to_collection('losses', weight_loss)

		self.all_weights.append(var)
		return var

class SqueezeNet1:
	def __init__(self, use_cons_init):
		self.all_weights = []
		self.debug_weights = []
		self.use_cons_init = use_cons_init

	def inference(self, images):
		# conv1
		conv1 = self.conv_layer(images,
								size=3,
								filters=64,
								stride=1,
								decay=False,
								name='conv1')

		# pool1
		pool1 = self.pool_layer(conv1,
								size=3,
								stride=2,
								name='pool1')

		# fire2
		fire2 = self.fire_layer(pool1, 32, 64, 64, decay=False, name='fire2')

		# fire3
		fire3 = self.fire_layer(fire2, 32, 64, 64, decay=False, name='fire3')

		# pool2
		pool2 = self.pool_layer(fire3,
								size=3,
								stride=2,
								name='pool2')

		# fire4
		fire4 = self.fire_layer(pool2, 32, 128, 128, decay=False, name='fire4')

		# fire5
		fire5 = self.fire_layer(fire4, 32, 128, 128, decay=False, name='fire5')

		# Final squeeze to get ten classes
		conv2 = self.conv_layer(fire5,
								size=1,
								filters=10,
								stride=1,
								decay=False,
								name='squeeze')

		# Average pooling on spatial dimensions
		predictions = self.avg_layer(conv2, name='avg_pool')

		return predictions

	def pool_layer(self, inputs, size, stride, name):
		with tf.variable_scope(name) as scope:
			outputs = tf.nn.max_pool(inputs,
									ksize=[1,size,size,1],
									strides=[1,stride,stride,1],
									padding='SAME',
									name=name)
		return outputs

	def fire_layer(self, inputs, s1x1, e1x1, e3x3, name, decay=False):
		with tf.variable_scope(name) as scope:
			# Squeeze sub-layer
			squeezed_inputs = self.conv_layer(inputs,
											  size=1,
											  filters=s1x1,
											  stride=1,
											  decay=decay,
											  name='s1x1')

			# Expand 1x1 sub-layer
			e1x1_outputs = self.conv_layer(squeezed_inputs,
										   size=1,
										   filters=e1x1,
										   stride=1,
										   decay=decay,
										   name='e1x1')

			# Expand 3x3 sub-layer
			e3x3_outputs = self.conv_layer(squeezed_inputs,
										   size=3,
										   filters=e3x3,
										   stride=1,
										   decay=decay,
										   name='e3x3')

			# Concatenate outputs along the last dimension (channel)
			return tf.concat([e1x1_outputs, e3x3_outputs], 3)

	def avg_layer(self, inputs, name):
		w = inputs.get_shape().as_list()[1]
		h = inputs.get_shape().as_list()[2]
		c = inputs.get_shape().as_list()[3]
		with tf.variable_scope(name) as scope:
			# Use current spatial dimensions as Kernel size to produce a scalar
			avg = tf.nn.avg_pool(inputs,
					 ksize=[1,w,h,1],
					 strides=[1,1,1,1],
					 padding='VALID',
					 name=scope.name)
		# Reshape output to remove spatial dimensions reduced to one
		return tf.reshape(avg, shape=[-1,c])

	def conv_layer(self, inputs, size, filters, stride, decay, name):
		channels = inputs.shape[3]
		shape = [size, size, channels, filters]
		with tf.variable_scope(name + '/conv') as scope:
			# For getting performance numbers, don't need to use the actual activations - just use constant activations
			if self.use_cons_init:
				weights = self.get_cons_variable(shape, 0.01)
			else:
				weights = self._get_weights_var('weights', shape=shape, decay=decay)

			biases = self.get_cons_variable([filters], 0.0)
			conv = tf.nn.conv2d(inputs,
								weights,
								strides=[1,stride,stride,1],
								padding='SAME')

			pre_activation = tf.nn.bias_add(conv, biases)
			outputs= tf.nn.relu(pre_activation, name=scope.name)

		return outputs

	def get_cons_variable(self, shape, val):
		initial = tf.constant(val, shape=shape)
		temp = tf.Variable(initial)
		self.all_weights.append(temp)
		return temp

	def _get_weights_var(self, name, shape, decay=False):
		"""Helper to create an initialized Variable with weight decay.

		The Variable is initialized using a normal distribution whose variance
		is provided by the xavier formula (ie inversely proportional to the number
		of inputs)

		Args:
			name: name of the tensor variable
			shape: the tensor shape
			decay: a boolean indicating if we apply decay to the tensor weights
			using a regularization loss

		Returns:
			Variable Tensor
		"""
		# Declare an initializer for this variable
		initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
		# Declare variable (it is trainable by default)
		var = tf.get_variable(name=name,
							  shape=shape,
							  initializer=initializer,
							  dtype=tf.float32)
		if decay:
			# We apply a weight decay to this tensor var that is equal to the
			# model weight decay divided by the tensor size
			weight_decay = self.wd
			for x in shape:
				weight_decay /= x
			# Weight loss is L2 loss multiplied by weight decay
			weight_loss = tf.multiply(tf.nn.l2_loss(var),
									  weight_decay,
									  name='weight_loss')
			# Add weight loss for this variable to the global losses collection
			tf.add_to_collection('losses', weight_loss)

		self.all_weights.append(var)
		return var

def train(sqn, save_model_path):
	print('Starting train...')

	# Hyper parameters
	epochs = 1
	batch_size = 128
	keep_probability = 0.7
	learning_rate = 0.001
	n_batches = 5 #CIFAR10 dataset in the python version has 5 batches

	x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
	y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	logits = sqn.inference(x)

	# Loss and Optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	l2Loss = sum(list(map(lambda x: tf.nn.l2_loss(x), sqn.all_weights)))
	beta = 1e-5
	cost = cost + beta*l2Loss
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Accuracy
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

	valid_features, valid_labels = Util.load_preprocess_validation_data()
	testing_features, testing_labels = Util.load_preprocess_testing_data()

	print('Training now...')
	with tf.Session() as sess:
		# Initializing the variables
		sess.run(tf.global_variables_initializer())

		# Training cycle
		for epoch in range(epochs):
			# Loop over all batches
			for batch_i in range(1, n_batches + 1):
				for batch_features, batch_labels in Util.load_preprocess_training_batch(batch_i, batch_size):
					sess.run(optimizer, feed_dict={x: batch_features,
												   y: batch_labels,
												   keep_prob: keep_probability
												   })

				print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
				
				# Print stats
				loss = sess.run(cost, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
				train_acc = sess.run(accuracy, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
				valid_acc = sess.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: keep_probability})
				testing_acc = sess.run(accuracy, feed_dict={x: testing_features, y: testing_labels, keep_prob: keep_probability})
				print('Loss: {:>10.4f} Train Acc: {:.6f} Validation Accuracy: {:.6f} Testing Acc: {:.6f}'.format(loss, train_acc, valid_acc, testing_acc))

			if (epoch % 10 == 0):
				# Save Model
				saver = tf.train.Saver()
				save_path = saver.save(sess, save_model_path)

#outputArgMax should only be used when findAcc is False
def infer(sqn, sess, images, labels, restoreModelPath, findAccOrArgMaxOrPredVal=0, restoreWeights=True, onlysavegraph=False):
	assert(findAccOrArgMaxOrPredVal>=0 and findAccOrArgMaxOrPredVal<=2)
	if restoreWeights: assert(not(onlysavegraph))
	if onlysavegraph: assert(findAccOrArgMaxOrPredVal==1)

	x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
	if (not(onlysavegraph)):
		y =  tf.placeholder(tf.int32, shape=(None, 10), name='output_y')
	logits = sqn.inference(x)

	if findAccOrArgMaxOrPredVal==0:
		correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
	elif findAccOrArgMaxOrPredVal==1:
		logits = tf.argmax(logits, axis=1)
	elif findAccOrArgMaxOrPredVal==2:
		pass
	else:
		assert False

	print("Doing inference on ", len(images), " images.")
	feed_dict = {x: images}
	if not(onlysavegraph):
		feed_dict[y] = labels

	sess.run(tf.global_variables_initializer())
	if onlysavegraph:
		output_tensor = None
		gg = tf.get_default_graph()
		for node in gg.as_graph_def().node:
			if node.name == 'ArgMax':
				output_tensor = gg.get_operation_by_name(node.name).outputs[0]
		optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)
		return

	if restoreWeights:
		saver = tf.train.Saver(sqn.all_weights)
		saver.restore(sess, restoreModelPath)

	print("*************** Starting Prediction****************")
	start_time = time.time()
	if findAccOrArgMaxOrPredVal==0:
		predictions = sess.run([accuracy], feed_dict=feed_dict)
	else:
		predictions = sess.run([logits], feed_dict=feed_dict)
	end_time = time.time()
	print("*************** Done Prediction****************")
	duration = end_time - start_time
	print("Time taken in prediction : ", duration)

	print("Inference result = ", predictions)
	return predictions

def getTrainedWeightsStrForm(sess, evalTensors, scalingFac):
	allWeightsStr = ''
	finalParameters = map(lambda x : sess.run(x), evalTensors)
	for curParameterVal in finalParameters:
		for xx in numpy.nditer(curParameterVal, order='C'):
			allWeightsStr += (str(int(xx * (1<<scalingFac))) + ' ')
		allWeightsStr += '\n\n'
	return allWeightsStr

def findAndSaveCorrectTestImg(pred, features, actualLabels, correctImgFolder, incorrectImgFolder, textFolder, sess, sqn, scalingFac):
	#Run with findAcc=False and outputArgMax=False
	assert(len(pred)==1 and len(pred[0].shape)==2)
	modelPred = numpy.argmax(pred[0], axis=1)
	trueLabel = numpy.argmax(actualLabels, axis=1)
	print("Pred = ", pred)
	print("actualLabels = ", actualLabels)
	print("ModelPred = ", modelPred)
	print("TrueLabel = ", trueLabel)
	numImages = len(features)
	allWeightsStr = getTrainedWeightsStrForm(sess, sqn.all_weights, scalingFac)
	for ii in range(numImages):
		curImage = features[ii]
		if (modelPred[ii]==trueLabel[ii]):
			matplotlib.image.imsave(os.path.join(correctImgFolder, str(ii)+'-test.png'), curImage)
		else:
			matplotlib.image.imsave(os.path.join(incorrectImgFolder, str(ii)+'-test.png'), curImage)
		textInpFileName = os.path.join(textFolder, str(ii)+'-test-inp.txt')
		dumpCifar10Image(curImage, textInpFileName, scalingFac, 'w')
		with open(textInpFileName, 'a') as ff:
			ff.write(allWeightsStr)
		if (ii%10==0):
			print("Processing ", ii, " images done.")

def main():
	scalingFac = 12
	findAccOrArgMaxOrPredVal = 0
	restoreWeights = True
	onlysavegraph = False
	save_model_path = './TrainedModel/model'
	doTraining = False

	inp = None
	if (len(sys.argv) > 1):
		inp = sys.argv[1]
		if (inp == 'train'):
			doTraining = True
		elif (inp == 'savegraph'):
			findAccOrArgMaxOrPredVal = 1
			restoreWeights = False
			onlysavegraph = True
			testing_features, testing_labels = Util.get_sample_points(2, 4555, 4556)
		elif (inp == 'testSingleTestInp'):
			testBatchInpNum = int(sys.argv[2])
			findAccOrArgMaxOrPredVal = 1
			restoreWeights = True
			onlysavegraph = False
			all_testing_features, all_testing_labels = Util.load_preprocess_testing_data()
			testing_features, testing_labels = all_testing_features[testBatchInpNum:testBatchInpNum+1], all_testing_labels[testBatchInpNum:testBatchInpNum+1]
		elif (inp == 'testSingleTestInpAndSaveData'):
			testBatchInpNum = int(sys.argv[2])
			findAccOrArgMaxOrPredVal = int(sys.argv[3])
			restoreWeights = True
			onlysavegraph = False
			all_testing_features, all_testing_labels = Util.load_preprocess_testing_data()
			testing_features, testing_labels = all_testing_features[testBatchInpNum:testBatchInpNum+1], all_testing_labels[testBatchInpNum:testBatchInpNum+1]
			# testing_features, testing_labels = numpy.full((1,32,32,3),0.01), numpy.full((1,10),0.01)
		elif (inp == 'savegraphAndDataBatch'):
			batchNum = int(sys.argv[2])
			imgStartNum = int(sys.argv[3])
			imgEndNum = int(sys.argv[4])
			findAccOrArgMaxOrPredVal = 1
			restoreWeights = False
			onlysavegraph = True
			testing_features, testing_labels = Util.get_sample_points(batchNum, imgStartNum, imgEndNum)
		elif (inp == 'testBatchInp'):
			imgStartNum = int(sys.argv[2])
			imgEndNum = int(sys.argv[3])
			findAccOrArgMaxOrPredVal = 1
			restoreWeights = True
			onlysavegraph = False
			all_testing_features, all_testing_labels = Util.load_preprocess_testing_data()
			testing_features, testing_labels = all_testing_features[imgStartNum:imgEndNum], all_testing_labels[imgStartNum:imgEndNum]
		elif (inp == 'findAndSaveCorrectTestImg'):
			findAccOrArgMaxOrPredVal = 2
			restoreWeights = True
			onlysavegraph = False
			testing_features, testing_labels = Util.load_preprocess_testing_data()
			testing_features, testing_labels = testing_features[0:100], testing_labels[0:100]
		else:
			if (inp != ""):
				print("WARNING : Given option didn't match any known value.")
			testing_features, testing_labels = Util.load_preprocess_testing_data()

	sqn = SqueezeNet1(use_cons_init=onlysavegraph)
	if doTraining:
		train(sqn, save_model_path)
	else:
		with tf.Session() as sess:
			pred = infer(sqn, sess, testing_features, testing_labels, save_model_path, findAccOrArgMaxOrPredVal=findAccOrArgMaxOrPredVal, 
																						restoreWeights=restoreWeights, 
																						onlysavegraph=onlysavegraph)
			if findAccOrArgMaxOrPredVal==1 and not(onlysavegraph): 
				print("Actual labels = ", testing_labels)
				print("ArgMax in actual label : ", numpy.argmax(testing_labels, axis=1))

			if (inp == 'findAndSaveCorrectTestImg'):
				print('Running ' + inp)
				print(pred[0].shape)
				findAndSaveCorrectTestImg(pred, testing_features, testing_labels, './testPred/CorrectImg/', './testPred/IncorrectImg/', './testPred/TestInputs/', sess, sqn, scalingFac)

			if (inp == 'savegraphAndDataBatch' or inp=='testSingleTestInpAndSaveData'):
				imgFileName = 'SqNet_CIFAR_img.inp'
				weightsFileName = 'SqNet_CIFAR_weights.inp'
				for ii,curFeature in enumerate(testing_features):
					if ii == 0 :
						DumpTFMtData.dumpImageDataInt(curFeature, imgFileName, scalingFac, 'w')
					else:
						DumpTFMtData.dumpImageDataInt(curFeature, imgFileName, scalingFac, 'a')
				DumpTFMtData.dumpTrainedWeightsInt(sess, sqn.all_weights, weightsFileName, scalingFac, 'w')

if __name__ == '__main__':
	main()
	
