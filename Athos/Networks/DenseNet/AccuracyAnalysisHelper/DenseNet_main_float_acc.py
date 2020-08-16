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
import argparse
import os, sys, time
import tensorflow as tf
import _pickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import nets_factory

batchsize = 1000
N = 50000

model_name = 'densenet121'
num_classes = 1000
network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes,
        is_training=False)

finalActivationsFileName = 'floating_point_acc.outp'
argmaxOutputFileName = 'floating_point_argmax.outp'

imagesPlaceHolder = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_x')
logits, end_points = network_fn(imagesPlaceHolder)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  modelPath = '../PreTrainedModel/tf-densenet121.ckpt'
  saver = tf.train.Saver()
  saver.restore(sess, modelPath)

  with open(finalActivationsFileName,'w') as ff:
      pass
  with open(argmaxOutputFileName,'w') as ff:
      pass
  numbatches = N//batchsize
  for batchNum in range(numbatches):
    startImgNum = (batchNum*batchsize) + 1
    endImgNum = N if (batchNum == numbatches-1) else (((batchNum+1)*batchsize))
    print("Processing images from start,end = {0}, {1}".format(startImgNum, endImgNum))
    images = numpy.zeros(shape=(endImgNum-startImgNum+1,224,224,3))
    for curImgNum in range(startImgNum, endImgNum+1):
      with open('./PreProcessedImages/ImageNum_'+str(curImgNum)+'.inp', 'r') as ff:
        line = ff.readline()
        images[curImgNum-startImgNum] = numpy.reshape(list(map(lambda x : float(x), line.split())), (224,224,3))
    feed_dict = {imagesPlaceHolder: images}
    predictions = sess.run(logits, feed_dict=feed_dict)
    with open(finalActivationsFileName, 'a') as ff:
        with open(argmaxOutputFileName, 'a') as gg:
            for i in range(endImgNum-startImgNum+1):
                ff.write('Answer for imgCounter = ' + str(startImgNum+i) + '\n')
                for elem in numpy.nditer(predictions[i],order='C'):
                    ff.write(str(elem)+' ')
                ff.write('\n\n')
                gg.write('Answer for imgCounter = '+str(startImgNum+i)+' is ')
                gg.write(str(numpy.argmax(predictions[i], 2))+'\n')

