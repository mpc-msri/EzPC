import numpy
import argparse
import os, sys, time
import tensorflow as tf
import _pickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ResNet_main

batchsize = 1000
N = 50000

x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_x')
imgnet_model = ResNet_main.ImagenetModel(50, 'channels_last')
pred = imgnet_model(x, False)

finalActivationsFileName = 'floating_point_acc.outp'
argmaxOutputFileName = 'floating_point_argmax.outp'

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  modelPath = '../PreTrainedModel/resnet_v2_fp32_savedmodel_NHWC/1538687283/variables/variables'
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
    feed_dict = {x: images}
    predictions = sess.run(pred, feed_dict=feed_dict)
    with open(finalActivationsFileName, 'a') as ff:
        with open(argmaxOutputFileName, 'a') as gg:
            for i in range(endImgNum-startImgNum+1):
                ff.write('Answer for imgCounter = ' + str(startImgNum+i) + '\n')
                for elem in numpy.nditer(predictions[i],order='C'):
                    ff.write(str(elem)+' ')
                ff.write('\n\n')
                gg.write('Answer for imgCounter = '+str(startImgNum+i)+' is ')
                gg.write(str(numpy.argmax(predictions[i]))+'\n')

