import numpy
import argparse
import os, sys, time
import tensorflow as tf
import _pickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import squeezenet_main as sqzmain

batchsize = 100
N = 200

data, sqz_mean = sqzmain.load_net('../PreTrainedModel/sqz_full.mat')
image = tf.placeholder(dtype=sqzmain.get_dtype_tf(), shape=(None,227,227,3), name="image_placeholder")
keep_prob = 0.0
sqznet = sqzmain.net_preloaded(data, image, 'max', True, keep_prob)

finalActivationsFileName = 'floating_point_acc.outp'
argmaxOutputFileName = 'floating_point_argmax.outp'

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  with open(finalActivationsFileName,'w') as ff:
      pass
  with open(argmaxOutputFileName,'w') as ff:
      pass
  numbatches = N//batchsize
  for batchNum in range(numbatches):
    startImgNum = (batchNum*batchsize) + 1
    endImgNum = N if (batchNum == numbatches-1) else (((batchNum+1)*batchsize))
    print("Processing images from start,end = {0}, {1}".format(startImgNum, endImgNum))
    images = numpy.zeros(shape=(endImgNum-startImgNum+1,227,227,3))
    for curImgNum in range(startImgNum, endImgNum+1):
      with open('./PreProcessedImages/ImageNum_'+str(curImgNum)+'.inp', 'r') as ff:
        line = ff.readline()
        images[curImgNum-startImgNum] = numpy.reshape(list(map(lambda x : float(x), line.split())), (227,227,3))
    feed_dict = {image: images}
    predictions = sess.run(sqznet['classifier_pool'], feed_dict=feed_dict)
    with open(finalActivationsFileName, 'a') as ff:
        with open(argmaxOutputFileName, 'a') as gg:
            for i in range(endImgNum-startImgNum+1):
                ff.write('Answer for imgCounter = ' + str(startImgNum+i) + '\n')
                for elem in numpy.nditer(predictions[i],order='C'):
                    ff.write(str(elem)+' ')
                ff.write('\n\n')
                gg.write('Answer for imgCounter = '+str(startImgNum+i)+' is ')
                gg.write(str(numpy.argmax(predictions[i], 2))+'\n')

