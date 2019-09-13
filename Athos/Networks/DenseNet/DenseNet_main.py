import numpy
import argparse
import os, sys, time
import tensorflow as tf
import _pickle as pickle

import nets_factory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

model_name = 'densenet121'
num_classes = 1000
network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes,
        is_training=False)

imagesPlaceHolder = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_x')
logits, end_points = network_fn(imagesPlaceHolder)
pred = tf.argmax(logits, 3)

# If want to run another file, uncomment the following and pass the preprocessed file which contains float values.
# with open('ImageNum_42677.inp', 'r') as ff:
#   line = ff.readline()
#   line = line.split()
#   imagesTemp = list(map(lambda x : float(x), line))
#   imagesTemp = numpy.reshape(imagesTemp, (224,224,3))

sampleImageFilePath = './SampleImages/n02109961_36_denseNet_preprocessed.pkl'
with open(sampleImageFilePath, 'rb') as ff:
  imagesTemp = pickle.load(ff)
images = numpy.zeros(shape=(1,224,224,3))
images[0] = imagesTemp
feed_dict = {imagesPlaceHolder : images}

def parseArgs():
  parser = argparse.ArgumentParser()

  parser.add_argument("--savePreTrainedWeightsInt", type=bool, default=False, help="savePreTrainedWeightsInt")
  parser.add_argument("--savePreTrainedWeightsFloat", type=bool, default=False, help="savePreTrainedWeightsFloat")
  parser.add_argument("--scalingFac", type=int, default=15, help="scalingFac")
  parser.add_argument("--runPrediction", type=bool, default=False, help="runPrediction")
  parser.add_argument("--saveImgAndWtData", type=bool, default=False, help="saveImgAndWtData")

  args = parser.parse_args()
  return args

args = parseArgs()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  output_tensor = None
  gg = tf.get_default_graph()
  for node in gg.as_graph_def().node:
    # if node.name == 'densenet121/logits/BiasAdd':
    if node.name == 'ArgMax':
      output_tensor = gg.get_operation_by_name(node.name).outputs[0]

  assert(output_tensor is not None)
  optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)

  if args.savePreTrainedWeightsInt or args.savePreTrainedWeightsFloat or args.runPrediction or args.saveImgAndWtData:
    modelPath = './PreTrainedModel/tf-densenet121.ckpt'
    saver = tf.train.Saver()
    saver.restore(sess, modelPath)
    if args.savePreTrainedWeightsInt or args.savePreTrainedWeightsFloat or args.saveImgAndWtData:
      DumpTFMtData.updateWeightsForBN(optimized_graph_def, sess, feed_dict)

  predictions = None
  if args.runPrediction:
    print("*************** Starting Prediction****************")
    start_time = time.time()
    predictions = sess.run(output_tensor, feed_dict=feed_dict)
    end_time = time.time()
    print("*************** Done Prediction****************")

  print(predictions)

  trainVarsName = []
  for node in optimized_graph_def.node:
    if node.op=="VariableV2":
      trainVarsName.append(node.name)
  trainVars = list(map(lambda x : tf.get_default_graph().get_operation_by_name(x).outputs[0] , trainVarsName))
  if args.saveImgAndWtData:
    DumpTFMtData.dumpImgAndWeightsData(sess, images[0], trainVars, 'DenseNet_img_input.inp', args.scalingFac)
  if args.savePreTrainedWeightsInt:
    DumpTFMtData.dumpTrainedWeightsInt(sess, trainVars, 'DenseNet_img_input_weights_int.inp', args.scalingFac, 'w')
  if args.savePreTrainedWeightsFloat:
    DumpTFMtData.dumpTrainedWeightsFloat(sess, trainVars, 'DenseNet_img_input_weights_float.inp', 'w')

