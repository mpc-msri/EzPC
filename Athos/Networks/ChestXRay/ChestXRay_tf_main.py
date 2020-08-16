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

import cv2, numpy, sys, os, argparse, time
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

def get_preprocessed_image(filename):
  resized_height = 224
  resized_width = 224
  test_image_path = filename
  cv2_image = cv2.resize(cv2.imread(test_image_path), (resized_height, resized_width))#.astype(numpy.float32)
  cv2_image = cv2_image - numpy.min(cv2_image)
  cv2_image = cv2_image/numpy.ptp(cv2_image)
  cv2_image = 255*cv2_image
  cv2_image = cv2_image.astype('uint8')
  return cv2_image

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

imagesTemp = get_preprocessed_image('./SampleImages/00014251_029.png')
images = numpy.zeros(shape=(1,224,224,3))
images[0] = imagesTemp
feed_dict={'input_1:0' : images}

with tf.Session() as sess:
  saver = tf.train.import_meta_graph("./PreTrainedModel/TFModel/model.meta")
  sess.run(tf.global_variables_initializer())

  # Find output tensor
  output_tensor = None
  gg = tf.get_default_graph()
  for node in gg.as_graph_def().node:
    if node.name == 'dense_1/Sigmoid':
      output_tensor = gg.get_operation_by_name(node.name).outputs[0]

  assert(output_tensor is not None)
  optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)

  if args.savePreTrainedWeightsInt or args.savePreTrainedWeightsFloat or args.runPrediction or args.saveImgAndWtData:
    saver.restore(sess, "./PreTrainedModel/TFModel/model")
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
  if args.savePreTrainedWeightsInt:
    DumpTFMtData.dumpTrainedWeights(sess, trainVars, 'ChestXRay_weights_{0}.inp'.format(args.scalingFac), args.scalingFac, 'w')
  if args.savePreTrainedWeightsFloat:
    DumpTFMtData.dumpTrainedWeightsFloat(sess, trainVars, 'ChestXRay_weights_float.inp', 'w')
  if args.saveImgAndWtData:
    DumpTFMtData.dumpImgAndWeightsDataSeparate(sess, images[0], trainVars, 'ChestXRay_img_{0}.inp'.format(args.scalingFac), 'ChestXRay_weights_{0}.inp'.format(args.scalingFac), args.scalingFac)
    
