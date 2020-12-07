# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to preprocess images.

Training images are sampled using the provided bounding boxes, and subsequently
cropped to the sampled bounding box. Images are additionally flipped randomly,
then resized to the target output size (without aspect-ratio preservation).

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped.

All images undergo mean color subtraction.

Note that these steps are colloquially referred to as "ResNet preprocessing,"
and they differ from "VGG preprocessing," which does not use bounding boxes
and instead does an aspect-preserving resize followed by random crop during
training. (These both differ from "Inception preprocessing," which introduces
color distortion steps.)

** Code modified for CrypTFlow **

"""

import os, sys
import numpy
import random
import tensorflow as tf
import xml.etree.ElementTree as ET
import imagenet_preprocessing
import _pickle as pickle

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3

#################################################
# Creating tf record
class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width

def _convert_to_example(filename, image_buffer, label, synset, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace.encode('utf-8')),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset.encode('utf-8')),
      'image/format': _bytes_feature(image_format.encode('utf-8')),
      'image/filename': _bytes_feature(os.path.basename(filename).encode('utf-8')),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example
#################################################


#################################################
# Parsing tf record
def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
      'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                 default_value=-1),
      'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.io.parse_single_example(serialized=example_serialized,
                                        features=feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

  return features['image/encoded'], label, bbox

def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = _parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=DEFAULT_IMAGE_SIZE,
      output_width=DEFAULT_IMAGE_SIZE,
      num_channels=NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label
#################################################


#################################################
# Parsing image to get bounding box info
class BoundingBox(object):
  pass

def GetItem(name, root, index=0):
  count = 0
  for item in root.iter(name):
    if count == index:
      return item.text
    count += 1
  # Failed to find "index" occurrence of item.
  return -1

def GetInt(name, root, index=0):
  return int(GetItem(name, root, index))

def FindNumberBoundingBoxes(root):
  index = 0
  while True:
    if GetInt('xmin', root, index) == -1:
      break
    index += 1
  return index

def ProcessXMLAnnotation(xml_file):
  """Process a single XML file containing a bounding box."""
  # pylint: disable=broad-except
  try:
    tree = ET.parse(xml_file)
  except Exception:
    print('Failed to parse: ' + xml_file, file=sys.stderr)
    return None
  # pylint: enable=broad-except
  root = tree.getroot()

  num_boxes = FindNumberBoundingBoxes(root)
  boxes = []

  for index in range(num_boxes):
    box = BoundingBox()
    # Grab the 'index' annotation.
    box.xmin = GetInt('xmin', root, index)
    box.ymin = GetInt('ymin', root, index)
    box.xmax = GetInt('xmax', root, index)
    box.ymax = GetInt('ymax', root, index)

    box.width = GetInt('width', root)
    box.height = GetInt('height', root)
    box.filename = GetItem('filename', root) + '.JPEG'
    box.label = GetItem('name', root)

    xmin = float(box.xmin) / float(box.width)
    xmax = float(box.xmax) / float(box.width)
    ymin = float(box.ymin) / float(box.height)
    ymax = float(box.ymax) / float(box.height)

    # Some images contain bounding box annotations that
    # extend outside of the supplied image. See, e.g.
    # n03127925/n03127925_147.xml
    # Additionally, for some bounding boxes, the min > max
    # or the box is entirely outside of the image.
    min_x = min(xmin, xmax)
    max_x = max(xmin, xmax)
    box.xmin_scaled = min(max(min_x, 0.0), 1.0)
    box.xmax_scaled = min(max(max_x, 0.0), 1.0)

    min_y = min(ymin, ymax)
    max_y = max(ymin, ymax)
    box.ymin_scaled = min(max(min_y, 0.0), 1.0)
    box.ymax_scaled = min(max(max_y, 0.0), 1.0)

    boxes.append(box)

  return boxes

#################################################


# Old function
def CreateTFRecordFromImage(input_filename, output_filename):
  coder = ImageCoder()
  
  writer = tf.python_io.TFRecordWriter(output_filename)
  image_buffer, height, width = _process_image(input_filename, coder)
  label = 0
  synset = 'n02323233'
  example = _convert_to_example(input_filename, image_buffer, label,
                                  synset, height, width)
  writer.write(example.SerializeToString())
  writer.close()

# Old function
def ReadAndPreProcessTFRecord(tfRecordFileName):
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([tfRecordFileName])
  _, serialized_example = reader.read(filename_queue)

  parsed_image_tensor = parse_record(serialized_example, False, tf.float32)
  return parsed_image_tensor

def get_sample(input_img_filename, input_xml_filename, coder = None):
  if coder is None:
    coder = ImageCoder()
  image_buffer, height, width = _process_image(input_img_filename, coder)

  boxes = ProcessXMLAnnotation(input_xml_filename)
  # print(boxes[0].xmin, boxes[0].ymin, boxes[0].xmax, boxes[0].ymax)
  # print(boxes[0].xmin_scaled, boxes[0].ymin_scaled, boxes[0].xmax_scaled, boxes[0].ymax_scaled)

  xmin = tf.expand_dims(boxes[0].xmin_scaled, 0)
  ymin = tf.expand_dims(boxes[0].ymin_scaled, 0)
  xmax = tf.expand_dims(boxes[0].xmax_scaled, 0)
  ymax = tf.expand_dims(boxes[0].ymax_scaled, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.expand_dims(bbox, 0)
  # bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=DEFAULT_IMAGE_SIZE,
      output_width=DEFAULT_IMAGE_SIZE,
      num_channels=NUM_CHANNELS,
      is_training=False)

  return image

def dumpImageDataFloat(imgData, filename, writeMode):
  with open(filename, writeMode) as ff:
    for xx in numpy.nditer(imgData, order='C'):
      ff.write(str(xx) + ' ')
    ff.write('\n\n')

def main():
  if not((len(sys.argv) >= 7) and (len(sys.argv) <= 8)):
    print("Args : <imgFolderName> <bboxFolderName> <fileNamePrefix> <preProcessedImgFolderName> <firstImgNum> <lastImgNum> <randomSubsetIdxFile>?", file=sys.stderr)
    exit(1)

  imgFolderName = sys.argv[1]
  bboxFolderName = sys.argv[2]
  fileNamePrefix = sys.argv[3]
  preProcessedImgFolderName = sys.argv[4]
  firstImgNum = int(sys.argv[5])
  lastImgNum = int(sys.argv[6])
  randomSubsetIdxFile = None
  if (len(sys.argv) == 8):
    randomSubsetIdxFile = sys.argv[7]
  
  randomIdxToBeChosen = None
  if randomSubsetIdxFile:
    with open(randomSubsetIdxFile, 'r') as ff:
      randomIdxToBeChosen = ff.readlines()
      randomIdxToBeChosen = list(map(lambda x : int(x.rstrip()), randomIdxToBeChosen))
    assert(lastImgNum <= len(randomIdxToBeChosen)+1) #Assert that the last img num passed is within bounds

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  coder = ImageCoder()
  for curImgNum in range(firstImgNum, lastImgNum):
    if (curImgNum % 100 == 0):
        print("CurImgNum = ", curImgNum)
    actualImgNum = curImgNum if not(randomIdxToBeChosen) else randomIdxToBeChosen[curImgNum-1]
    imgFileName = os.path.join(imgFolderName, fileNamePrefix + "{:08d}".format(actualImgNum) + '.JPEG')
    bboxFileName = os.path.join(bboxFolderName, fileNamePrefix + "{:08d}".format(actualImgNum) + '.xml')
  
    image = get_sample(imgFileName, bboxFileName, coder)
    outp = sess.run([image], feed_dict={})

    saveFilePath = os.path.join(preProcessedImgFolderName, 'ImageNum_' + str(actualImgNum) + '.inp')
    dumpImageDataFloat(outp, saveFilePath, 'w')

if __name__=='__main__':
  main()
