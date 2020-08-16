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

Part of the code in this file is taken from https://github.com/pudae/tensorflow-densenet.

'''

import os, sys, numpy
import _pickle as pickle
import tensorflow as tf
import DenseNet_preprocessing

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

  return image, height, width

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

  def helper_img(curImgNum):
    actualImgNum = curImgNum if not(randomIdxToBeChosen) else randomIdxToBeChosen[curImgNum-1]
    saveFilePath = os.path.join(preProcessedImgFolderName, 'ImageNum_' + str(actualImgNum) + '.inp')
    if (os.path.exists(saveFilePath)):
      print("Preprocessed file already exists. Skipping. Img Num = {0}".format(actualImgNum))
      return 
    imgFileName = os.path.join(imgFolderName, fileNamePrefix + "{:08d}".format(actualImgNum) + '.JPEG')
    image_buffer, height, width = _process_image(imgFileName, coder)
    preprocessed_image_buffer = sess.run(DenseNet_preprocessing.preprocess_image(image_buffer, 224, 224))

    dumpImageDataFloat(preprocessed_image_buffer, saveFilePath, 'w')

  pid = os.getpid()
  print("PID={0}, firstImgNum={1}, lastImgNum={2}: Starting with first image.".format(pid, firstImgNum, lastImgNum))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coder = ImageCoder()
    for curImgNum in range(firstImgNum, (firstImgNum + lastImgNum)//2):
      helper_img(curImgNum)

  print("PID={0}, firstImgNum={1}, lastImgNum={2}: Crossed half mark.".format(pid, firstImgNum, lastImgNum))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coder = ImageCoder()
    for curImgNum in range((firstImgNum + lastImgNum)//2, lastImgNum):
      helper_img(curImgNum)

  print("PID={0}, firstImgNum={1}, lastImgNum={2}: All images done.".format(pid, firstImgNum, lastImgNum))

if __name__=='__main__':
  main()
