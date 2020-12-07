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
Parts of this code were used from https://github.com/avoroshilov/tf-squeezenet.
**

'''

import os, sys
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
import time
import numpy

def imread_resize(path):
    #img_orig =imread(path)
    img_orig = Image.open(path).convert("RGB")
    img_orig = numpy.asarray(img_orig)
    img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img, img_orig.shape

mean_pixel = np.array([104.006, 116.669, 122.679], dtype=np.float32)

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

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

  for curImgNum in range(firstImgNum, lastImgNum):
    if (curImgNum % 100 == 0):
        print("CurImgNum = ", curImgNum)
    actualImgNum = curImgNum if not(randomIdxToBeChosen) else randomIdxToBeChosen[curImgNum-1]
    imgFileName = os.path.join(imgFolderName, fileNamePrefix + "{:08d}".format(actualImgNum) + '.JPEG')
    imgData, imgShape = imread_resize(imgFileName)
    preprocessed_image_buffer = preprocess(imgData, mean_pixel)

    saveFilePath = os.path.join(preProcessedImgFolderName, 'ImageNum_' + str(actualImgNum) + '.inp')
    dumpImageDataFloat(preprocessed_image_buffer, saveFilePath, 'w')

if __name__=='__main__':
  main()
