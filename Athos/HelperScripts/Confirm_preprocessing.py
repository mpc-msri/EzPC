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

import os,sys, functools

if (len(sys.argv)!=4):
	print("Incorrect args. Error.", file=sys.stderr)
	exit(1)
preProcessedImagesDir = sys.argv[1]
startImgNum = int(sys.argv[2])
endImgNum = int(sys.argv[3])
numImg = endImgNum-startImgNum+1
expectedShape = [224,224,3]
expectedNumElements = functools.reduce(lambda x,y : x*y, expectedShape)
print("ExpectedNumElements in all preprocessed images = ", expectedNumElements)

badImages = []
for i in range(startImgNum,endImgNum):
	if ((i % (numImg/10))==0):
		print("Reached i = {0}.".format(i))
	filename = os.path.join(preProcessedImagesDir, 'ImageNum_'+str(i)+'.inp')
	with open(filename, 'r') as ff:
		line = ff.readline()
	val = list(map(lambda x : float(x) , line.split()))
	if (len(val)!=expectedNumElements):
		print("Expected num of elements not found in imagenum = {0}.".format(i))
		badImages.append(i)

print("Found {0} bad images.".format(len(badImages)))
