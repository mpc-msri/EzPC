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

import os, sys

numImages = 10
newScaledFileNameSuffix = '_scaled_'
scalingFactors = [12]

if (len(sys.argv)!=3):
	print("Incorrect args: Run as python3 Scale_img_and_model.py <model file name> <float img directory>", file=sys.stderr)
	exit(1)

modelFileName = sys.argv[1]
floatImgDir = sys.argv[2]
randomImgIdxFileName = "./ImageNet_ValData/Random_Img_Idx.txt"

def checkIfFileExists(filename):
	if not(os.path.exists(filename)):
		print("Expected file doesn't exist. Error. FileName = {0}.".format(filename), file=sys.stderr)
		exit(1)

checkIfFileExists(randomImgIdxFileName)

with open(randomImgIdxFileName, 'r') as ff:
	imgIdx = ff.readlines()
imgIdx = list(map(lambda x: int(x.rstrip()) , imgIdx))

def scaleImg(filename, scalingFac):
	checkIfFileExists(filename)
	with open(filename, 'r') as ff:
		line = ff.readline()
	val = line.split()
	val = list(map(lambda x : int(float(x)*(1<<scalingFac)), val))
	path = (os.path.splitext(filename)[0])
	with open(path + newScaledFileNameSuffix + str(scalingFac) + '.inp', 'w') as ff:
		for elem in val:
			ff.write(str(elem) + ' ')

def scaleModel(modelFileName, scalingFac):
	checkIfFileExists(modelFileName)
	path = os.path.splitext(modelFileName)[0]
	with open(path + newScaledFileNameSuffix + str(scalingFac) + '.inp', 'w') as newff:
		with open(modelFileName, 'r') as ff:
			line = ff.readline()
			while line:
				if (line != ''):
					val = line.split()
					val = list(map(lambda x : int(float(x)*(1<<scalingFac)), val))
					for elem in val:
						newff.write(str(elem) + ' ')
					newff.write('\n')
				line = ff.readline()

for curScale in scalingFactors:
	print("Starting for scale factor = {0}.".format(curScale))
	for i in range(numImages):
		if (i % 100 == 0):
			print("Processing img number = {0} for scale factor = {1}.".format(i, curScale))
		floatFileName = os.path.join(floatImgDir, 'ImageNum_'+str(imgIdx[i])+'.inp')
		scaleImg(floatFileName, curScale)
	print("All images processed. Starting processing of model.")
	scaleModel(modelFileName, curScale)

print("All done.")
