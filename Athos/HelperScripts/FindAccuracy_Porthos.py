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

# Use this script to find accuracy once the full exploration has been done on relevant scale factors.
# NOTE: The ground truth labels are in [1, 1000].
#		Resnet outputs labels in [0,1000] -- class 0 is extraneous and is the other category.
#		For DenseNet and SqueezeNet, the labels after argmax are in [0,999]:
#			so either we have to do ground truth labels-1 or while outputing from the code for SqNet/DenseNet,
#			add a +1. Choosing to go with the former.
#		So, in summary, when running resnet, use the last parameter as 1, while for SqueezeNet/DenseNet use it as 0.

import os, sys
import numpy as np

if (len(sys.argv) < 4):
	print("Usage : python3 FindAccuracy_Porthos.py <groundTruthLabelsFileName> <inferenceOutputDirectory> <lowerBoundOfOutputLabels>")
	exit(1)

# Change following parameters accordingly
ScalesToCheck = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# ScalesToCheck = [11]
numImages = 96
topK = 5

groundTruthLabelsFileName = sys.argv[1]
inferenceOutputDirectory = sys.argv[2]
lowerBoundOfOutputLabels = int(sys.argv[3])
if (lowerBoundOfOutputLabels != 0 and lowerBoundOfOutputLabels != 1):
	print("lowerBoundOfOutputLabels should be either 0 or 1. Exiting.", file=sys.stderr)
	exit(1)

with open(groundTruthLabelsFileName, 'r') as ff:
	groundTruthLabels = ff.readlines()

groundTruthLabels = list(map(lambda x : int(x.rstrip()), groundTruthLabels)) #For imagenet, this is in [1,1000]
if (lowerBoundOfOutputLabels==0):
	groundTruthLabels = list(map(lambda x : x-1, groundTruthLabels)) #If the labels in the output start from 0, 
																	 # subtract 1 from the ground truth labels.

def parseInferenceOutputFile(predictions, i, outputFileName):
	with open(outputFileName, 'r') as ff:
		lines = ff.readlines()
	lines = list(map(lambda x : x.rstrip(), lines))
	lines = list(filter(lambda x : x!='', lines))
	if(len(lines)!=2):
		print("Error in parsing : "+outputFileName)
		assert(False)
	imgCounter = None
	for line in lines:
		if (line.startswith('Answer for')):
			imgCounter = int(line.split('=')[1].split(':')[0]) #This is assumed to be 0-indexed
		else:
			assert(imgCounter is not None)
			preds = line.split()
			# print(imgCounter,preds)
			preds = np.array(list(map(lambda x : int(x), preds)))
			topKPredsIdx = np.argpartition(preds, -1*topK)[-1*topK:]
			topKPredsIdx = topKPredsIdx[np.argsort(preds[topKPredsIdx])]
			for i, val in enumerate(topKPredsIdx):
				predictions[imgCounter][i] = val

def calculateAccuracy(predictions):
	global groundTruthLabels
	top1CorrectPred = 0
	topKCorrectPred = 0
	for i in range(numImages):
		if not(predictions[i][0]):
			continue
		if (groundTruthLabels[i] == predictions[i][-1]):
			top1CorrectPred += 1
		if (groundTruthLabels[i] in predictions[i]):
			topKCorrectPred += 1
	return (top1CorrectPred/(1.0*numImages), topKCorrectPred/(1.0*numImages))


for curScale in ScalesToCheck:
	predictions = [[None]*topK for _ in range(numImages)]
	for i in range(numImages):
		curFileName = os.path.join(inferenceOutputDirectory, 'stderr_' + str(curScale) + '_' + str(i) +'_proc_1.outp')
		parseInferenceOutputFile(predictions, i, curFileName)
	for i in range(numImages):
		for j in range(topK):
			assert(predictions[i][j] is not None)
	(top1Acc, topKAcc) = calculateAccuracy(predictions)
	print("curScale = " + str(curScale) + ", top1Acc = " + str(top1Acc) + ", topKAcc = " + str(topKAcc))


