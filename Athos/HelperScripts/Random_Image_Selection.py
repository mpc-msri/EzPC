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

# Choose a random subset of SELECT_IMAGES from TOTAL_IMAGES
import random

TOTAL_IMAGES = 50000 #ImageNet validation dataset size is 50k images
SELECT_IMAGES = 1000 #Choose a random subset of 1k images
with open('./ImageNet_ValData/imagenet12_val_nlabels.txt', 'r') as ff:
	labels = ff.readlines()

idxSelected = random.sample(range(TOTAL_IMAGES), SELECT_IMAGES)
labelSelected = [labels[i] for i in idxSelected]

with open("./ImageNet_ValData/Random_Img_Idx.txt", 'w') as ff:
	for x in idxSelected:
		ff.write(str(x+1)+"\n") #make it 1-indexed as image numbers are 1 indexed

with open("./ImageNet_ValData/Random_Img_Labels.txt", "w") as ff:
	for x in labelSelected:
		ff.write(str(x))
