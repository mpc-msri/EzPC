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
