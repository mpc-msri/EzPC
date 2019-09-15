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
		print("Reched i = {0}.".format(i))
	filename = os.path.join(preProcessedImagesDir, 'ImageNum_'+str(i)+'.inp')
	with open(filename, 'r') as ff:
		line = ff.readline()
	val = list(map(lambda x : float(x) , line.split()))
	if (len(val)!=expectedNumElements):
		print("Expected num of elements not found in imagenum = {0}.".format(i))
		badImages.append(i)

print("Found {0} bad images.".format(len(badImages)))
