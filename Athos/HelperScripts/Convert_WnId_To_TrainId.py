with open('./ImageNet_ValData/imagenet_2012_validation_synset_labels.txt', 'r') as ff:
	allLines = ff.readlines()

labels = list(map(lambda x : x.rstrip(), allLines))
sortedLabels = sorted(set(labels))

labelsMap = {}
for k,v in enumerate(sortedLabels):
	labelsMap[v] = k+1

with open('./ImageNet_ValData/sortedWordnetIds.txt', 'w') as ff:
	for x in sortedLabels:
		ff.write(x + '\n')

with open('./ImageNet_ValData/imagenet12_val_nlabels.txt', 'w') as ff:
	for curLabel in labels:
		ff.write(str(labelsMap[curLabel]))
		ff.write('\n')
