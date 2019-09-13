import sys

filename = sys.argv[1]
scalingFac = 12
maxibits = -1
bitsdict = {}
with open(filename, 'r') as ff:
	line = ff.readline()
	while line:
		if not(line.startswith("Matmul")):
			val = line.split()
			val = list(map(lambda x : int(x), val))
			for elem in val:
				curbits = len(bin(elem))-2
				if curbits in bitsdict:
					bitsdict[curbits] += 1
				else:
					bitsdict[curbits] = 0
				if (curbits > maxibits):
					maxibits = curbits
		line = ff.readline()

print(maxibits)
summ = 0
for k in sorted(bitsdict.keys()):
	print(k,bitsdict[k])
	summ+=bitsdict[k]

print("summ = ", summ)
prob = 0
for k in sorted(bitsdict.keys()):
	 curprob = bitsdict[k]/(1<<(64-k-1))
	 print("curprob ",k,curprob)
	 prob += curprob
print(prob)
