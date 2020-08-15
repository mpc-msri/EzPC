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
