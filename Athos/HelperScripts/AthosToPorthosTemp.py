'''

Authors: Nishant Kumar.

Copyright:
Copyright (c) 2018 Microsoft Research
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

import sys,os

if (len(sys.argv)!=2):
	exit(1)

filename = sys.argv[1]
with open(filename, 'r') as ff:
	lines = ff.readlines()

newfunc = ['void Conv2DCSF(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, auto& inputArr, auto& filterArr, int32_t consSF, auto& outArr){',
'#ifdef CONV_OPTI',
'	if ((FH>=5) || (FW>=5)){',
'		funcConv2DCSF(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);',
'	}',
'	else{',
'		Conv2DCSFMain(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);',
'	}',
'#else',
'	Conv2DCSFMain(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);',
'#endif',
'}',
'\n'
]

ii=0
while(ii < len(lines)):
	curLine = lines[ii]
	if curLine.startswith("void Conv2DReshapeInput("):
		jj = ii+1
		found = 0
		while(found != 2 and jj<ii+100):
			if 'funcSSCons' in lines[jj]:
				found+=1
				lines[jj] = lines[jj].replace('funcSSCons','')
			jj+=1
	elif curLine.startswith('void Pad442('):
		jj = ii+1
		found = 0
		while(found != 1 and jj<ii+100):
			if 'funcSSCons' in lines[jj]:
				found+=1
				lines[jj] = lines[jj].replace('funcSSCons','')
			jj+=1
	elif curLine.startswith('void Conv2DCSF('):
		lines[ii] = lines[ii].replace('Conv2DCSF', 'Conv2DCSFMain')
		jj = ii+1
		found = 0
		while(found != 1 and jj<ii+100):
			if (lines[jj].startswith('}')):
				found+=1
			jj+=1
		for curFuncLine in reversed(newfunc):
			lines.insert(jj+1,curFuncLine+'\n')
		ii = jj+len(newfunc)
	ii+=1

with open(filename, 'w') as ff:
	for line in lines:
		ff.write(line)
