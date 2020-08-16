'''

Authors: Sridhar Gopinath, Nishant Kumar.

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
import os
import _pickle as pickle

# Target word length.

class Version:
	Fixed = "fixed"
	Float = "float"
	All = [Fixed, Float]

class Target:
	EzPC = "ezpc"
	All = [EzPC]

class SFType:
	Constant = "constant"
	Variable = "variable"
	All = [Constant, Variable]

class Config:
	version = None
	target = None
	sfType = None
	astFile = None
	consSF = None
	outputFileName = None
	printASTBool = None
	wordLength = None
	actualWordLength = None
	disableRMO = None
	disableLivenessOpti = None
	disableAllOpti = None
	debugOnnx = None

###### Helper functions ######
def loadASTFromFile():
	return Config.astFile is not None

def forEzPC():
	return Config.target == Target.EzPC

def copy_dict(dict_src:dict, diff={}):
	dict_res = dict(dict_src)
	dict_res.update(diff)
	return dict_res

# z = [y1,y2,..] = [[x1,..], [x2,..], ..] --> [x1,.., x2,.., ..]
def flatten(z:list): 
	return [x for y in z for x in y]

def write_debug_info(name_mapping):
	if not os.path.exists('debug'):
		os.makedirs('debug')	

	with open('debug/seedot_ezpc_name_map.pkl', 'wb') as f:
		pickle.dump(name_mapping, f)

	with open('debug/seedot_ezpc_name_map.txt', 'w') as f:
		for val in name_mapping:
			f.write(val + '   ' + name_mapping[val] + '\n')		
