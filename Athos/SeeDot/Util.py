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
	disableRMO = None
	disableLivenessOpti = None
	disableAllOpti = None

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
