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

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SeeDot')) #Add SeeDot directory to path

import Graph, AST.AST as AST, _pickle as pickle, os
from TFNodesAST import TFNodesAST
from AST.PrintAST import PrintAST
from AST.MtdAST import MtdAST

def checkTFNodeNameForEq(curNodeOp:str, givenOp:str):
	return (curNodeOp == "\"" + givenOp + "\"")

def generateASTForNode(graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict):
	curNodeOp = curNode.getOp()
	ast = None
	func = getattr(TFNodesAST, curNodeOp[1:-1]) #To remove the " at the begin and end
	(assignedVarAST, curAST) = func(graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict)
	return (assignedVarAST, curAST)

#Takes the graph DS and outputs IR in SeeDot for the same
def generateIRCode(graph, extraInfoDict):
	program = None
	innerMostLetASTNode = None
	dictNodeNameToOutVarStr = {}
	outVarCt = 0
	outVarPrefix = "J"
	mtdAST = MtdAST()
	for curNode in graph.getAllNodesRef():
		for curInp in curNode.getInputsRef():
			assert(curInp in dictNodeNameToOutVarStr) #Consequence of topological sorting of the TF graph
		(assignedVarAST, curAst) = generateASTForNode(graph, curNode, dictNodeNameToOutVarStr, extraInfoDict)

		mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName : curNode.getOp()[1:-1],
						AST.ASTNode.mtdKeyTFNodeName : curNode.getName()[1:-1]}

		if (curAst is None):
			dictNodeNameToOutVarStr[curNode.getName()] = None
			continue
		curOutVarStr = outVarPrefix + str(outVarCt)
		curOutVarAstNode = (assignedVarAST if assignedVarAST else AST.ID(curOutVarStr))
		if program:
			assert(type(innerMostLetASTNode) is AST.Let)
			newNode = AST.Let(curOutVarAstNode, curAst, curOutVarAstNode)
			mtdAST.visit(newNode, mtdForCurAST)
			innerMostLetASTNode.expr = newNode
			innerMostLetASTNode = newNode
		else:
			innerMostLetASTNode = AST.Let(AST.ID(curOutVarStr), curAst, curOutVarAstNode)
			mtdAST.visit(innerMostLetASTNode, mtdForCurAST)
			innerMostLetASTNode.depth = 0
			program = innerMostLetASTNode
		dictNodeNameToOutVarStr[curNode.getName()] = curOutVarStr
		outVarCt += 1
	return (program, dictNodeNameToOutVarStr)

def readSizeInfo(fileName):
	allLines = None
	with open(fileName) as f:
		allLines = f.readlines()
	sizeInfo = {}
	for line in allLines:
		tokens = line.split()
		nodeName = tokens[0]
		tokens = tokens[1:]
		nodeOPSize = []
		if (not tokens):
			nodeOPSize = [1]
		else:
			for curDimStr in tokens:
				if (curDimStr == ''): continue
				nodeOPSize.append(int(curDimStr))
		sizeInfo[nodeName] = nodeOPSize
	return sizeInfo

# Since later on in the pipeline, the placeholder nodes which come up as cin statements
# 	are to be excluded from the timing calculation, output all such PlaceHolder nodes together first.
#	This doesn't violate the topological ordering because all such PlaceHolder nodes are leaf nodes 
# 	in the graph.
def prefixAllPlaceHolderNodes(graph):
	allNodes = graph.getAllNodesRef()
	placeHolderNodes = []
	remNodes = []
	for curNode in allNodes:
		if (curNode.getOp() == "\"Placeholder\"" or curNode.getOp() == "\"VariableV2\""):
			# Assert this is indeed a leaf node
			assert(len(curNode.getInputsRef()) == 0)
			placeHolderNodes.append(curNode)
		else:
			remNodes.append(curNode)
	graph.setNodesList(placeHolderNodes + remNodes)

def main():
	sys.setrecursionlimit(10000)

	# First read the graph file
	if (len(sys.argv) < 2):
		print("TF python file unspecified.", file=sys.stderr)
		exit(1)

	filename = sys.argv[1]
	folderName = os.path.dirname(filename)
	graphFileName = os.path.join(folderName, 'graphDef.mtdata')
	graph = Graph.Graph()
	with open(graphFileName) as file:
		graph.readFromFilePointer(file)

	# Read the sizeInfo also
	sizeInfoFileName = os.path.join(folderName, 'sizeInfo.mtdata')
	sizeInfo = readSizeInfo(sizeInfoFileName)

	# Place all PlaceHolder nodes together at the beginning
	prefixAllPlaceHolderNodes(graph)

	# Re-format the input names of nodes
	for curNode in graph.getAllNodesRef():
		inputsRef = curNode.getInputsRef()
		for i,curInput in enumerate(inputsRef):
			if (curInput.startswith('"^')):
				# My hypothesis from empirical observation is that inputs which have '^' ahead of the node name
				#	denote control flow dependency and not data dependency.
				#	For all purposes for this compilation, control and data dependency is considered same.
				#	The reasoning being that everything is serial -- and graph execution is done in a 
				#		a topological sort.
				inputsRef[i] = '"' + curInput.split('^')[-1]

	# Create extra info dict
	# Format : (sizeInfo)
	extraInfoDict = {}
	for k,v in sizeInfo.items():
		extraInfoDict["\"" + k + "\""] = (v,)
	for curNode in graph.getAllNodesRef():
		if (curNode.getName() not in extraInfoDict):
			extraInfoDict[curNode.getName()] = (None,)
	
	print("Generating code from TF graph def : ", graphFileName, " ...")
	(program, dictNodeNameToOutVarStr) = generateIRCode(graph, extraInfoDict)

	print("SeeDot AST generation done. Pickling the AST.")
	with open(os.path.join(folderName, 'astOutput.pkl'), 'wb') as f:
		pickle.dump(program, f)

if __name__ == "__main__":
	main()
