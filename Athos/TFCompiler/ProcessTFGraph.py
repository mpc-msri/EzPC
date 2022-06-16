"""

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

"""

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SeeDot")
)  # Add SeeDot directory to path

import Graph, AST.AST as AST, _pickle as pickle, os
from TFNodesAST import TFNodesAST
from AST.PrintAST import PrintAST
from AST.MtdAST import MtdAST


def checkTFNodeNameForEq(curNodeOp: str, givenOp: str):
    return curNodeOp == '"' + givenOp + '"'


def generateASTForNode(graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict):
    curNodeOp = curNode.getOp()
    ast = None
    func = getattr(TFNodesAST, curNodeOp)
    (assignedVarAST, curASTs) = func(
        graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict
    )
    return (assignedVarAST, curASTs)


# Takes the graph DS and outputs IR in SeeDot for the same
def generateIRCode(graph, extraInfoDict):
    program = None
    innerMostLetASTNode = None
    dictNodeNameToOutVarStr = {}
    outVarCt = 0
    outVarPrefix = "J"
    mtdAST = MtdAST()
    for curNode in graph.getAllNodesRef():
        for curInp in curNode.getInputsRef():
            assert (
                curInp in dictNodeNameToOutVarStr
            ), "input={} expected as input for node={} but not yet processed".format(
                curInp, curNode.getName()
            )  # Consequence of topological sorting of the TF graph
        (assignedVarAST, curAsts) = generateASTForNode(
            graph, curNode, dictNodeNameToOutVarStr, extraInfoDict
        )
        for outputName, curAst in curAsts.items():
            mtdForCurAST = {
                AST.ASTNode.mtdKeyTFOpName: curNode.getOp(),
                AST.ASTNode.mtdKeyTFNodeName: outputName,
            }

            if curAst is None:
                dictNodeNameToOutVarStr[outputName] = None
                continue
            curOutVarStr = outVarPrefix + str(outVarCt)
            curOutVarAstNode = (
                assignedVarAST if assignedVarAST else AST.ID(curOutVarStr)
            )
            if program:
                assert type(innerMostLetASTNode) is AST.Let
                newNode = AST.Let(curOutVarAstNode, curAst, AST.ASTNode())
                mtdAST.visit(newNode, mtdForCurAST)
                innerMostLetASTNode.expr = newNode
                innerMostLetASTNode = newNode
            else:
                innerMostLetASTNode = AST.Let(
                    AST.ID(curOutVarStr), curAst, curOutVarAstNode
                )
                mtdAST.visit(innerMostLetASTNode, mtdForCurAST)
                innerMostLetASTNode.depth = 0
                program = innerMostLetASTNode
            dictNodeNameToOutVarStr[outputName] = curOutVarStr
            outVarCt += 1
    return (program, dictNodeNameToOutVarStr)


def addOutputs(program, dictNodeNameToOutVarStr, output_tensors):
    mtdAST = MtdAST()
    assert type(program) is AST.Let
    lastLetASTNode = program
    while True:
        if type(lastLetASTNode.expr) is AST.Let:
            lastLetASTNode = lastLetASTNode.expr
        else:
            break
    assert lastLetASTNode is not None
    if output_tensors is None:
        output_name = lastLetASTNode.name
        tf_node_name = lastLetASTNode.decl.metadata[AST.ASTNode.mtdKeyTFNodeName]
        print(
            "Output not specified, taking output of ",
            tf_node_name,
            " as program output.",
        )
        output = AST.Output(output_name, AST.Party.CLIENT)
        mtdForCurAST = {
            AST.ASTNode.mtdKeyTFOpName: "Output",
            AST.ASTNode.mtdKeyTFNodeName: tf_node_name,
        }
        mtdAST.visit(output, mtdForCurAST)
        lastLetASTNode.expr = output
    else:
        outVarCt = 0
        outVarPrefix = "O"
        for i in range(0, len(output_tensors)):  # name, decl, expr
            t_name = output_tensors[i]
            if t_name not in dictNodeNameToOutVarStr:
                if ":" in t_name:
                    try:
                        op_name, out_n = t_name.split(":")
                        out_n = int(out_n)
                    except:
                        raise ValueError(
                            "The tensor name {} looks like a tensor name but is not a valid one".format(
                                name
                            )
                        )
                    if out_n == 0:
                        if op_name in dictNodeNameToOutVarStr:
                            t_name = op_name
                        else:
                            t_name = op_name + "_mpc_const_var"
                    else:
                        t_name = op_name + "_mpc_const_var" + ":" + str(out_n)
                else:
                    t_name += "_mpc_const_var"
            if i == len(output_tensors) - 1:
                output_name = AST.ID(dictNodeNameToOutVarStr[t_name])
                output = AST.Output(output_name, AST.Party.CLIENT)
                newNode = output
            else:
                output_name = AST.ID(dictNodeNameToOutVarStr[t_name])
                output = AST.Output(output_name, AST.Party.CLIENT)
                let_name_id = AST.ID(outVarPrefix + str(outVarCt))
                newNode = AST.Let(let_name_id, output, AST.ASTNode())
                mtdForCurAST = {
                    AST.ASTNode.mtdKeyTFOpName: "Output",
                    AST.ASTNode.mtdKeyTFNodeName: t_name,
                }
                mtdAST.visit(newNode, mtdForCurAST)
            lastLetASTNode.expr = newNode
            lastLetASTNode = newNode
            outVarCt += 1

    return program


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
        if not tokens:
            nodeOPSize = [1]
        else:
            for curDimStr in tokens:
                if curDimStr == "":
                    continue
                nodeOPSize.append(int(curDimStr))
        sizeInfo[nodeName] = nodeOPSize
    return sizeInfo


# Since later on in the pipeline, the placeholder nodes which come up as cin statements
# are to be excluded from the timing calculation, output all such PlaceHolder nodes together first.
# This doesn't violate the topological ordering because all such PlaceHolder nodes are leaf nodes
# in the graph.
# This however extends live ranges of inputs and increases peak memory usage.
# This also maintains the partial order between placeholder/variable nodes
def prefixAllPlaceHolderNodes(graph):
    allNodes = graph.getAllNodesRef()
    placeHolderNodes = []
    variableNodes = []
    remNodes = []
    for curNode in allNodes:
        if curNode.getOp() in ["Placeholder", "VariableV2"]:
            assert len(curNode.getInputsRef()) == 0  # Assert this is indeed a leaf node
            placeHolderNodes.append(curNode)
        else:
            remNodes.append(curNode)
    graph.setNodesList(placeHolderNodes + remNodes)


# List of Optimisations
# 1. Split squared difference into (a-b)*(a-b)
# 2. Reshape filter of depth separable convolution to convert it to a grouped convolution
def simplifyGraph(graph, sizeInfo):
    allNodes = graph.getAllNodesRef()
    nodesMap = graph.getAllNodes()
    newNodes = []
    inputsFixup = {}
    for curNode in allNodes:
        inputs = curNode.getInputsRef()
        for i in range(len(inputs)):
            if inputs[i] in inputsFixup:
                inputs[i] = inputsFixup[inputs[i]]
        if curNode.getOp() == "SquaredDifference":
            sub = Graph.Node("Sub", inputs.copy(), curNode.getName() + "__sub")
            mul = Graph.Node(
                "Mul", [sub.getName(), sub.getName()], curNode.getName() + "__mul"
            )
            newNodes.append(sub)
            newNodes.append(mul)
            nodesMap[sub.getName()] = sub
            nodesMap[mul.getName()] = mul
            inputsFixup[curNode.getName()] = mul.getName()
            nodesMap.pop(curNode.getName())
        elif curNode.getOp() == "DepthwiseConv2dNative":
            filter_shape = sizeInfo[inputs[1]]
            in_channels = filter_shape[2]
            channel_multiplier = filter_shape[3]
            output_channels = in_channels * channel_multiplier
            # new filter shape = [FH, FW, 1, CI*CM]
            new_filter_shape = filter_shape[0:2] + [1, output_channels]
            reshape = Graph.Node(
                "Reshape", [inputs[1]], curNode.getName() + "__reshape"
            )
            newNodes.append(reshape)
            newNodes.append(curNode)
            nodesMap[reshape.getName()] = reshape
            inputs[1] = reshape.getName()
            sizeInfo[reshape.getName()] = new_filter_shape
        else:
            newNodes.append(curNode)
    graph.setNodesList(newNodes)


# We have to process all input nodes before output nodes.
# However we cannot change the partial order of the placeholder and variable nodes.
# The model weights are dumped from tensorflow in the original graphdef order and if
# we don't adhere to that, different inputs will be read by the program.
def arrange_input_before_output(graph):
    allNodes = graph.getAllNodesRef()
    visited = set()
    already_sorted = True
    for curNode in allNodes:
        visited.add(curNode.getName())
        for inp in curNode.getInputsRef():
            if inp not in visited:
                already_sorted = False
                break

    # True almost all the time
    if already_sorted:
        return

    adjList = {i: [] for i in range(len(allNodes))}
    position = {node.getName(): i for i, node in enumerate(allNodes)}
    for i, curNode in enumerate(allNodes):
        inputs = curNode.getInputsRef()
        for inp in inputs:
            adjList[position[inp]].append(i)

    # Additionally create edges between all placeholder and variable nodes
    nodes_seen = []
    for i, curNode in reversed(list(enumerate(allNodes))):
        if curNode.getOp() in ["Placeholder", "VariableV2"]:
            adjList[i].extend(nodes_seen)
            nodes_seen.append(i)

    no_nodes = len(allNodes)
    visited = [False] * no_nodes
    final_order = []

    def topo_sort(v):
        visited[v] = True
        for i in adjList[v]:
            if visited[i] == False:
                topo_sort(i)
        final_order.insert(0, v)

    for i in range(no_nodes):
        if visited[i] == False:
            topo_sort(i)

    assert len(final_order) == no_nodes, "Lost some nodes while sorting"
    newNodes = [allNodes[i] for i in final_order]
    graph.setNodesList(newNodes)
    return


def process_tf_graph(filename, output_tensors=None):
    sys.setrecursionlimit(10000)

    if os.path.isfile(filename):
        folderName = os.path.dirname(filename)
    else:
        folderName = filename
    graphFileName = os.path.join(folderName, "graphDef.mtdata")
    graph = Graph.Graph()
    with open(graphFileName) as file:
        graph.readFromFilePointer(file)

    arrange_input_before_output(graph)

    # Read the sizeInfo also
    sizeInfoFileName = os.path.join(folderName, "sizeInfo.mtdata")
    sizeInfo = readSizeInfo(sizeInfoFileName)

    # Tensorflow graph level optimisations
    simplifyGraph(graph, sizeInfo)
    # Place all PlaceHolder and variable nodes together at the beginning
    prefixAllPlaceHolderNodes(graph)

    # Re-format the input names of nodes
    for curNode in graph.getAllNodesRef():
        inputsRef = curNode.getInputsRef()
        for i, curInput in enumerate(inputsRef):
            if curInput.startswith("^"):
                # My hypothesis from empirical observation is that inputs which have '^' ahead of the node name
                #   denote control flow dependency and not data dependency.
                #   For all purposes for this compilation, control and data dependency is considered same.
                #   The reasoning being that everything is serial -- and graph execution is done in a
                #       a topological sort.
                inputsRef[i] = curInput.split("^")[-1]

    # Create extra info dict
    # Format : (sizeInfo)
    extraInfoDict = {}
    for k, v in sizeInfo.items():
        extraInfoDict[k] = (v,)
    for curNode in graph.getAllNodesRef():
        if curNode.getName() not in extraInfoDict:
            extraInfoDict[curNode.getName()] = (None,)

    print("Generating code from TF graph def : ", graphFileName, " ...")
    (program, dictNodeNameToOutVarStr) = generateIRCode(graph, extraInfoDict)
    program = addOutputs(program, dictNodeNameToOutVarStr, output_tensors)

    print("SeeDot AST generation done. Pickling the AST.")
    with open(os.path.join(folderName, "astOutput.pkl"), "wb") as f:
        pickle.dump(program, f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("TF python file unspecified.", file=sys.stderr)
        exit(1)

    filename = sys.argv[1]
    process_tf_graph(filename)
