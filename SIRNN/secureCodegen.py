#! /usr/bin/env python


# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import os
from os import writev
import subprocess
import numpy
import argparse

modelParams = []
varSizeDict = {}
outputVar = ""
datasetname = ""


def readfile(dir, filename):
    cur_dir = os.path.abspath(os.curdir)
    os.chdir(os.path.join(cur_dir, dir))
    infile = open(filename).read().splitlines()
    os.chdir(cur_dir)
    return infile


def writefile(args, dir, filename, file):
    cur_dir = os.getcwd()
    os.chdir(dir)
    outfile = open(filename, "w")
    for i in range(len(file)):
        outfile.write(file[i])
        outfile.write("\n")
    outfile.close()
    os.chdir(cur_dir)


def mainToFuncWithHeader(file):
    header = [
        "#include <iostream>",
        "#include <cstring>",
        "#include <cmath>",
        "",
        '#include "defines.h"',
        '#include "datatypes.h"',
        '#include "predictors.h"',
        '#include "library_fixed.h"',
        '#include "model_fixed.h"',
        '#include "vars_fixed.h"',
        "",
        "using namespace std;",
        "using namespace sirnn_fixed;",
        "",
        "extern int party;",
        "extern string address;",
        "extern int port;",
        "extern int num_threads;",
        "void sirnnFixed(MYINT *Xtemp, int64_t* res) {",
    ]
    mainlineno = file.index("int main () {")
    trimmedfile = file[mainlineno + 1 :]
    header.extend(trimmedfile)
    return header


def getLocStrings(args):
    srcDir = args.predict_dir
    dstDir = args.results_dir
    infile = "predict0.cpp"
    outfile = "sirnn_fixed.cpp"
    return srcDir, dstDir, infile, outfile


def removeCinsCouts(file):
    global varSizeDict
    global outputVar
    global modelParams
    for i in range(len(file)):
        spaceless = file[i].lstrip()

        if spaceless.find("make_vector") != -1:
            if spaceless.find("temp") != -1:
                keywords = spaceless.rstrip().split(" ")
                for j in range(len(keywords)):
                    keyword = keywords[j]
                    if keyword.find("temp") != -1:
                        modelParams.append(keyword[:-4])
                file[i] = ""
            else:
                varname = spaceless.split(" ")[1]
                lenStart = spaceless.find("(")
                size = spaceless[lenStart:-1]
                varSizeDict[varname] = size
                pass
        elif spaceless[0:3] == "cin":
            file[i] = ""
        elif spaceless[0:4] == "cout":
            if spaceless.find("Value of") != -1:
                start = spaceless.find("(")
                end = spaceless.find(")")
                varname = spaceless[start + 2 : end - 2].split(" ")[-1]
                outputVar = varname
            file[i] = ""
        if spaceless.find("=") != -1:
            splitline = spaceless.split("=")
            if splitline[0].find("temp") != -1:
                file[i] = ""
    return file


def addPartyDistinction(file):
    for i in range(len(file)):
        spaceless = file[i].lstrip().rstrip()
        if spaceless.find("=") != -1:
            splitline = spaceless.split("=")
            party = 1
            newline = ""
            for j in range(len(modelParams)):
                if splitline[0].find(modelParams[j]) != -1:
                    if splitline[1].find("make_vector") == -1:
                        if modelParams[j] == "X":
                            party = 2
                        newline = splitline[0] + " = ( party == %d ? %s : 0);" % (
                            party,
                            splitline[1].replace(";", ""),
                        )
                        modelParams.pop(j)
                        file[i] = newline
                        break
    return file


def correctTempName(file):
    for i in range(len(file)):
        file[i] = file[i].replace("temp", "_temp")
    return file


def addReconstruct(file):
    bwStart = outputVar.find("bw")
    bw = 16
    if bwStart != -1:
        bw = outputVar[bwStart + 2 :]
    funccall = "reconstruct(%s, res, 1, %s, %s);" % (
        outputVar,
        varSizeDict[outputVar],
        bw,
    )
    file.insert(-5, funccall)
    return file


def addSuffix(file):
    file[-3] = "return;"
    suffix = [
        "const int switches = 0;",
        "void sirnnFixedSwitch(int i, MYINT **X_temp, int32_t* res) {",
        "	switch(i) {",
        "		default: res[0] = -1; return;",
        "	}",
        "}",
    ]
    file.extend(suffix)
    return file


def deallocateModelAndVars(file):
    for var in varSizeDict.keys():
        file.insert(-4, "delete[] %s;" % (var))
    return file


def removeComments(file):
    for i in range(len(file)):
        if file[i][0:2] == "/*" or file[i][0:2] == "//":
            file[i] = ""
    return file


def replaceMakeVector(file):
    for i in range(len(file)):
        line = file[i].rstrip()
        if line.find("make_vector") != -1:
            typeEnd = line.find(">")
            line = line[:typeEnd] + "[" + line[typeEnd + 2 :]
            line = line.replace("make_vector", "new ")
            line = line.replace("<", "")
            line = line.replace(">", "")
            line = line[:-2] + "]" + ";"
            file[i] = line
    return file


def getLeftWhiteSpace(string):
    return string.count(" ") + 2 * string.count("\t")


def createSpace(count):
    string = "".join(" " for _ in range(count))
    return string


def replaceDivisions(file):
    start = False
    for i in range(len(file)):
        spaceless = file[i]
        if spaceless.find("main") != -1:
            start = True
        if start:
            if (spaceless.find("/") != -1) and (spaceless.find("temp") == -1):
                # print(i)
                # input()
                numFors = int(spaceless.count("+") / 2) + 1
                size = ""
                for j in range(numFors):
                    forLine = (file[i - 2 * numFors + j]).lstrip().rstrip()
                    point = forLine.find(":")
                    length = forLine[point + 1 :]
                    length = length[: length.find("]")]
                    # print(length)
                    size = size + length + "*"
                size = "(" + size[:-1] + ")"

                to, fro = spaceless.split("=")
                to = to.rstrip()
                numSpaces = getLeftWhiteSpace(to)
                indentSpaces = createSpace(numSpaces)
                fro = fro.lstrip().rstrip()
                fro = fro[
                    2:-3
                ]  # Removing parantheses and semicolon at the end of the line
                froVar, shr = fro.split("/")
                froVar = froVar.lstrip().rstrip()
                varInd = froVar[froVar.find("[") :]
                froVar = froVar[: froVar.find("[")]
                shr = shr.lstrip().rstrip()[:-1]
                adjustStr = "%sAdjustScaleShr(1L, %s, %s, %s, %s);" % (
                    indentSpaces,
                    size,
                    shr,
                    froVar[froVar.find("bw") + 2 :],
                    froVar,
                )
                file.insert(i - 2 * numFors, adjustStr)
                file[i + 1] = "%s = %s%s;" % (to, froVar, varInd)
    return file


def runEzPC(args, dir, file):
    global datasetname
    cur_dir = os.getcwd()
    os.chdir(args.ezpc_dir)
    # proc = subprocess.Popen("opam switch 4.05.0 && eval $(opam env)", stdout=subprocess.PIPE)
    proc = subprocess.Popen(
        "./ezpc.sh %s/%s --bitlen 64 --codegen CPP --disable-cse" % (dir, file),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate()
    # print(out.decode() + "\n\n" + err.decode())
    if out.decode().find("error") != -1:
        assert (
            False
        ), "Common solutions to error: Try building predict.ezpc file  independently and Initialise the opam environment."
    os.chdir(cur_dir)


def appendLibrary(file):

    lib = open("Library_SIRNN.ezpc").read().split("\n")
    lib.extend(file)
    return lib


def replaceDivisionsAndRunEzPC(args, srcDir):
    global datasetname
    filename = "predict.ezpc"
    append_lib = True

    file = open(os.path.join(srcDir, filename)).read()
    # Check if SIRNN library needs to be added to the file
    if file.find("(* <><><><><><><> Auto-generated code <><><><><><><> *)") != -1:
        append_lib = False
    file = file.split("\n")

    noDivFile = replaceDivisions(file)

    if append_lib:
        noDivFile = appendLibrary(noDivFile)

    writefile(args, srcDir, filename, noDivFile)
    runEzPC(args, os.path.abspath(srcDir), filename)


def indentFile(file):
    indent = 0
    for i in range(len(file)):
        f = file[i]
        if f.find("}") != -1 and f.find("{") != -1:
            pass
        elif f.find("}") != -1:
            indent = indent - 1
        indentnow = "".join("\t" for j in range(indent))
        file[i] = indentnow + file[i]
        if f.find("{") != -1:
            indent = indent + 1
    return file


def run(args):
    srcDir, dstDir, infile, outfile = getLocStrings(args)
    replaceDivisionsAndRunEzPC(args, srcDir)

    file = readfile(srcDir, infile)
    newfile = mainToFuncWithHeader(file)

    convfile = removeCinsCouts(newfile)

    partyfile = addPartyDistinction(convfile)
    tempcorrect = correctTempName(partyfile)
    reconstructFile = addReconstruct(tempcorrect)
    deallocatedFile = deallocateModelAndVars(reconstructFile)
    fullfile = addSuffix(deallocatedFile)
    fullfile_with_new = replaceMakeVector(fullfile)
    nocomment = removeComments(fullfile_with_new)
    indentedFile = indentFile(nocomment)
    writefile(args, dstDir, outfile, indentedFile)


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-pdir",
        "--predict_dir",
        type=str,
        metavar="",
        help="The location of the 'predict.ezpc' file to be converted to SIRNN C++ format.",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        choices=["Google-30"],
        default="Google-30",
        metavar="",
        help="Dataset to use ['Google-30'] \
                           (Default: 'Google-30')",
    )
    parser.add_argument(
        "--ezpc_dir",
        type=str,
        metavar="",
        default="../EzPC/EzPC/",
        help="The path to location with 'ezpc.sh'",
    )
    parser.add_argument(
        "-rdir",
        "--results_dir",
        type=str,
        metavar="",
        help="Location to store the output 'seedot_fixed.cpp' file.",
    )
    parser.add_argument(
        "-ptype",
        "--problem_type",
        choices=["regression", "classification"],
        metavar="",
        default="classification",
        help="Whether this is an instance of regression or classification problem.",
    )

    args = parser.parse_args()
    return args


def main():
    global modelParams
    global varSizeDict
    global outputVar
    global datasetname

    args = parseArgs()
    # datasetnamelist = ["Google-12", "Google-30", "HAR-2", "HAR-6", "MNIST-10", "Wakeword-2", "spectakoms", "dsa", "usps10"]
    # datasetnamelist = ["face-2"]
    if not isinstance(args.datasets, list):
        datasetnamelist = [args.datasets]
    else:
        datasetnamelist = args.datasets

    for dataset_name in datasetnamelist:
        print("dataset: ", dataset_name)
        modelParams = []
        varSizeDict = {}
        outputVar = ""
        datasetname = dataset_name
        run(args)


if __name__ == "__main__":
    main()

    # convertFunCall(["MatMul( (int64_t)1,  (int64_t)8,  (int64_t)64,  (int64_t)1,  (int64_t)1,  (int64_t)0,  (int64_t)3,  (int64_t)64,  (int32_t)8,  (int32_t)8,  (int32_t)16,  (int32_t)8, tmp24, U2, tmp26, tmp25);"])
