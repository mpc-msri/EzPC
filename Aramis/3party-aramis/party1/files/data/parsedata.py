# Written by Mayank
# This program is used to first parse the dataset files for training
# and testing and then compile them into dataparsed.cpp file
# from where we can load the data into sgx enclave.

accdataAC = ""
accdataBD = ""
acclabelAC = ""
acclabelBD = ""
accdatasample = ""
acclabelsample = ""

#First accdataAC
file = open("mnist_data_8_AC", "r")
while True:
    curline = file.readline()
    #print(curline)
    if len(curline) is 0:
        break
    curline = curline[:-1]
    if len(curline) > 0:
        accdataAC += curline + ' '

file.close()

#First accdataBD
file = open("mnist_data_8_BD", "r")
while True:
    curline = file.readline()
    #print(curline)
    if len(curline) is 0:
        break
    curline = curline[:-1]
    if len(curline) > 0:
        accdataBD += curline + ' '

file.close()

#First acclabelAC
file = open("mnist_labels_8_AC", "r")
while True:
    curline = file.readline()
    #print(curline)
    if len(curline) is 0:
        break
    curline = curline[:-1]
    if len(curline) > 0:
        acclabelAC += curline + ' '

file.close()

#First acclabelBD
file = open("mnist_labels_8_BD", "r")
while True:
    curline = file.readline()
    #print(curline)
    if len(curline) is 0:
        break
    curline = curline[:-1]
    if len(curline) > 0:
        acclabelBD += curline + ' '

file.close()

#First accdatasample
file = open("mnist_data_8_samples", "r")
while True:
    curline = file.readline()
    #print(curline)
    if len(curline) is 0:
        break
    curline = curline[:-1]
    if len(curline) > 0:
        accdatasample += curline + ' '

file.close()

#First acclabelsample
file = open("mnist_labels_8_samples", "r")
while True:
    curline = file.readline()
    #print(curline)
    if len(curline) is 0:
        break
    curline = curline[:-1]
    if len(curline) > 0:
        acclabelsample += curline + ' '

file.close()

# Slice on whitespaces
split_dataAC = accdataAC.split()
split_dataBD = accdataBD.split()
split_labelAC = acclabelAC.split()
split_labelBD = acclabelBD.split()
split_datasample = accdatasample.split()
split_labelsample = acclabelsample.split()

cppfile = open("dataparsed.cpp", "w")
cppfile.seek(0)
cppfile.truncate()

boilerplate = "#include \"dataparsed.h\"\n"
cppfile.write(boilerplate)

lenlist = [];

#DataAC
declaration_full = ""
curlen = len(split_dataAC)
lenlist.append(curlen)
declaration_full = declaration_full + "int dataAC["+str(curlen)+"] = {"

for i in range(curlen):
    if i == curlen-1:
        declaration_full = declaration_full + split_dataAC[i]
    else:
        declaration_full = declaration_full + split_dataAC[i] + ", "

declaration_full = declaration_full + "};\n"
cppfile.write(declaration_full)

#DataBD
declaration_full = ""
curlen = len(split_dataBD)
lenlist.append(curlen)
declaration_full = declaration_full + "int dataBD["+str(curlen)+"] = {"

for i in range(curlen):
    if i == curlen-1:
        declaration_full = declaration_full + split_dataBD[i]
    else:
        declaration_full = declaration_full + split_dataBD[i] + ", "

declaration_full = declaration_full + "};\n"
cppfile.write(declaration_full)

#LabelAC
declaration_full = ""
curlen = len(split_labelAC)
lenlist.append(curlen)
declaration_full = declaration_full + "int labelAC["+str(curlen)+"] = {"

for i in range(curlen):
    if i == curlen-1:
        declaration_full = declaration_full + split_labelAC[i]
    else:
        declaration_full = declaration_full + split_labelAC[i] + ", "

declaration_full = declaration_full + "};\n"
cppfile.write(declaration_full)

#LabelBD
declaration_full = ""
curlen = len(split_labelBD)
lenlist.append(curlen)
declaration_full = declaration_full + "int labelBD["+str(curlen)+"] = {"

for i in range(curlen):
    if i == curlen-1:
        declaration_full = declaration_full + split_labelBD[i]
    else:
        declaration_full = declaration_full + split_labelBD[i] + ", "

declaration_full = declaration_full + "};\n"
cppfile.write(declaration_full)

#Datasample
declaration_full = ""
curlen = len(split_datasample)
lenlist.append(curlen)
declaration_full = declaration_full + "int datasample["+str(curlen)+"] = {"

for i in range(curlen):
    if i == curlen-1:
        declaration_full = declaration_full + split_datasample[i]
    else:
        declaration_full = declaration_full + split_datasample[i] + ", "

declaration_full = declaration_full + "};\n"
cppfile.write(declaration_full)

#Labelsample
declaration_full = ""
curlen = len(split_labelsample)
lenlist.append(curlen)
declaration_full = declaration_full + "int labelsample["+str(curlen)+"] = {"

for i in range(curlen):
    if i == curlen-1:
        declaration_full = declaration_full + split_labelsample[i]
    else:
        declaration_full = declaration_full + split_labelsample[i] + ", "

declaration_full = declaration_full + "};\n"
cppfile.write(declaration_full)

print("Contents of data files compiled and written to dataparsed.cpp")
cppfile.close()

cppheader = open("dataparsed.h", "w")
cppheader.seek(0)
cppheader.truncate()
headerinfo = "#include <string>\n#include <stdio.h>\n#include <stdlib.h>\nextern int dataAC["+str(lenlist[0])+"];\nextern int dataBD["+str(lenlist[1])+"];\nextern int labelAC["+str(lenlist[2])+"];\nextern int labelBD["+str(lenlist[3])+"];\nextern int datasample["+str(lenlist[4])+"];\nextern int labelsample["+str(lenlist[5])+"];"
cppheader.write(headerinfo)
cppheader.close()
print("Header file written")
