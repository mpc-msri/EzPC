"""

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

"""
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


def copy_dict(dict_src: dict, diff={}):
    dict_res = dict(dict_src)
    dict_res.update(diff)
    return dict_res


# z = [y1,y2,..] = [[x1,..], [x2,..], ..] --> [x1,.., x2,.., ..]
def flatten(z: list):
    return [x for y in z for x in y]


def write_debug_info(name_mapping):
    if not os.path.exists("debug"):
        os.makedirs("debug")

    with open("debug/seedot_ezpc_name_map.pkl", "wb") as f:
        pickle.dump(name_mapping, f)

    with open("debug/seedot_ezpc_name_map.txt", "w") as f:
        for val in name_mapping:
            f.write(val + "   " + name_mapping[val] + "\n")


# Broadcasting Rules:
# A      (4d array):  8 x 1 x 6 x 1
# B      (3d array):      7 x 1 x 5
# Result (4d array):  8 x 7 x 6 x 5
# Return Values
# Shape A broadcast mask:  [False, True, False, True]
# Shape B broadcast mask:  [True, False, True, False]
# Result shape:  [8, 7, 6, 5]
#
# If input is a scalar, pass shape as []
def getBroadcastShapes(Shape1: list, Shape2: list):
    # Broadcast rules apply in reverse direction
    shape1 = Shape1[::-1]
    shape2 = Shape2[::-1]
    len1 = len(shape1)
    len2 = len(shape2)
    outputshape = []
    swapped = False
    if len1 != len2:
        if len1 > len2:
            len1, len2 = len2, len1
            shape1, shape2 = shape2, shape1
            swapped = True
        assert len1 < len2

    broadcastMask1 = [False] * len1
    broadcastMask2 = [False] * len2

    for i in range(len2):
        length = 0
        if i >= len1:
            # broadcastMask1[i] = True
            outputshape.append(shape2[i])
            continue
        if shape1[i] != shape2[i]:
            if shape1[i] == 1:
                outputshape.append(shape2[i])
                broadcastMask1[i] = True
            elif shape2[i] == 1:
                outputshape.append(shape1[i])
                broadcastMask2[i] = True
            else:
                print("Dimension no. {} has a mismatch of length.".format(len2 - i))
                assert (
                    False
                ), "Cannot broadcast. Program is malformed. Atleast one length should have been 1. i1: {} i2: {}".format(
                    shape1[i], shape2[i]
                )
        else:
            outputshape.append(shape1[i])

    if swapped:
        broadcastMask1, broadcastMask2 = broadcastMask2, broadcastMask1

    outputshape.reverse()
    broadcastMask1.reverse()
    broadcastMask2.reverse()
    return outputshape, broadcastMask1, broadcastMask2


def get_volume(shape: list):
    vol = 1
    for i in shape:
        vol = vol * i
    return vol


class DisjointSet:
    class Node:
        def __init__(self):
            self.parent = self
            self.children = []

        def get_root(self):
            if self.parent != self:
                old_parent = self.parent
                self.parent = self.parent.get_root()
                if self.parent != old_parent:
                    self.parent.children.append(self)
                    old_parent.children.remove(self)
                return self.parent
            else:
                return self

        def get_all_children(self):
            all_children = []
            all_children.extend(self.children)
            tmp = []
            for i in all_children:
                tmp.extend(i.get_all_children())
            all_children.extend(tmp)
            return all_children

    def __init__(self):
        self.key_to_node = {}
        self.node_to_key = {}

    def inSet(self, inp):
        return inp in self.key_to_node

    def make_set(self, inp):
        if self.inSet(inp):
            return
        n = self.Node()
        self.key_to_node[inp] = n
        self.node_to_key[n] = inp

    def union(self, inp1, inp2):
        n1 = self.key_to_node[inp1]
        n2 = self.key_to_node[inp2]
        r1 = n1.get_root()
        r2 = n2.get_root()
        if r1 != r2:
            r1.parent = r2
            r2.children.append(r1)

    def find(self, inp):
        if not self.inSet(inp):
            return None
        return self.key_to_node[inp].get_root()

    def find_key(self, inp):
        node = self.find(inp)
        if node is None:
            return None
        return self.node_to_key[node]

    def get_set(self, inp):
        if not self.inSet(inp):
            return None
        n = self.key_to_node[inp].get_root()
        return [n] + n.get_all_children()

    def get_key_set(self, inp):
        nodes = self.get_set(inp)
        if nodes is None:
            return None
        return [self.node_to_key[i] for i in nodes]

    def print(self):
        print(self.key_to_node)
        print(self.node_to_key)

    def print_set(self, inp):
        print(self.get_key_set(inp))
