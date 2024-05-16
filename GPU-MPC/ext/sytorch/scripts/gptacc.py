# 
# Copyright:
# 
# Copyright (c) 2024 Microsoft Research
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

nt = 8

folder = "results_final_4june"
corr = 0
tot = 0
lab = "pos"
for i in range(nt):
    lines = open(folder + "/" + lab + "/thread-" + str(i)).readlines()
    for l in lines:
        # print(l.split(" ")[3])
        if lab == l.split(" ")[3].strip():
            corr += 1
        tot += 1

lab = "neg"
for i in range(nt):
    lines = open(folder + "/" + lab + "/thread-" + str(i)).readlines()
    for l in lines:
        # print(l.split(" ")[3])
        if lab == l.split(" ")[3].strip():
            corr += 1
        tot += 1

print(corr, tot)
print(corr / tot)
