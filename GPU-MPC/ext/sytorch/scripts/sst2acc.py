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

ground_truth = open('../../../transformers/datasets/sst2/labels.txt').readlines()
ground_truth = [int(x.strip()) for x in ground_truth]

import sys
import numpy as np
predictions = open(sys.argv[1]).readlines()

corr = 0
for i in range(len(ground_truth)):
    pred = predictions[i].strip().split(' ')
    pred = [float(x) for x in pred]
    pred = np.argmax(pred)
    if pred == ground_truth[i]:
        corr += 1

print('Accuracy: %d / %d = %.2f%%' % (corr, len(ground_truth), (100.0 * corr / len(ground_truth))))