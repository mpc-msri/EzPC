ground_truth = open('../../../transformers/gpt2-qnli/dataset/labels.txt').readlines()
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

print('Accuracy: %.2f%%' % (100.0 * corr / len(ground_truth)))
print(corr, len(ground_truth))