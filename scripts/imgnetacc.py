actual_labels = list(map(lambda x: int(x.rstrip()), open("/data/kanav/dataset/ImageNet_ValData/imagenet12_val_nlabels.txt").readlines()))

print(len(actual_labels))

correct = 0
tot = 0
for t in range(4):
    observed_labels = list(map(lambda x: (int(x.rstrip().split(' ')[0]), int(x.rstrip().split(' ')[1])), open("thread_outputs/thread-" + str(t)).readlines()))
    tot += len(observed_labels)
    for p in observed_labels:
        i, l = p
        if actual_labels[i-1] == l:
            correct += 1

print(correct, "/", tot)
print(100.0 * correct / tot)
