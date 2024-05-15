import sys

f1 = open(sys.argv[1], 'r').readlines()
f2 = open(sys.argv[2], 'r').readlines()
assert(len(f1) == len(f2))

for i in range(len(f1)):
    if f1[i] != f2[i]:
        print(i)
        print(f1[i])
        print(f2[i])
        break
