import sys
filename1 = sys.argv[1]
filename2 = sys.argv[2]

file1 = open(filename1)
file2 = open(filename2)

arr1 = []
arr2 = []

# read an integer from each file, a line might contain multiple integers
# so we need to split the line into a list of integers
# then we can iterate over the list and compare each integer
for line1 in file1:
    for num1 in line1.split():
        num1 = int(num1)
        arr1.append(num1)

for line2 in file2:
    for num2 in line2.split():
        num2 = int(num2)
        arr2.append(num2)

assert(len(arr1) == len(arr2))
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

print(np.array(arr1) - np.array(arr2))
