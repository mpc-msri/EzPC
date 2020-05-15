import numpy as np
f = open("debug/onnx_output.txt", 'r')

class_vals = [0, 0, 0]
mx = float('-inf')
cls = ""
lines = f.readlines()
amax = -1
arr = np.arange(3)
arr = arr.astype(np.float32)
# print(arr.shape)
for i in range(len(class_vals)):
    # class_vals[i] = float(lines[i])
    arr[i] = float(lines[i][:-1])
    # mx = max(mx, class_vals[i])
    # if(mx == class_vals[i]):
        # amax = i
        # if(i == 0):
            # cls = "Covid19"
        # elif(i == 1):
            # cls = "Normal"
        # else:
            # cls = "Pneumonia"
# assert(np.argmax(arr) == amax)
# print(arr)
amax = np.argmax(arr)
if(amax == 0):
    cls = "Covid19"
elif(amax == 1):
    cls = "Normal"
elif(amax == 2):
    cls = "Pneumonia"

print(cls)
f.close()

