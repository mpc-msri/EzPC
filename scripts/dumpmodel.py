import torch
import struct

# get this file from https://github.com/vaibhavhaswani/resnet9-cifar10-pt
state_dict = torch.load("cifar10_resnet9-01.pth", map_location=torch.device('cpu'))

with open("cifar10_resnet9-float.dat", 'wb') as f:
  for k in state_dict:
    print(k)
    dims = len(state_dict[k].size())
    if dims == 4:
      CO = state_dict[k].size()[0]
      CI = state_dict[k].size()[1]
      FH = state_dict[k].size()[2]
      FW = state_dict[k].size()[3]
      print("co =", CO, "ci =", CI, "fh =", FH, "fw =", FW, "type =", struct.pack('f', state_dict[k][0][0][0][0].item()))
      for co in range(CO):
        for fh in range(FH):
          for fw in range(FW):
            for ci in range(CI):
              f.write(struct.pack('f', state_dict[k][co][ci][fh][fw].item()))
    elif dims == 2:
      CO = state_dict[k].size()[0]
      CI = state_dict[k].size()[1]
      print("co =", CO, "ci =", CI)
      for ci in range(CI):
        for co in range(CO):
          f.write(struct.pack('f', state_dict[k][co][ci].item()))
    elif dims == 1:
      C =  state_dict[k].size()[0]
      print("c =", C)
      for c in range(C):
        f.write(struct.pack('f', state_dict[k][c].item()))