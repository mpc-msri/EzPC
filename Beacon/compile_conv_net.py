import torch
import pickle
import sys

nn = torch.nn
F = nn.functional
optim = torch.optim
# torch.set_printoptions(precision=30)
dtype = torch.float32

import beacon_frontend_conv as bcf_conv

if __name__ == "__main__":
    pickle_file = sys.argv[1]

    with open(pickle_file, "rb") as file:
        net = pickle.load(file)

    print(net)
