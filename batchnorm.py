

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=20)

lr_rate = 1.0/64.0
momentum = 58.0 / 64.0

class Model(torch.nn.Module):
    def __init__(self, channels):
        super(Model, self).__init__()
        self.bn = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.bn(x)
        return y.view(-1, 6)

model = Model(3)
img = torch.ones(3, 3, 1, 2)
img[0][0][0][0] = 4567
img[0][0][0][1] = 4567
img[0][1][0][0] = 328
img[0][1][0][1] = 328
img[0][2][0][0] = 9785
img[0][2][0][1] = 9785
img[1][0][0][0] = 3109
img[1][0][0][1] = 3109
img[1][1][0][0] = 2389
img[1][1][0][1] = 2389
img[1][2][0][0] = 238
img[1][2][0][1] = 238
img[2][0][0][0] = 5478
img[2][0][0][1] = 5478
img[2][1][0][0] = 623
img[2][1][0][1] = 623
img[2][2][0][0] = 2349
img[2][2][0][1] = 2349
# img = torch.ones(2, 3, 1, 1)
# img[0][0][0][0] = 0
# img[0][1][0][0] = 1
# img[0][2][0][0] = 2
# img[1][0][0][0] = 3
# img[1][1][0][0] = 4
# img[1][2][0][0] = 5

# model.eval()
# y = model(img)
# print(y)

# now training
label = torch.tensor([0, 0, 0])
model.train()
# print("#### BEFORE ####")
# print(model.bn.weight.data)
# print(model.bn.bias.data)
# print(output)
# print(model.bn.running_mean)
# print(model.bn.running_var)
# exit(0)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=momentum)


for i in range(5):
    output = model(img)
    optimizer.zero_grad()
    loss = criterion(output,label)
    loss.backward()
    optimizer.step()


print("#### AFTER ####")
print(model.bn.weight.data)
print(model.bn.bias.data)

model.eval()
output = model(img)
print(output)