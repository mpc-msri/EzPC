import torch
from torch.autograd import Variable

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(2)

    def forward(self, x):
        return self.avgpool(self.conv1(x))

model = Model()
model.conv1.weight.data[0][0][0][0] = 1.0
model.conv1.weight.data[0][0][0][1] = 2.0
model.conv1.weight.data[0][0][1][0] = 3.0
model.conv1.weight.data[0][0][1][1] = 4.0

model.conv1.bias.data[0] = 0.0

e = torch.ones(1, 1, 2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

image = Variable(torch.ones(1, 1, 3, 3), requires_grad=True)
output = model(image)
output.backward(e)
optimizer.step()

optimizer.zero_grad()
image = Variable(torch.ones(1, 1, 3, 3), requires_grad=True)
output = model(image)
output.backward(e)
optimizer.step()


print(model.conv1.weight.grad)
print(model.conv1.weight.data)
print(image.grad)
# print(model.conv1.weight.grad)
# print(model.conv1.weight.data)