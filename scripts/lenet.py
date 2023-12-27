import torch
from torch.autograd import Variable
from torch.functional import F
from torchsummary import summary

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = torch.nn.Linear(120, 84)
        self.fc2 = torch.nn.Linear(84, 10)
        self.avgpool = torch.nn.AvgPool2d(2)

    def forward(self, x):
        x = self.avgpool(F.relu(self.conv1(x)))
        x = self.avgpool(F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Model()
summary(model, (1, 28, 28))

image = Variable(torch.ones(1, 1, 28, 28), requires_grad=True)
output = model(image)
print(output)
