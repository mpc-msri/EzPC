from secrets import randbelow
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import random
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=20)

batch_size = 20
epochs = 1 #n_iters / (len(train_dataset) / batch_size)
input_dim = 784
output_dim = 10
lr_rate = 0.01

train_dataset = dsets.MNIST(root='./datatorch', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./datatorch', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 500)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(500, output_dim)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        self.inp = x
        self.x = self.relu1(self.fc1(x))
        self.y = self.relu2(self.fc2(self.x))
        # self.y.requires_grad = True
        return self.y

model = Model(input_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

for i in range(500):
    for j in range(784):
        model.fc1.weight.data[i][j] = 0.5

for i in range(500):
    model.fc1.bias.data[i] = 0.5

for i in range(10):
    for j in range(500):
        model.fc2.weight.data[i][j] = 0.5

for i in range(10):
    model.fc2.bias.data[i] = 0.5


# image = torch.ones(784)
# labels = torch.zeros(10)
# labels[0] = 1.0

# outputs = model(image)
# print(outputs.size())
# print(model.fc2.weight.data.size())
# loss = criterion(outputs, labels)
# loss.backward()
# optimizer.step()
# # print(model.fc1.weight.grad)
# # print(model.fc1.weight.data)
# print(model(image))

# exit(0)

iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        # print(labels)
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        # print(model.x.sum())
        # print(model.fc2.weight.size())
        # print(model.fc2.weight.t())
        # print(torch.matmul(model.x, model.fc2.weight.t()))
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # print(loss)
        optimizer.step()

        iter+=1
        if i == 10:
            # print(outputs)
            break
    # for i in range(784):
    #     print(1 if model.inp[0].data[i] > 0.0 else 0, end='')
    #     if i % 28 == 27:
    #         print()
    # for i in range(784):
    #     print(1 if model.inp[1].data[i] > 0.0 else 0, end='')
    #     if i % 28 == 27:
            # print()
    # print(model.x)
    # print((model.fc2.weight.grad.data))
    print((model.fc2.bias.data))
    # for i in range(10):
    #     print(model.fc2.weight.grad.data[i][0])
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total+= labels.size(0)
        # for gpu, bring the predicted and labels back to cpu fro python operations to work
        correct+= (predicted == labels).sum()
    accuracy = 100 * correct/total
    print("Epoch: {}. Accuracy: {}.".format(epoch, accuracy))
    # for u in range(10):
    #     print(model.fc2.weight.grad.data[u][0])