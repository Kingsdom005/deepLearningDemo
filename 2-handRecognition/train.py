import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from utils import *

batch_size = 512

# step1: load dataset
# 加载训练数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

# 加载测试数据集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

# x, y = next(iter(train_loader))
# print(x.shape, y.shape, x.min(), x.max())
# plot_image(x, y , "image sample")

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 前向传播函数
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1 + b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(xw2 + b2)
        x = F.relu(self.fc2(x))
        # h3 = xw3 + b3
        x = self.fc3(x)
        return x
    
net = Net()    
# [w1, b1, w2, b2, w3, b3]
# 初始化优化器
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []

# 训练模型
for epoch in range(3):
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # x: [b, 1, 28, 28] y:[512]
        # [b, 1, 28, 28] => [b, feature]
        x = x.view(x.size(0), 28*28)
        out = net(x)
        y_onehot = one_hot(y,10)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)
        
        optimizer.zero_grad()
        loss.backward()
        # w' = w - lr * grad
        optimizer.step()

        train_loss.append(loss.item())

        # if batch_idx % 10 == 0:
        #     print(epoch, batch_idx, loss.item())
            
# plot_curve(train_loss)

# 测试模型
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
    
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("test acc:", acc)

# 可视化测试结果
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, "test result")