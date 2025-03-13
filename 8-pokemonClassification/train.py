import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from utils import *
from pokemon import Pokemon  # 导入 Pokemon 数据集类
from resnet import ResNet18  # 导入 ResNet18 网络
import os
from tqdm import tqdm  # 导入 tqdm 库

# 设置当前工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 超参数
batch_size = 32  # 适当减小 batch_size
resize = 64  # 调整图像大小
num_epochs = 10  # 训练轮数

# step1: 加载数据集
train_db = Pokemon(root="pokemon", resize=resize, mode='train')
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)

val_db = Pokemon(root="pokemon", resize=resize, mode='val')
val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False)

test_db = Pokemon(root="pokemon", resize=resize, mode='test')
test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型、优化器和损失函数
net = ResNet18().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 使用 Adam 优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
train_loss = []
val_loss = []
best_val_loss = float('inf')  # 用于保存验证集上的最小损失
best_model_path = "./model/best_model.pth"  # 最优模型保存路径

# 训练模型
for epoch in range(num_epochs):
    net.train()  # 设置模型为训练模式
    epoch_loss = 0

    # 使用 tqdm 创建进度条
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as tepoch:
        for batch_idx, (x, y) in enumerate(tepoch):
            x, y = x.to(device), y.to(device)  # 将数据移动到设备

            # 前向传播
            out = net(x)
            loss = criterion(out, y)  # 计算损失

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 更新进度条描述
            tepoch.set_postfix(loss=loss.item())

    # 记录每个 epoch 的平均损失
    train_loss.append(epoch_loss / len(train_loader))

    # 验证模型
    net.eval()  # 设置模型为评估模式
    val_epoch_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = criterion(out, y)
            val_epoch_loss += loss.item()

    val_epoch_loss /= len(val_loader)
    val_loss.append(val_epoch_loss)

    # 如果验证集上的损失更小，则保存模型
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(net.state_dict(), best_model_path)
        print(f"New best model saved with val_loss: {best_val_loss:.4f}")

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_epoch_loss:.4f}")

# 可视化训练损失和验证损失
plot_curve(train_loss, "Training Loss Curve")
plot_curve(val_loss, "Validation Loss Curve")

# 测试模型
net.eval()  # 设置模型为评估模式
total_correct = 0
total_samples = 0

with torch.no_grad():  # 禁用梯度计算
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        pred = out.argmax(dim=1)  # 获取预测结果
        correct = pred.eq(y).sum().item()  # 计算正确预测的数量
        total_correct += correct
        total_samples += y.size(0)

# 计算测试准确率
test_accuracy = total_correct / total_samples
print(f"Test Accuracy: {test_accuracy:.4f}")

# 加载最优模型进行测试
net.load_state_dict(torch.load(best_model_path))
net.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().item()
        total_correct += correct
        total_samples += y.size(0)

# 计算最优模型的测试准确率
best_test_accuracy = total_correct / total_samples
print(f"Best Model Test Accuracy: {best_test_accuracy:.4f}")