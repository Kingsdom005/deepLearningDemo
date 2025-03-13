import torch
from dataLoader import generater, loadData
from pathlib import Path

print(torch.cuda.is_available())
# 检查是否有可用的 CUDA 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# loss = (WX + b - Y)^2
def loss(b, w, points):
    x = points[0]
    y = points[1]
    totalError = torch.sum((w * x + b - y) ** 2)
    return totalError / float(len(x))

def step_gradient(b_current, w_current, points, learningRate):
    x = points[0]
    y = points[1]
    N = float(len(x))
    b_gradient = - (2 / N) * torch.sum(w_current * x + b_current - y)
    w_gradient = - (2 / N) * torch.sum(x * (w_current * x + b_current - y))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

def gradient_descent(points, starting_b, starting_w, learning_rate, num_iterations):
    b = torch.tensor(starting_b, dtype=torch.float32, device=device, requires_grad=False)
    w = torch.tensor(starting_w, dtype=torch.float32, device=device, requires_grad=False)
    points = torch.tensor(points, dtype=torch.float32, device=device)
    
    for i in range(num_iterations):
        b, w = step_gradient(b, w, points, learning_rate)
    
    return [b.item(), w.item()]

if __name__ == '__main__':
    # 获取当前文件的路径对象
    current_file_path = Path(__file__).resolve()

    # 获取当前文件所在的目录路径
    current_dir_path = str(current_file_path.parent) + "/data.csv"
    generater(path=current_dir_path)
    
    # 加载数据
    data = loadData(path=current_dir_path)
    
    # 训练模型配置
    learning_rate = 0.000000001
    initial_b = 0
    initial_w = 0
    num_iterations = 10000
    
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w, 
                  loss(torch.tensor(initial_b, dtype=torch.float32, device=device),
                       torch.tensor(initial_w, dtype=torch.float32, device=device),torch.tensor(data, dtype=torch.float32, device=device))))
    
    print("Running...")
    [b, w] = gradient_descent(data, initial_b, initial_w, learning_rate, num_iterations)
    
    print("After {0} iterations b = {1}, w = {2}, error = {3}".format(
        num_iterations, b, w, loss(torch.tensor(b, dtype=torch.float32, device=device),
        torch.tensor(w, dtype=torch.float32, device=device),
        torch.tensor(data, dtype=torch.float32, device=device))))