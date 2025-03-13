import numpy as np
from dataLoader import generater,loadData
from pathlib import Path

# loss = (WX+b-Y)^2
def loss(b, w , points):
    totalError = 0
    for i in range(len(points[0])):
        x = points[0, i]
        y = points[1, i]
        # print("x=",x, "y=",y)
        totalError += (w*x + b - y)**2
    return totalError/ float(len(points[0]))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points[0]))
    for i in range(len(points[0])):
        x = points[0, i]
        y = points[1, i]
        b_gradient += -(2/N) * (w_current*x + b_current - y)
        w_gradient += -(2/N) * x * (w_current*x + b_current - y)
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

def gradient_descent(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


if __name__ == '__main__': 


    # 获取当前文件的路径对象
    current_file_path = Path(__file__).resolve()

    # 获取当前文件所在的目录路径    
    current_dir_path = str(current_file_path.parent) + "/data.csv"
    generater(path=current_dir_path)
    
    # 加载数据
    data = loadData(path=current_dir_path)
    
    # 训练模型配置
    learning_rate = 0.00000001
    initial_b = 0
    initial_w = 0
    num_iterations = 10000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}".format(initial_b, initial_w, loss(initial_b, initial_w, data)))
    print("Running...")
    [b, w] = gradient_descent(data, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".format(num_iterations, b, w, loss(b, w, data)))
    
    
