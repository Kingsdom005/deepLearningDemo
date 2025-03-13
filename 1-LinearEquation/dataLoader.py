import numpy as np

# y = 3.486 * x + 0.6845 + noise
def func(x):
    noise = np.random.normal(0, 0.1)  # 添加噪声
    return 3.486 * x + 0.6845 #+ noise

def generater(path):
    with open(path, 'w') as f:
        count = 20000  # 生成 20 个数据点
        x = np.random.uniform(-1000, 1000, count)  # 将 x 的范围缩小到 [-10, 10]
        for i in x:
            f.write(str(i) + "," + str(func(i)) + "\n")

def loadData(path):
    x = []
    y = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            x.append(float(line[0]))
            y.append(float(line[1]))
    # 将 x 和 y 组合成一个二维数组
    points = np.vstack((x, y))
    return points