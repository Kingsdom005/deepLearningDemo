# 任务简介

## 1. 线性方程预测

使用噪声模拟生成原始数据点，手动实现梯度下降算法，预测线性方程。

## 2. 手写数字识别

构建简单网络，使用mnist数据集实现手写数字识别。

## 3. 优化器与可视化

使用Adam优化器，利用mpl_toolkits.mplot3d中的Axes3D实现3D可视化展示。

<img src=".\fig\Himmelblau.png" alt="Himmelblau" style="zoom:72%;" />

## 4. 图像分类

使用Lenet-5网络，实现对cifar10数据集的预测。

## 5. 残差网络

使用Resnet-18网络，实现对cifar10数据集的预测。

## 6. 循环神经网络(RNN)

使用RNN实现时间序列预测。

<img src=".\fig\Figure_1.png" alt="Figure_1" style="zoom:72%;" />

## 7. 长短期记忆循环神经网络（LSTM）

使用LSTM实现时间序列预测。

<img src=".\fig\Figure_2.png" alt="Figure_2" style="zoom:72%;" />

## 8. 综合项目

任务：图片分类

数据集：宝可梦5类图片（自定义数据集，图片size不统一）

项目结构：

```
project-name/
│
├── model/ 						# Store best model
│   └── best_model.pth
│
├── pokemon/					# Dataset
│   ├── bulbasaur
│   │   └── *.jpg/png/jpeg/...
│   ├── charmander/
│   │ 	└── *.jpg/png/jpeg/...
│   ├── mewtwo/
│   │ 	└── *.jpg/png/jpeg/...
│   ├── pikachu/
│   │	└── *.jpg/png/jpeg/...
│   ├── squirtle/
│   │	└── *.jpg/png/jpeg/...
│	└──	images.csv				# Retrieval table for images
│
├── pokemon.py 					# Dataset loader and generate retrieval table
├── resnet.py					# Resnet-18 model
├── test.py						# Use best model to predict
├── train.py					# Train and save model
└── utils.py					# Tools
```

