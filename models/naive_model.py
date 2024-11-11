# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
from tqdm import tqdm  # 用于循环中显示进度条
import os  # 用于操作系统交互

# 用于读取和显示图像的库
from skimage.io import imread  # 用于读取图像
import matplotlib.pyplot as plt  # 用于绘图和结果可视化

# 用于创建训练集和验证集
from sklearn.model_selection import train_test_split  # 用于将数据集分为训练集和验证集
# 用于评估模型的性能
from sklearn.metrics import accuracy_score  # 用于计算分类模型的准确率

# 导入 PyTorch 库和模块
import torch  # PyTorch 主库
from torch.autograd import Variable  # 用于支持自动微分的张量
import torch.nn as nn  # 用于定义神经网络层
import torch.nn.functional as F  # 包含一些功能层，如激活函数
from torch.optim import *  # 优化算法 (例如 SGD, Adam)
import h5py  # 用于处理 .h5 文件（大数据集常用格式）


# 定义一个简单的 3D 卷积模型类 NaiveModel
class NaiveModel(nn.Module):
    def __init__(self, num_output):
        super(NaiveModel, self).__init__()

        # 使用 _conv_layer_set 方法定义两个卷积层
        self.conv_layer1 = self._conv_layer_set(1, 32)  # 从1个通道到32个通道的卷积层
        self.conv_layer2 = self._conv_layer_set(32, 16)  # 从32个通道到16个通道的卷积层

        # 全连接层（用于卷积和池化后的特征）
        self.fc1 = nn.Linear(864, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, num_output)  # 第二个全连接层（输出层）

        # 非线性激活和正则化层
        self.relu = nn.LeakyReLU()  # Leaky ReLU 激活
        self.batch = nn.BatchNorm1d(128)  # 批量归一化层，输入128个特征
        self.drop = nn.Dropout(p=0.15)  # Dropout 层，丢弃概率为15%

    # 辅助函数，用于定义一个 Conv3D 卷积块
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=0),  # Conv3D 卷积层
            nn.LeakyReLU(),  # Leaky ReLU 激活函数
            nn.MaxPool3d((1, 3, 3)),  # 最大池化层，池化窗口为 (1, 3, 3)
        )
        return conv_layer

    # 前向传播
    def forward(self, x):
        # 应用第一个卷积层
        x = self.conv_layer1(x)
        # 应用第二个卷积层
        x = self.conv_layer2(x)

        # 将卷积层输出展平，以便传递到全连接层
        x = x.view(x.size(0), -1)

        # 第一个全连接层，带有 ReLU 激活
        x = self.fc1(x)
        x = self.relu(x)

        # 可选：应用批量归一化和 Dropout（当前注释掉了）
        # x = self.batch(x)  # 批量归一化
        # x = self.drop(x)  # Dropout

        # 输出层（最终输出没有激活函数）
        x = self.fc2(x)

        return x
