# TII_multimodal_explain

# TransFuser毫米波波束预测系统概述

## 一、数据处理方法

数据来源于deepsense6G官网提供。代码对图像，雷达，激光雷达，GPS定位这些多模态数据进行处理。

### 数据处理内容：
* 图像：用的是亮度调整，饱和度调整等增强方法，具体细节在 1 图像数据增强技术报告 里面。

* 雷达：主要是对雷达的npy数据进行傅里叶变换，提取环境中的移动物体的距离、角度和速度。细节在 2 雷达数据处理研究 里面。

* 激光雷达数据：主要进行了背景过滤，下采样增强等方法，具体细节在 3 LiDAR点云数据处理方法详解 里面。

## 二、模型构建方法

### 主要框架：
A. 多模态特征编码器 
B. 多层次特征融合Transformer 使用的传统的多头注意力机制进行融合 
C. 特征变换网络，实现预测

### 具体实现：
* A. 对预处理好的数据进行特征编码，主要用的是resnet（针对图像，雷达，激光雷达数据）和MLP线性特征（针对GPS数据）。细节在4.1 TransFuser中的图像编码方法和4.2 TransFuser中的多传感器数据编码方法 中概括。

* B. 传统的多头注意力机制，进行特征融合。

* C. 将融合好的512维特征，转换为64维度的毫米波波束强度预测。具体在5 TransFuser波束强度预测的训练实现详解 中。
