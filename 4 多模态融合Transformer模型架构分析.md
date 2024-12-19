
## 1. 总体架构

论文提出的TransFuser是一个基于Transformer的多模态融合网络，模型包含以下主要组件：

1. 多模态特征编码器
2. 多层次特征融合Transformer 使用的传统的多头注意力机制进行融合
3. 特征变换网络

## 2. 核心组件

### 2.1 图像编码器 (ImageCNN)

```python
class ImageCNN(nn.Module):
    def __init__(self, c_dim, normalize=True):
        self.features = models.resnet34(pretrained=True)

```

- 基于预训练的ResNet34网络
- 支持ImageNet标准化处理
- 移除原始分类头，只保留特征提取部分

### 2.2 激光雷达编码器 (LidarEncoder)

```python
class LidarEncoder(nn.Module):
    def __init__(self, num_classes=512, in_channels=2):
        self._model = models.resnet18(pretrained=True)

```

- 使用修改版ResNet18
- 支持不同输入通道配置
- 专门适配点云数据处理

### 2.3 自注意力机制 (SelfAttention)

- 实现标准的多头注意力机制
- 主要组件：
    - 查询(Q)、键(K)、值(V)的线性变换
    - 注意力分数计算和缩放
    - Dropout正则化
    - 输出投影

### 2.4 Transformer块 (Block)

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        self.attn = SelfAttention(...)
        self.mlp = nn.Sequential(...)

```

- 标准Transformer结构
- 包含自注意力层和前馈网络
- 使用LayerNorm和残差连接
- 可配置的dropout率

### 2.5 GPT特征融合器

```python
class GPT(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, n_layer, ...):
        self.pos_emb = nn.Parameter(...)
        self.blocks = nn.Sequential(...)

```

- 基于GPT架构的特征融合模块
- 可学习的位置编码
- 多层Transformer块堆叠
- 支持多模态输入序列处理

## 3. 多层次特征融合策略

### 3.1 四阶段特征提取

每个阶段的特征维度逐步增加：

1. 第一阶段: 64维
2. 第二阶段: 128维
3. 第三阶段: 256维
4. 第四阶段: 512维

### 3.2 残差连接

```python
# 特征融合和残差连接示例
image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8)
image_features = image_features + image_features_layer1

```

- 每个阶段都包含残差连接
- 使用双线性插值进行上采样
- 融合不同尺度的特征

### 3.3 GPS特征处理

```python
self.vel_emb1 = nn.Linear(2, 64)
self.vel_emb2 = nn.Linear(64, 128)
self.vel_emb3 = nn.Linear(128, 256)
self.vel_emb4 = nn.Linear(256, 512)

```

- 逐层增加GPS特征维度
- 与其他模态特征保持一致的维度

## 4. 特征融合流程

### 4.1 输入处理

- 图像序列标准化
- 点云数据转换
- 雷达数据处理
- GPS数据嵌入

### 4.2 特征提取

每个模态通过各自的编码器：

- 图像 → ResNet34
- 激光雷达 → 修改的ResNet18
- 雷达 → 修改的ResNet18

### 4.3 多层融合

每一层都执行：

1. 特征提取
2. 特征池化
3. Transformer融合
4. 上采样和残差连接保全信息，保证融合质量。

### 4.4 最终融合

```python
fused_features = torch.cat([image_features, lidar_features,
                           radar_features, gps_features], dim=1)
fused_features = torch.sum(fused_features, dim=1)

```

- 拼接所有模态特征
- 序列维度求和
- 生成最终的融合特征

## 5. 特征变换网络

### 5.1 维度压缩

```python
self.join = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 64)
)

```

- 三层全连接网络
- 逐步降低特征维度
- 保持关键信息的同时实现压缩

### 

[4+ ResNet图像特征提取一般方法](https://www.notion.so/4-ResNet-16142a179edf8082b8e1dd030b9f3e9b?pvs=21)

[4.2 TransFuser中的多传感器数据编码方法](https://www.notion.so/4-2-TransFuser-16142a179edf80e08e71d2333ab4a468?pvs=21)
