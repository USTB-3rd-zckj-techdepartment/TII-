
## 输入数据

- **数据类型**: 点云数据（.ply格式）
- **数据来源**: 多个场景(scenario31-34)的LiDAR扫描数据
- **数据路径**:
    
    ```
    /UserProject/TII_DEEPSENSE6G_Multimodal_learning/DeepSense6G_TII-main/Dataset/
    ├── Adaptation_dataset_multi_modal/
    ├── Multi_Modal/
    └── Multi_Modal_Test/
    
    ```
    
- **数据要求**: 每个场景的点云数据需满足最小点数要求
    - scenario31: 16400点
    - scenario32: 18000点
    - scenario33: 18000点
    - scenario34: 18600点

## 处理方法

### 1. 背景提取

1. **初始化**:
    - 选择点数量超过阈值的点云作为初始背景模型
    - 使用KD树进行最近邻点搜索
2. **点云配准与融合**:
    - 对每个点执行最近邻搜索
    - 计算点对距离：`distance = sqrt(dx^2 + dy^2)`
    - 计算动态过滤阈值：
        
        ```python
        filter_distance = filter_distance_min +
                         (filter_distance_max - filter_distance_min) *
                         (point_distance/lidar_distance_cst)^4
        
        ```
        
    - 当距离小于阈值时，取两点平均值作为背景点

### 2. 前景提取

1. **加载背景模型**:
    - 读取对应场景的背景点云
2. **动态物体识别**:
    - 对每个点云文件中的点：
        - 在背景模型中查找最近邻点
        - 计算点对距离
        - 应用动态阈值过滤
        - 距离大于阈值的点被认为是前景（动态物体）
3. **关键参数**:
    
    ```python
    filter_distance_min = 0.3  # 最小过滤距离
    filter_distance_max = 5    # 最大过滤距离
    lidar_distance_min = 40   # LiDAR最小检测距离
    lidar_distance_cst = 30   # 距离计算常数
    
    ```
    

## 输出数据

- **文件格式**: PLY格式点云文件
- **输出目录结构**:
    
    ```
    /preprocess_lidar/
    ├── Background/                 # 存放背景点云
    │   └── scenario{31-34}_background.ply
    ├── Adaptation_dataset_multi_modal/
    ├── Multi_Modal/
    └── Multi_Modal_Test/          # 存放处理后的点云
    
    ```
    
- **数据特点**:
    - 背景文件：提取的静态场景点云
    - 处理后文件：仅包含动态物体的点云数据
    - ASCII格式存储，便于读取和验证

## LiDAR点云数据增强方法

## 输入数据

- **数据类型**: 点云数据（.ply格式）
- **数据来源**: 3个场景的LiDAR扫描数据（scenario31-33）
- **数据路径**:
    
    ```
    /Dataset/Adaptation_dataset_multi_modal/
    ├── scenario31/unit1/lidar_data/
    ├── scenario32/unit1/lidar_data/
    └── scenario33/unit1/lidar_data/
    
    ```
    

## 数据增强方法

### 1. 下采样增强

- **方法**: 随机下采样（Random Downsampling）
- **参数**: 保留90%的原始点云数据（采样率0.9）
- **实现代码**:
    
    ```python
    downpcd1 = pcd.random_down_sample(0.9)
    
    ```
    
- **目的**:
    - 模拟传感器分辨率变化
    - 增加数据多样性
    - 提高模型鲁棒性

### 2. 噪声增强

- **方法**: 添加均匀分布噪声
- **参数**:
    - 噪声范围: ±0.4
    - 三个维度(x,y,z)独立添加噪声
- **实现代码**:
    
    ```python
    for point_item in pcd.points:
        point_item[0] += random.uniform(-noise_range, noise_range)
        point_item[1] += random.uniform(-noise_range, noise_range)
        point_item[2] += random.uniform(-noise_range, noise_range)
    
    ```
    
- **目的**:
    - 模拟传感器测量误差
    - 增强模型对噪声的适应能力
    - 提高模型泛化能力

## 输出数据

- **文件命名规则**:
    - 下采样增强: `{原文件名}_1.ply`
    - 噪声增强: `{原文件名}_2.ply`
- **输出目录结构**:
    
    ```
    /lidar_data_aug/
    ├── scenario31/unit1/lidar_data_aug/
    ├── scenario32/unit1/lidar_data_aug/
    └── scenario33/unit1/lidar_data_aug/
    
    ```
    
- **数据格式**: ASCII格式的PLY文件

### 一点补丁：论文中的FoV在data2_seq中实现

FOV的实现主要在`lidar_to_histogram_features`函数中,通过以下方式实现:

```python
if custom_FoV:
    if 'scenario31' in addr:
        xbins = np.linspace(-70, 0, 257)
        ybins = np.linspace(-25, 14, 257)
    elif 'scenario32' in addr:
        xbins = np.linspace(-60, 0, 257)
        ybins = np.linspace(-40, 5.5, 257)
    elif 'scenario33' in addr:
        xbins = np.linspace(-50, 0, 257)
        ybins = np.linspace(-12, 7, 257)
    elif 'scenario34' in addr:
        xbins = np.linspace(-50, 0, 257)
        ybins = np.linspace(-20, 10, 257)

```

- 对不同场景设置不同的x轴和y轴范围
- 使用`np.linspace`创建均匀分布的点来定义视场范围
- 通过这种方式为每个场景定制化视场角范围

## 拓展思考

## **深度学习与注意力机制在3D点云数据处理中的应用**

## **方案**

结合PointNet等深度学习模型与注意力机制，可以显著提高对局部特征的提取能力。这种方法能够更有效地识别和分割动态物体，提升模型在复杂场景中的表现。通过引入注意力机制，网络能够聚焦于有效数据，忽略无关信息，从而提升整体识别精度。

## **关键研究与论文**

以下是一些相关的研究论文，探讨了深度学习与注意力机制在3D点云处理中的应用：

- **FatNet: A Feature-attentive Network for 3D Point Cloud Processing**该论文提出了一种新的特征关注神经网络层（FAT层），结合全局点特征和局部边缘特征，以改善点云分析的效果。该架构在ModelNet40数据集上实现了最先进的分类结果。[阅读论文](https://arxiv.org/abs/2104.03427)
- **Point Cloud Semantic Segmentation Network Based on Adaptive Convolution and Attention Mechanism**本文提出了一种基于自适应卷积和注意力机制的点云语义分割网络，旨在增强点云数据的空间表达能力。实验结果表明，该方法在S3DIS数据集上表现优异。[阅读论文](https://www.semanticscholar.org/paper/4c5ba7008dda2adb91aded1576d4d2301eba736f)
- **SCA-PointNet++: Fusion Spatial-Channel Attention for Point Cloud Extraction Network of Buildings in Outdoor Scenes**本研究提出了一种融合空间-通道注意力机制的PointNet++网络，用于大规模户外场景中建筑物的提取。通过改进特征编码和注意力机制，该方法在SensatUrban数据集上表现良好。[阅读论文](https://www.semanticscholar.org/paper/13ef76b90576d9b13652b420f3b7a4b75e77b370)
- **DetailPoint: detailed feature learning on point clouds with attention mechanism**本文介绍了一种新的方法，通过注意力机制进行详细特征学习，以提升点云数据的处理效果。该方法展示了在复杂场景下提取细节特征的潜力。[阅读论文](https://www.semanticscholar.org/paper/cefd6746cb27c36f91be3b43eea92876e24fecf5)

##
