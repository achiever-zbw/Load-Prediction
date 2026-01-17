import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset , DataLoader , Dataset
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from .channel_embedding import FeaturesCatBlock , create_sequences

def get_dataloader(features , targets , batch_size , shuffle = True) : 
    """
    将 numpy 转化为 torch 张量并处理
    
    :param features: 构建的 X(np.array)
    :param targets:  构建的 y(np.array)
    :param batch_size: 批次
    :param shuffle: 训练集不打乱数据
    """

    # 转化为张量
    X_tensor = torch.tensor(features , dtype = torch.float32)
    y_tensor = torch.tensor(targets , dtype=torch.float32).unsqueeze(1)
    # 封装为 dataset
    dataset = TensorDataset(X_tensor , y_tensor)
    # 创建 dataloader
    loader = DataLoader(dataset , batch_size , shuffle=True)

    return loader

# 划分训练集与验证集
def data_split(datas , train_size , val_size , test_size) : 
    train_data = datas[:train_size]
    val_data = datas[train_size : train_size + val_size]
    test_data = datas[train_size + val_size : ]
    return train_data , val_data , test_data



class SubwayDataset(Dataset) : 
    """
    构造数据集，形状为 [10 , 6]  , [1, ]  , 表示，每一个窗口的长度为 10 ，6 个特征 ; 以及 1 个对应的预测值
    """
    def __init__(self , data_e , data_s , data_r , time_index , targets , time_step):
        """
        初始化数据集
        
        :param self: 说明
        :param data_e: 环境特征 [N , 3]
        :param data_s: 系统特征 [N , 3]
        :param data_r: 工况特征 [N , 4]
        :param time_index: 时间索引 , [N , ] -> 用于周期增强
        :param targets: 目标负荷数据 [N , 1]
        :param time_step: 窗口长度
        """
        
        self.time_step = time_step
        self.data_e = data_e
        self.data_s = data_s
        self.data_r = data_r
        self.time_index = time_index
        self.targets = targets

        # 可用的样本数
        self.num_samples = len(targets) - time_step

    def __len__(self) : 
        return self.num_samples
    
    # 每个 index 的 dataset 代表一个窗口，由 dataloader 打包后传入 SubwayLoadModel 进行特征的融合
    def __getitem__(self, index):
        """
        根据索引构建窗口
        """
        x_e = torch.from_numpy(self.data_e[index : index + self.time_step]).float()
        x_s = torch.from_numpy(self.data_s[index : index + self.time_step]).float()
        x_r = torch.from_numpy(self.data_r[index : index + self.time_step]).float()

        # 提取目标值
        target = torch.from_numpy(np.array(self.targets[index + self.time_step])).float()
        # 返回整个窗口的时间索引序列，形状: [time_step]
        t_idx = torch.from_numpy(self.time_index[index : index + self.time_step]).float()

        return x_e , x_s , x_r , t_idx , target

class SubwayLoadModel(nn.Module) : 
    """
    通道嵌入与特征融合模型，总流程 : 
    - 1. 共三个特征，E(环境) , S(系统) , R(工况) , 对应 3 3 4 个子特征，即通道
    - 2. 首先进行每种输入特征（对应多个通道）通过线性映射进行通道嵌入
        - 首先读取数据，按照三个分类进行读取，大小分别为 [8640 , 3] [8640 , 3] [8640 , 4]
        - 进行标准化，将三组特征进行标准化，统一量纲 ，通过 SubwayDataset 里的 StandardScaler() 实现
        - 构建窗口，以 10 为步长，对三个特征分别构建窗口，得到 [8630 , 10 , 3] [8630 , 10 , 3] [8630 , 10 , 4] 
        - 三组数据进行批次 32 打包，构建出 [32 , 10 , 3] [32 , 10 , 3] [32 , 10 , 4]
        - 将 3 个特征进行各自的线性映射，实现通道的嵌入 。得到 [32 , 10 , 64] [32 , 10 , 64] [32 , 10 , 64]
        - 嵌入后进行层归一化和激活函数处理
        - 将 嵌入特征进行拼接，生成统一的特征向量 Z(concat) ，形状为 [32 , 10 , 192]
    
    """

    def __init__(self , dim = 64 , time_step = 24) :
        super().__init__()
        self.fusion_model = FeaturesCatBlock(dim)

    def forward(self , x_e , x_s , x_r) : 
        """
        x 是从 DataLoader 传入的数据，[batch_size , time_step , num_features] , e s r 分别为 3 3 4 
        
        :param self: 说明
        :param x: 说明
        """
        # 特征嵌入与融合,执行了 linear -> LayerNorm -> Tanh -> Cat
        fusion_data = self.fusion_model(x_e , x_s , x_r)
        return fusion_data


class NoneChannelAttnDataset(Dataset) : 
    """
    不需要特征分类，直接使用 10 个特征
    """
    def __init__(self , data_x, time_index , targets , time_step) : 
        self.time_step = time_step
        self.data_x = data_x
        self.targets = targets
        self.time_index = time_index
        self.num_samples = len(data_x) - time_step
    
    def __len__(self) : 
        return self.num_samples
    
    def __getitem__(self, index):
        # 返回时间窗口：[index : index + time_step]
        x = torch.from_numpy(self.data_x[index : index + self.time_step]).float()
        # 提取目标值（与 SubwayDataset 保持一致）
        target = torch.from_numpy(np.array(self.targets[index + self.time_step])).float()
        t_idx = torch.tensor([self.time_index[index + self.time_step]]).float()

        return x, target, t_idx