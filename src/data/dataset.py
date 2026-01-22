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
        :param data_e: 环境特征 [N , channel_e]
        :param data_s: 系统特征 [N , channel_s]
        :param data_r: 工况特征 [N , channel_r]
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
        target = torch.from_numpy(self.targets[index + self.time_step]).float()
        # 返回整个窗口的时间索引序列，形状: [time_step]
        t_idx = torch.from_numpy(self.time_index[index : index + self.time_step]).float()
        return x_e , x_s , x_r , target , t_idx

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
        self.num_samples = len(targets) - time_step
    
    def __len__(self) : 
        return self.num_samples
    
    def __getitem__(self, index):
        # 返回时间窗口：[index : index + time_step]
        x = torch.from_numpy(self.data_x[index : index + self.time_step]).float()
        # 提取目标值（与 SubwayDataset 保持一致）
        target = torch.from_numpy(self.targets[index + self.time_step]).float()
        t_idx = torch.from_numpy(self.time_index[index : index + self.time_step]).float()

        return x, target, t_idx
    

class DatasetProvideWeek(Dataset) :
    """
    提供星期几信息的 Dataset，用于支持日周期 + 周周期的特征增强
    返回: x, target, t_index, day_of_week
    """
    def __init__(self, data_x, time_index, day_of_week, targets, time_step):
        """
        初始化数据集

        :param data_x: 特征数据 [N, num_features]
        :param time_index: 日时间索引 (0-287)，表示一天中的第几个5分钟点 [N]
        :param day_of_week: 周时间索引 (0-6)，0=周一, 6=周日 [N]
        :param targets: 目标负荷数据 [N]
        :param time_step: 时间窗口长度
        """
        self.time_step = time_step
        self.data_x = data_x
        self.targets = targets
        self.time_index = time_index
        self.day_of_week = day_of_week
        self.num_samples = len(targets) - time_step

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        返回一个时间窗口的数据

        :return: x [time_step, num_features]
                 target [1]
                 t_idx [time_step] - 日时间索引
                 day_of_week [time_step] - 周时间索引
        """
        # 特征窗口
        x = torch.from_numpy(self.data_x[index : index + self.time_step]).float()

        # 目标值（预测窗口后的下一个时刻）
        target = torch.from_numpy(self.targets[index + self.time_step]).float()

        # 日时间索引窗口
        t_idx = torch.from_numpy(self.time_index[index : index + self.time_step]).float()

        # 周时间索引窗口
        dow = torch.from_numpy(self.day_of_week[index : index + self.time_step]).float()

        return x, target, t_idx, dow


class CycleNetDataset(Dataset) : 
    """
     x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
    支持 CycleNet 模型的数据集构建，生成负荷的 [B , time_step , channel] , 以及 time_index : [B , ]
    """
    def __init__(self , load_data ,  time_index , time_step = 24 , cycle_len = 288):
        super().__init__()
        self.load_data = load_data
        self.time_index = time_index
        self.time_step = time_step
        self.cycle_len = cycle_len
        self.num_samples = len(self.load_data) - time_step

    def __len__(self) : 
        return self.num_samples

    def __getitem__(self, index):
        # 确保 load_data 是 1 维的
        if self.load_data.ndim == 2:
            data = self.load_data.reshape(-1)
        else:
            data = self.load_data

        # 1. 提取输入序列: [time_step] -> [time_step, 1]
        x_seq = data[index : index + self.time_step]
        x = torch.from_numpy(x_seq).float().unsqueeze(-1)

        # 2. 提取目标值: [1]
        target_val = data[index + self.time_step]
        target = torch.tensor([target_val], dtype=torch.float32)

        # 3. 计算周期索引（除以5转换为点索引）
        start_time_idx = self.time_index[index]
        cycle_index = int((start_time_idx / 5) % self.cycle_len)
        cycle_index = torch.tensor([cycle_index], dtype=torch.long)

        return x, target, cycle_index