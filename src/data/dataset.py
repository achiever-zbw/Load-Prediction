import pandas as pd
import torch
from torch.utils.data import TensorDataset , DataLoader

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