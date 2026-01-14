# 特征嵌入模块，包含特征的 线性映射、层归一化、激活
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class FeatureEmbeddingBlock(nn.Module) :
    """
    特征嵌入模块，线性映射 + 层归一化 + 激活函数 ，确保喂的数据是已经 标准化 过的
    """
    def __init__(self , input_dim , output_dim ) :
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 线性映射
        self.linear = nn.Linear(input_dim , output_dim)
        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)
        # 激活函数
        self.activation = nn.Tanh()

    def forward(self , x) : 
        """
        x 的形状 : [batch_size , length  , 1]
        """
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x
    

class FeaturesCatBlock(nn.Module) : 
    """
    特征拼接模块 ，将 6 个特征进行拼接
    """
    def __init__(self , num_features = 6 , dim = 32) :
        super().__init__()
        # 将 6 个特征向量集合在一个 List 中
        self.embeddings = nn.ModuleList([
            FeatureEmbeddingBlock(1 , dim ) for i in range(num_features)
        ])

    def forward(self , x_list) : 
        """
        x_list : 包含 6 个向量的列表，每个向量的形状 [batch_size , length , 1]
        
        :param self: 说明
        :param x_list: 说明
        """
        embedded_features = []
        for i , block in enumerate(self.embeddings) : 
            embedded_features.append(block(x_list[i]))
        
        # 拼接
        features_cat = torch.cat(embedded_features , dim=-1)
        return features_cat
    
def create_sequences(data , time_step) : 
    X = []
    for i in range(len(data) - time_step) : 
        X.append(data[i : i + time_step , :])
    return np.array(X)

if __name__ == '__main__' : 
    # 1. 数据读取
    df = pd.read_csv("./data/raw/一个月数据总表.csv" , thousands=",")
    # 6 个特征列
    raws = ["passengers" , "structure_load" , "vent_load" , "temp" , "hum" , "equip_num"]
    raw_data = df[raws].values

    # print(raw_data.shape) # (8640 , 6)

    # 2. 标准化，把特征的量纲进行统一，数据的尺度要一致
    scaler = StandardScaler()
    scaler_data = scaler.fit_transform(raw_data)

    # 3. 构造滑动窗口   
    input_seq = create_sequences(scaler_data , time_step=10)
    input_tensor = torch.tensor(input_seq).float()
    # print(input_tensor.shape) # torch.Size([8630, 10, 6])

    # 4.切成 6 个 8630 * 10 * 1 的形式，这样每一块都代表一个特征
    x_list = [input_tensor[:, :, i : i + 1] for i in range(len(raws))]
    # print(len(x_list))  # 6

    # 5.把构建好的 x_list 进行特征的嵌入、融合
    cat_model = FeaturesCatBlock(6 , 32)
    features_cat = cat_model(x_list)
    # print(features_cat.shape)   # torch.Size([8630, 10, 192]) 1 -> 32 , 6 个 32 融合为 192 

    