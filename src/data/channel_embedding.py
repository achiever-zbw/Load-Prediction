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
        self.activation = nn.ReLU()

    def forward(self , x) : 
        """
        x 的形状 : [batch_size , length , num_features]
        """
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x
    

# class FeaturesCatBlock(nn.Module) : 
#     """
#     特征拼接模块 ，将 6 个特征进行拼接
#     """
#     def __init__(self , num_features = 6 , dim = 32) :
#         super().__init__()
#         # 将 6 个特征向量集合在一个 List 中
#         self.embeddings = nn.ModuleList([
#             FeatureEmbeddingBlock(1 , dim ) for i in range(num_features)
#         ])

#     def forward(self , x_list) : 
#         """
#         x_list : 包含 6 个向量的列表，每个向量的形状 [batch_size , length , 1]
        
#         :param self: 说明
#         :param x_list: 说明
#         """
#         embedded_features = []
#         for i , block in enumerate(self.embeddings) : 
#             embedded_features.append(block(x_list[i]))
        
#         # 拼接
#         features_cat = torch.cat(embedded_features , dim=-1)
#         return features_cat

class FeaturesCatBlock(nn.Module) :
    """
    将 3 个特征首先进行通道嵌入，然后进行拼接
    """
    def __init__(self , dim = 64) : 
        super().__init__()
        self.embed_e = FeatureEmbeddingBlock(3 , dim)   # E 环境特征，对应 温度、湿度、风速
        self.embed_s = FeatureEmbeddingBlock(3 , dim)   # S 系统特征，对应 冷机功率、冷却水温度、冷冻水温度
        self.embed_r = FeatureEmbeddingBlock(4 , dim)   # R 工况特征，对应 客流量、风机频率、水泵频率、空调设备启停状态
    
    def forward(self , x_e , x_s , x_r) :
        """
        x_e : [batch_size , length , 3]
        x_s : [batch_size , length , 3]
        x_r : [batch_size , length , 4]
        """
        # 1. 特征内部进行通道融合
        ze = self.embed_e(x_e)
        zs = self.embed_s(x_s)
        zr = self.embed_r(x_r)

        # 2. 三个特征拼接
        feature_cat = torch.cat([ze , zs , zr] , dim = -1)

        return feature_cat

def create_sequences(data , time_step) : 
    X = []
    for i in range(len(data) - time_step) : 
        X.append(data[i : i + time_step , :])
    return np.array(X)

if __name__ == '__main__' : 
    # 1. 数据读取
    df = pd.read_csv("./data/processed/一个月数据总表_10特征.csv" , thousands=",")

    raws_e = ["temp","hum"	,"wind"]
    raw_e_data = df[raws_e].values
    raws_s = ["power",	"cw_temp", "chw_temp"]
    raw_s_data = df[raws_s].values
    raws_r = ["pax"	, "status"	 , "fan_freq"	, "pump_freq"]
    raw_r_data = df[raws_r].values
    
    scaler_e = StandardScaler()
    scaler_s = StandardScaler()
    scaler_r = StandardScaler()

    scaler_e_data = scaler_e.fit_transform(raw_e_data)
    scaler_s_data = scaler_s.fit_transform(raw_s_data)
    scaler_r_data = scaler_r.fit_transform(raw_r_data)

    # 2. 创建时序序列 
    time_step = 10
    seq_e = create_sequences(scaler_e_data, time_step) 
    seq_s = create_sequences(scaler_s_data, time_step) 
    seq_r = create_sequences(scaler_r_data, time_step) 

    # 3. 转换为 PyTorch 张量
    tensor_e = torch.FloatTensor(seq_e)
    tensor_s = torch.FloatTensor(seq_s)
    tensor_r = torch.FloatTensor(seq_r)

    # 4. 实例化模型并测试
    dim_size = 64
    model = FeaturesCatBlock(dim=dim_size)
    
    # 执行前向传播

    output = model(tensor_e, tensor_s, tensor_r)

    # 5. 打印形状验证结果
    print("-" * 50)
    print(f"输入 E 形状: {tensor_e.shape}")  # [Batch, 10, 3]
    print(f"输入 S 形状: {tensor_s.shape}")  # [Batch, 10, 3]
    print(f"输入 R 形状: {tensor_r.shape}")  # [Batch, 10, 4]
    print("-" * 50)
    print(f"嵌入维度 (dim): {dim_size}")
    print(f"预期输出特征维度: {dim_size} * 3 = {dim_size * 3}")
    print(f"模型输出实际形状: {output.shape}")
    print("-" * 50)

    # 形状检查逻辑
    expected_shape = (tensor_e.shape[0], time_step, dim_size * 3)
    if output.shape == expected_shape:
        print("验证通过：拼接后的形状符合预期！")
    else:
        print("验证失败：形状不匹配。")