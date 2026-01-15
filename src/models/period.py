import torch
import torch.nn as nn
import math
from .lstm import LSTMBlock

class PeriodEnhanceBlock(nn.Module) :
    """
    周期特征增强模块，以 24 小时为一个周期 24 * 60 / 5 = 288
    """
    def __init__(self , hidden_dim = 64) : 
        super().__init__()
        self.steps = 288
        self.mlp = nn.Sequential(
            nn.Linear(2 , 16) ,     # 输入是 [sin , cos ] 二维
            nn.ReLU() , 
            nn.Linear(16 , hidden_dim) ,    # 维度为 64 ，与 lstm 的输出对齐
        )
    
    def forward(self , h2 , t_index) : 
        """
        h2 : LSTM 的输出隐藏状态 [batch_size , hidden_dim]
        t_index : 时间步 取值为 0 ~ 287 , 大小为 [batch_size , 1]
        - 要先计算出 SinCos(t_index) , 变成 [batch_size , 2]
        - 通过 MLP 线性映射到 [batch_size , hidden_dim] , 与 h2 同维度，进行拼接
        """
        alpha = 2.0 * math.pi * t_index / self.steps
        sin_t = torch.sin(alpha)
        cos_t = torch.cos(alpha)
        # SinCos(t) : [sin_t , cos_t] 拼接一下
        sin_cos_t = torch.cat([sin_t , cos_t] , dim = -1)
        print(f"SinCos(t) : {sin_cos_t.shape}")

        # MLP 映射维度
        period_feature_enhance = self.mlp(sin_cos_t)

        # 与 LSTM 模块融合
        h_sp = h2 + period_feature_enhance
        return h_sp

