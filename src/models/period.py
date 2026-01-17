import torch
import torch.nn as nn
import math
from .lstm import LSTMBlock

class PeriodEnhanceBlock(nn.Module) :
    """
    周期特征增强模块，以 24 小时为一个周期 24 * 60 / 5 = 288
    """
    def __init__(self , hidden_dim = 64, time_step = 10) :
        super().__init__()
        self.time_step = time_step
        self.steps = 288
        self.mlp = nn.Sequential(
            nn.Linear(2 , 64) ,     # 输入是 [sin , cos ] 二维
            nn.ReLU() ,
            nn.Linear(64 , 64) ,    # 维度为 64 ，与 lstm 的输出对齐
            nn.ReLU() , 
            nn.Linear(64 , hidden_dim)
        )

    def forward(self , h2 , t_index) :
        """
        h2 : LSTM 的输出隐藏状态，注意，这里是整个序列，而不是最后一个时间步， [batch_size , time_step , hidden_dim]
        t_index : 时间步 取值为 0 ~ 287 , 大小为 [batch_size , time_step]
        
        """
        alpha = 2.0 * math.pi * t_index / self.steps
        sin_t = torch.sin(alpha).unsqueeze(-1)  # 增加最后一维用于拼接 [B , time_step , 1]
        cos_t = torch.cos(alpha).unsqueeze(-1)
        # SinCos(t) : [sin_t , cos_t] 拼接一下
        sin_cos_t = torch.cat([sin_t , cos_t] , dim = -1) # [B , time_step , 2]
        # print(f"SinCos(t) : {sin_cos_t.shape}")

        # MLP 映射维度
        period_feature_enhance = self.mlp(sin_cos_t)

        # 与 LSTM 模块融合
        h_sp = h2 + period_feature_enhance
        return h_sp

