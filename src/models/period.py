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

class PeriodEnhanceDayWeekBlock(nn.Module) :
    """
    周期特征增强模块，包含日周期和周周期
    - 日周期 : 24 * 12 = 288 个点
    - 周周期 : 7
    """
    def __init__(self , hidden_dim = 64 , time_step = 288) :
        super().__init__()
        self.day_steps = 288
        self.week_steps = 7

        # 日周期编码 MLP - 增强版：3层 MLP
        self.day_mlp = nn.Sequential(
            nn.Linear(2 , 64) ,      # 增加中间层维度
            nn.ReLU() ,
            nn.Linear(64 , 64) ,
            nn.ReLU() ,
            nn.Linear(64 , hidden_dim // 2)  # 32维
        )

        # 周周期编码 MLP - 增强版：3层 MLP
        self.week_mlp = nn.Sequential(
            nn.Linear(2 , 64) ,      # 增加中间层维度
            nn.ReLU() ,
            nn.Linear(64 , 64) ,
            nn.ReLU() ,
            nn.Linear(64 , hidden_dim // 2)  # 32维
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim) ,
            nn.LayerNorm(hidden_dim) ,
            nn.GELU()
        )

    def forward(self , h2 , t_index , day_of_week ) :
        """
        forward 的 Docstring

        :param self: 说明
        :param h2: LSTM 的输出隐藏状态，[batch_size , time_step , hidden_dim]
        :param t_index: 日时间索引，取值为 0 ~ 287，[batch_size , time_step]
        :param day_of_week: 周时间索引，取值为 1 ~ 7，[batch_size , time_step]
                           1=周一, 2=周二, ..., 7=周日
        """

        # 1. 日周期编码
        alpha_day = 2.0 * math.pi * t_index / self.day_steps
        sin_day = torch.sin(alpha_day).unsqueeze(-1)  # [B , time_step , 1]
        cos_day = torch.cos(alpha_day).unsqueeze(-1)
        sin_cos_day = torch.cat([sin_day , cos_day] , dim = -1)  # [B , time_step , 2]
        day_period = self.day_mlp(sin_cos_day)  # [B , time_step , hidden_dim // 2]

        # 2. 周周期编码
        # 将 day_of_week 从 1-7 转换为 0-6
        day_of_week_0to6 = day_of_week - 1
        alpha_week = 2.0 * math.pi * day_of_week_0to6 / self.week_steps
        sin_week = torch.sin(alpha_week).unsqueeze(-1)  # [B , time_step , 1]
        cos_week = torch.cos(alpha_week).unsqueeze(-1)
        sin_cos_week = torch.cat([sin_week , cos_week] , dim = -1)  # [B , time_step , 2]
        week_period = self.week_mlp(sin_cos_week)  # [B , time_step , hidden_dim//2]

        # 拼接
        multi_period = torch.cat([day_period , week_period] , dim = -1)     # [B , time_step , hidden_dim]
        # 融合
        period_feature_enhance = self.fusion(multi_period)

        # 与 LSTM 融合
        h_sp = h2 + period_feature_enhance
        return h_sp
 