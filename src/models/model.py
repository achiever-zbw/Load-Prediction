# 整合全流程模型
import torch
import torch.nn as nn 
from src.models.attention import ChannelAttentionBlock , TransformerBlock
from src.models.lstm import LSTMBlock
from src.models.period import PeriodEnhanceBlock
from src.models.output import TaskOutPutBlock
from src.data.dataset import SubwayLoadModel

class MainModel(nn.Module) : 
    """
    冷负荷预测模型
    """
    def __init__(self , dim = 64 , time_step = 24) :
        super().__init__()
        self.time_step = time_step
        self.name = "MainModel"
        # 特征融合模块
        self.feature_fusion = SubwayLoadModel(64 , time_step)
        # 通道注意力层
        self.channel_attention = ChannelAttentionBlock(dim * 3)
        # 短时序建模
        self.lstm = LSTMBlock(input_dim = dim * 3 , hidden_dim=dim)
        # 周期特征增强模块
        self.period = PeriodEnhanceBlock(hidden_dim=64, time_step=time_step)
        # 长期依赖建模
        self.transformer = TransformerBlock(dim=64, nhead=4 , num_layer=2)
        # 多任务输出
        self.output_layer = TaskOutPutBlock(dim=64)

    def forward(self , x_e ,x_s , x_r , x_t) :
        """
        x_e : 环境特征 [B , time_step , 3]
        x_s : 系统特征 [B , time_step , 3]
        x_r : 工况特征 [B , time_step , 4]
        x_t : 时间周期特征 [B , 1]
        """
        # 1. 通道嵌入，特征融合
        z_concat = self.feature_fusion(x_e , x_s , x_r)
        # 2. 通道注意力
        z_attn = self.channel_attention(z_concat)
        # 3. 短时序建模
        h_2 = self.lstm(z_attn)
        # 4. 周期特征增强
        h_sp = self.period(h_2 , x_t)
        # 5. 长期依赖建模
        h_l = self.transformer(h_sp)
        # 6. 输出       
        main_output = self.output_layer(h_l)

        return main_output


class NoneChannelAttnModel(nn.Module) : 
    """
    无通道注意力的影响，直接传入 [B , time_step , 10] 的特征向量
    """
    def __init__(self , dim = 64 , time_step = 24) : 
        super().__init__()
        self.time_step = time_step
        self.layer1 = nn.Linear(in_features=10 , out_features=dim)
        self.lstm = LSTMBlock(input_dim=64 , hidden_dim=dim)
        self.period = PeriodEnhanceBlock(hidden_dim=64 , time_step=time_step)
        self.transformer = TransformerBlock(dim = 64 , nhead=4 , num_layer=2)
        self.output_layer = TaskOutPutBlock(dim = 64)

    def forward(self , x , x_t) : 
        # 1. 线性映射
        li = self.layer1(x)
        # 2. lstm
        lstm_out = self.lstm(li)
        # 3. 融合周期特征
        lstm_period = self.period(lstm_out , x_t)
        # 4. transformer
        trans_out = self.transformer(lstm_period)
        # out
        output = self.output_layer(trans_out)
        return output