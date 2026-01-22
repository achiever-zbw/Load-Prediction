# 整合全流程模型
import torch
import torch.nn as nn 
from src.models.attention import ChannelAttentionBlock , TransformerBlock
from src.models.lstm import LSTMBlock
from src.models.period import PeriodEnhanceBlock , PeriodEnhanceDayWeekBlock
from src.models.output import TaskOutPutBlock
from src.data.dataset import SubwayLoadModel
from src.data.channel_embedding import FeaturesCatBlock

class MainModel(nn.Module) :
    """
    冷负荷预测模型
    """
    def __init__(self , dim , time_step ) :
        super().__init__()
        self.time_step = time_step
        self.name = "MainModel"
        # 特征融合模块
        self.feature_fusion = FeaturesCatBlock(channel_e=3 , channel_s=3 , channel_r=5 , dim=dim)
        # 通道注意力层
        self.channel_attention = ChannelAttentionBlock(in_channels = dim * 3)
        # 短时序建模
        self.lstm = LSTMBlock(input_dim = dim * 3 , hidden_dim = dim * 2)
        # 周期特征增强模块
        self.period = PeriodEnhanceBlock(hidden_dim = dim * 2 , time_step = time_step)
        # self.period = PeriodEnhanceDayWeekBlock(hidden_dim=64 , time_step=time_step)
        # 长期依赖建模
        self.transformer = TransformerBlock(dim=dim * 2, nhead=4 , num_layer=2 , dropout_rate=0.2)
        # 多任务输出
        self.output_layer = TaskOutPutBlock(dim=dim * 2)

    def forward(self , x_e ,x_s , x_r , x_t) :
        """
        x_e : 环境特征 [B , time_step , 3]
        x_s : 系统特征 [B , time_step , 3]
        x_r : 工况特征 [B , time_step , 5]
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
    消融实验模型：无通道注意力版本
    与 MainModel 结构完全一致，只是去掉了通道注意力层
    用于验证通道注意力模块的作用
    """
    def __init__(self , dim = 64 , time_step = 48) :
        super().__init__()
        self.name = "NoneChannelAttnModel"
        self.time_step = time_step
        # 特征融合模块（与 MainModel 一致）
        self.feature_fusion = FeaturesCatBlock(channel_e=3 , channel_s=3 , channel_r=5 , dim=64)
        # 注意：没有通道注意力层
        # 短时序建模（与 MainModel 一致）
        self.lstm = LSTMBlock(input_dim = dim * 3 , hidden_dim = dim)
        # 周期特征增强模块（与 MainModel 一致）
        self.period = PeriodEnhanceBlock(hidden_dim = dim, time_step = time_step)
        # 长期依赖建模（与 MainModel 一致）
        self.transformer = TransformerBlock(dim=64, nhead=4 , num_layer=2)
        # 多任务输出（与 MainModel 一致）
        self.output_layer = TaskOutPutBlock(dim=64)

    def forward(self , x_e , x_s , x_r , x_t) :
        """
        x_e : 环境特征 [B , time_step , 3]
        x_s : 系统特征 [B , time_step , 3]
        x_r : 工况特征 [B , time_step , 5]
        x_t : 时间周期特征 [B , time_step]
        """
        # 1. 通道嵌入，特征融合（与 MainModel 一致）
        z_concat = self.feature_fusion(x_e , x_s , x_r)
        # 注意：跳过通道注意力层
        # 2. 短时序建模
        h_2 = self.lstm(z_concat)
        # 3. 周期特征增强
        h_sp = self.period(h_2 , x_t)
        # 4. 长期依赖建模
        h_l = self.transformer(h_sp)
        # 5. 输出
        main_output = self.output_layer(h_l)
        return main_output
    

class NoneTransformerModel(nn.Module) :
    """
    消融实验模型：无长期依赖建模版本
    与 MainModel 结构完全一致，只是去掉了 Transformer 长期依赖建模层
    用于验证 Transformer 模块的作用
    """
    def __init__(self , dim = 64 , time_step = 48) :
        super().__init__()
        self.name = "NoneTransformerModel"
        self.time_step = time_step
        # 特征融合模块（与 MainModel 一致）
        self.feature_fusion = FeaturesCatBlock(channel_e=3 , channel_s=3 , channel_r=5 , dim=64)
        # 通道注意力层（与 MainModel 一致）
        self.channel_attention = ChannelAttentionBlock(in_channels = dim * 3)
        # 短时序建模（与 MainModel 一致）
        self.lstm = LSTMBlock(input_dim = dim * 3 , hidden_dim = dim)
        # 周期特征增强模块（与 MainModel 一致）
        self.period = PeriodEnhanceBlock(hidden_dim = dim, time_step = time_step)
        # 注意：没有 Transformer 长期依赖建模层
        # 多任务输出（与 MainModel 一致）
        self.output_layer = TaskOutPutBlock(dim=64)

    def forward(self , x_e , x_s , x_r , x_t) :
        """
        x_e : 环境特征 [B , time_step , 3]
        x_s : 系统特征 [B , time_step , 3]
        x_r : 工况特征 [B , time_step , 5]
        x_t : 时间周期特征 [B , time_step]
        """
        # 1. 通道嵌入，特征融合（与 MainModel 一致）
        z_concat = self.feature_fusion(x_e , x_s , x_r)
        # 2. 通道注意力（与 MainModel 一致）
        z_attn = self.channel_attention(z_concat)
        # 3. 短时序建模（与 MainModel 一致）
        h_2 = self.lstm(z_attn)
        # 4. 周期特征增强（与 MainModel 一致）
        h_sp = self.period(h_2 , x_t)
        # 跳过 Transformer 长期依赖建模层
        # 5. 直接取最后一个时间步进行输出
        h_l = h_sp[: , -1 , :]      # [B, 64] - 取最后一个时间步
        # 6. 输出
        main_output = self.output_layer(h_l)
        return main_output


class NoneChannelTransAttnModel(nn.Module) : 
    """
    无通道注意力，无 Transformer
    """
    def __init__(self , dim = 64 , time_step = 48) : 
        super().__init__()
        self.name = "无通道注意力 - 无长期依赖建模模型"
        self.time_step = time_step
        self.layer1 = nn.Linear(in_features=10 , out_features=dim * 3)
        self.lstm = LSTMBlock(input_dim=64 * 3 , hidden_dim=dim)
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
        # 4. 直接取最后一个时间步
        trans_out = lstm_period[: , -1 , :]
        # out
        output = self.output_layer(trans_out)
        return output
    

class NonePeriodModel(nn.Module) :
    """
    无周期特征编码
    """
    def __init__(self , dim = 64 , time_step = 48) : 
        super().__init__()
        self.name = "无周期特征模型"
        self.time_step = time_step
        self.layer1 = nn.Linear(in_features=10 , out_features=dim * 2)
        self.channel_attention = ChannelAttentionBlock(in_channels=dim * 2)
        self.lstm = LSTMBlock(input_dim=dim * 2 , hidden_dim=dim)
        self.transformer = TransformerBlock(dim = 64 , nhead=4 , num_layer=2)
        self.output_layer = TaskOutPutBlock(dim=64)

    def forward(self , x) : 
        x = self.layer1(x)
        x = self.channel_attention(x)
        x = self.lstm(x)
        x = self.transformer(x)
        output = self.output_layer(x)
        return output
