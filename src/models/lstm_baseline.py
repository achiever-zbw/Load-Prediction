"""
普通 LSTM 基线模型
用于对比实验，使用与 MainModel 相同的特征融合方式
"""
import torch
import torch.nn as nn
from src.models.lstm import LSTMBlock
from src.models.attention import ChannelAttentionBlock


class LSTMBaseline(nn.Module) : 
    """
    LSTMBaseline 的 Docstring
    """
    def __init__(self , input_dim = 10 , hidden_dim = 64 , time_step = 24) : 
        super().__init__()
        self.lstm = LSTMBlock(input_dim=192 , hidden_dim=64)
        self.fc_layer = nn.Linear(in_features=input_dim , out_features=192)
        self.attention = ChannelAttentionBlock(in_channels=192)
        self.output_layer = nn.Linear(hidden_dim , 1)

    def forward(self , x) : 
        """
        x : [B , time_step , 10]
        """
        x = self.fc_layer(x)
        x = self.attention(x)
        x = self.lstm(x)
        x = x[: , -1  , :]
        output = self.output_layer(x)
        
        return output

