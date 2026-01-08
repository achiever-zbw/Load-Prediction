import torch
import torch.nn as nn
from .lstm import LSTM
from .transformer import Transformer

class FusionModel(nn.Module) :
    """融合模型"""
    def __init__(self , input_size = 7 , d_model = 32 , n_head = 2 , num_layers = 2 , output_size = 1 , dropout = 0.2) :
        super(FusionModel , self).__init__()

        self.transformer = Transformer(input_size , d_model , n_head , num_layers , output_size)
        self.lstm = LSTM(input_size , hidden_size=d_model , num_layers=num_layers , output_size=output_size , dropout_rate=dropout)
        self.fc = nn.Linear(input_size , output_size)

    def forward(self , x) :
        transformer_out = self.transformer(x)
        lstm_out = self.lstm(x)
        fc_out = self.fc(x[: , -1 ,:])

        return transformer_out + lstm_out + fc_out
