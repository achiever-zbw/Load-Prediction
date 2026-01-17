import torch
import torch.nn as nn
from src.data.dataset import SubwayLoadModel
class SimpleRNN(nn.Module) :
    def __init__(self , input_size , hidden_size , num_layers , output_size) : 
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size , hidden_size , num_layers)
        self.fc = nn.Linear(hidden_size , output_size)

    def forward(self , x) : 
        # x : [batch_size , len_sqen , input_size]
        output , _ = self.rnn(x) # output : [batch_size , len_sqen , hidden_size]
        output = output[ :,-1, :] # 取最后一个时间步
        output = self.fc(output)
        return output
    

    
class RNNBaseline(nn.Module) : 
    """
    RNN Baseline 模型，使用 MainModel 相同的特征融合方式
    输入 x_e , x_s , x_r
    """
    def __init__(self , input_size = 64 , hidden_size = 128 , num_layers = 2 , dropout_rate = 0.2 , time_step = 24) : 
        super().__init__()
        self.time_step = time_step
        self.feature_fusion = SubwayLoadModel(dim=input_size , time_step=self.time_step)
        self.rnn = nn.RNN(
            input_size=input_size * 3 ,  # 输入的大小是融合后的 192
            hidden_size=hidden_size , 
            num_layers=num_layers , 
            batch_first=True , 
            dropout=dropout_rate
        )

        # MLP 层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size , hidden_size // 2) , 
            nn.ReLU() , 
            nn.Dropout(dropout_rate) , 
            nn.Linear(hidden_size // 2 , 1)
        )

    def forward(self , x_e , x_s , x_r) : 
        z_concat = self.feature_fusion(x_e , x_s , x_r)     # [B , time_step , 192]
        out , _ = self.rnn(z_concat)    # RNN 向前传播
        last_out = out[: , -1 , :]      # 取最后一个时间步 [B , hidden_size]
        output = self.fc(last_out)

        return output
