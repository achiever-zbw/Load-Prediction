import torch
import torch.nn as nn

"""LSTM模块
"""
class LSTM(nn.Module) :
    def __init__(self , input_size , hidden_size , num_layers , output_size  , dropout_rate ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size , hidden_size , num_layers , batch_first = True , dropout = dropout_rate 
        )

        # 简化的输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self , x):
        # LSTM输出
        out , _ = self.lstm(x)
        # 取最后一个时间步
        out = out[:,-1,:]
        # Dropout和线性输出
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
class LSTMBlock(nn.Module) : 
    """
    短时序建模层 , 两层 LSTM 建模
    """
    def __init__(self , input_dim , hidden_dim , dropout_rate = 0.2) : 
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim , hidden_size=hidden_dim , batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim , hidden_size=hidden_dim , batch_first=True
        )
    
    def forward(self , x) : 
        """
        x : [batch_size , time_step , 192]
        """
        # h1 = LSTM1(x)
        out1 , _ = self.lstm1(x)
        out1 = self.dropout(out1)

        # h2 = LSTM2(h1)
        out2 , (h_n2 , c_n2) = self.lstm2(out1)
        # 取最后一个时间步
        # final_feature = out2[ : , - 1 ,  :]
        
        # 这里返回整个序列，[batch_size , 10 , 64]，以便于传递给 Transformer 层
        return out2