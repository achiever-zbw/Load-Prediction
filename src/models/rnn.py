import torch
import torch.nn as nn

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
    

    

