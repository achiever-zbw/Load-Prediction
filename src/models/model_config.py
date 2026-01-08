# 多种模型的配置

from src.models.fusion_model import FusionModel
from src.models.lstm import LSTM
from src.models.rnn import SimpleRNN

def get_model(option) : 
    if option == "LSTM" : 
        return LSTM(
            input_size = 7 , hidden_size = 32 , num_layers = 2 , output_size = 1 , dropout_rate = 0.3
        )
    
    elif option == "RNN" : 
        return SimpleRNN(
            input_size = 7 , hidden_size = 32 , num_layers = 2 , output_size = 1
        )
    
    elif option == "LSTM_Transformer" : 
        return FusionModel(
            input_size=7 , d_model=32 , n_head=2 , num_layers=2 , output_size=1 , dropout=0.2
        )
    else :
        return None