import torch
import torch.nn as nn
import math

"""
Transformer 模块，用于全局信息掌握
"""
class Transformer(nn.Module):
    # 输入 7 个维度，映射到 32 ， 4个注意力头，2层编码，输出 1 维
    def __init__(self, input_size=7 , d_model=32, nhead=4, num_layers=2, output_size=1):
        super(Transformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # Transformer 需要 (seq_len, batch, d_model)
        out = self.transformer(x)
        out = out[-1, :, :]
        out = self.fc(out)
        return out