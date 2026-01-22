import torch
import torch.nn as nn
from src.data.dataset import SubwayDataset , SubwayLoadModel 
from torch.utils.data import DataLoader

# 通道注意力
class ChannelAttentionBlock(nn.Module) :
    """
    通道注意力层（SENet风格）
    使用全局平均池化聚合时间维度信息，然后学习通道权重
    """
    def __init__(self , in_channels) :
        super().__init__()
        # 使用自适应平均池化在时间维度上进行全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # 学习各个通道的得分层
        self.score_layer = nn.Sequential(
            nn.Linear(in_channels , in_channels // 4) ,
            nn.ReLU(inplace=True) ,
            nn.Linear(in_channels // 4 , in_channels) ,
            nn.Sigmoid()
        )

    def forward(self , x) :
        """
        x : [batch_size , time_step , channel_num]
        """
        # 1. 全局平均池化：使用 AdaptiveAvgPool1d
        # 需要先转置：[B, T, C] -> [B, C, T] 才能使用 AdaptiveAvgPool1d
        x_permuted = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        channel_status = self.global_pool(x_permuted)  # [B, C, T] -> [B, C, 1]
        channel_status = channel_status.squeeze(-1)      # [B, C, 1] -> [B, C]

        # 2. 学习得分
        e = self.score_layer(channel_status)  # [B, C]
        e = e.unsqueeze(1)                       # [B, C] -> [B, 1, C]

        # 3. 应用注意力权重
        z_attn = x * e                           # [B, T, C] * [B, 1, C] -> [B, T, C]

        return z_attn
    

# 长期依赖
class TransformerBlock(nn.Module) : 
    """
    长期依赖建模层 , 使用Transformer编码器对增强后的特征 H_sp 进行建模，捕捉冷负荷序列中的长期依赖关系
    """
    def __init__(self , dim , nhead , num_layer , dropout_rate) : 
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim ,
            nhead=nhead , 
            dim_feedforward=dim * 2 , 
            dropout=dropout_rate , 
            batch_first=True ,
            activation='relu'
        )

        # Transformer编码器
        self.transformer_layer = nn.TransformerEncoder(encoder_layer , num_layers=num_layer)

    def forward(self , x) : 
        """
        x : [batch_size , 10 , 64]
        """
        h_l = self.transformer_layer(x)
        # 取最后一个时间步输出
        return h_l[: , -1 , :]



if __name__ == '__main__' : 
    # print(30 * 24 * 12)

    dataset = SubwayDataset(file_path="data/processed/一个月数据总表_10特征.csv")
    print(len(dataset))

    x_e , x_s , x_r , target = dataset[0]
    print(f"x_e 的shape : {x_e.shape}")  
    print(f"x_s 的shape : {x_s.shape}")   
    print(f"x_r 的shape : {x_r.shape}")  

    model = SubwayLoadModel(64 , 10)
    
    load_data = DataLoader(dataset , 32 , True)
    batch_e , batch_s , batch_r , y = next(iter(load_data))
    
    fusion_data = model(batch_e , batch_s , batch_r)
    
    # 6. 验证结果
    print(f"融合后的特征向量形状: {fusion_data.shape}") # torch.Size([32, 10, 192])

    attention_model = ChannelAttentionBlock(192)
    z_atten = attention_model(fusion_data)