import torch
import torch.nn as nn
from src.data.dataset import SubwayDataset , SubwayLoadModel 
from torch.utils.data import DataLoader

# 通道注意力
class ChannelAttentionBlock(nn.Module) :
    """
    通道注意力层，模型通过通道注意力机制计算每个通道的注意力权重

    数据形状变化 : 
    - 1. x 传入形状为 [32 , 10 , 192]
    - 2. 首先学习通道的得分，用于后续计算权重, Linear(192 -> 192) , 形状还是 [32 , 10 , 192]
    - 3. 计算权重，使用 Softmax 对得分 e 进行归一化，确保每个权重的总和为 1 
    - 4. 将权重与原始特征对应元素相乘得到加权的特征 , [32 , 10 , 192]
    - 5. 将通道压缩，192 -> 3 ，回到 3 个特征阶段，[32 , 10 , 3]
    """
    def __init__(self , in_channels = 192) :
        super().__init__()
        # 学习各个通道的得分层
        self.score_layer = nn.Sequential(
            nn.Linear(in_channels , in_channels // 4) , 
            nn.ReLU() , 
            nn.Linear(in_channels // 4 , in_channels) , 
            nn.Sigmoid() 
        )
        # self.feature_layer = nn.Linear(64 , 3)
       
    def forward(self , x) :
        """
        x : [batch_size , time_step , 192]
        
        :param self: 说明
        :param x: 说明
        """
        batch_size, seq_len, channels = x.shape

        # 1. 把时间压缩，降维到 [batch_size , 192]，感受全局的通道状态
        channel_status = torch.mean(x , dim = 1)    # 以时间为依据，取均值

        # 2. 学习得分
        e = self.score_layer(channel_status)
        e = e.unsqueeze(1)
        

        # # 3. 得到归一化的注意力权重 , 根据最后一个维度 ， 即特征数 3
        # # 得到三个特征的对应的权重, 注意拓展为 三维张量
        # alpha = torch.softmax(e , dim = -1)
        # alpha_e = alpha[:, 0].view(batch_size , 1 , 1)
        # alpha_s = alpha[:, 1].view(batch_size , 1 , 1)
        # alpha_r = alpha[:, 2].view(batch_size , 1 , 1)

        # # 4. 得到 e s r 三个特征各自原始特征
        # ze = x[:, :, 0 : 64]
        # zs = x[:, :, 64 : 64 * 2]
        # zr = x[:, :, 64 * 2 : ]

        # # 5. 将权重 * 原始数据
        # z_attn = alpha_e * ze + alpha_s * zs + alpha_r * zr     # [32 , 10 , 64]
        # # print(f"融合特征 : {z_attn.shape}")
        # # z_attn = self.feature_layer(z_attn)
        # # print(f"压缩后 : {z_attn.shape}")      # [32 , 10 , 3]
        z_attn = x * e
        return z_attn
    
# 长期依赖
class TransformerBlock(nn.Module) : 
    """
    长期依赖建模层 , 使用Transformer编码器对增强后的特征 H_sp 进行建模，捕捉冷负荷序列中的长期依赖关系
    """
    def __init__(self , dim = 64 , nhead = 4 , num_layer = 3 ) : 
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim ,
            nhead=nhead , 
            dim_feedforward=dim * 4 , 
            # dropout=dropout_rate , 
            batch_first=True ,
            # activation='relu'
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