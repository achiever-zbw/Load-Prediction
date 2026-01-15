from .lstm import LSTMBlock
from src.data.dataset import SubwayDataset , SubwayLoadModel
from torch.utils.data import DataLoader
from .attention import ChannelAttentionBlock
from .period import PeriodEnhanceBlock

if __name__ == '__main__' : 
    # print(30 * 24 * 12)

    # 1. 读取数据
    dataset = SubwayDataset(file_path="data/processed/一个月数据总表_10特征.csv")
    print(len(dataset))

    # 第一个块的
    x_e , x_s , x_r , _ ,target = dataset[0]
    print(f"x_e 的shape : {x_e.shape}")  
    print(f"x_s 的shape : {x_s.shape}")   
    print(f"x_r 的shape : {x_r.shape}")  

    # 2. 特征融合
    model = SubwayLoadModel(64 , 10)
    load_data = DataLoader(dataset , 32 , True)
    batch_e , batch_s , batch_r , batch_t , y = next(iter(load_data))
    print(f"第一批次的 E : {batch_e.shape}")
    print(f"第一批次的 S : {batch_s.shape}")
    print(f"第一批次的 R : {batch_r.shape}")
    print(f"第一批次的时间步  : {batch_t.shape}")

    fusion_data = model(batch_e , batch_s , batch_r)
    print(f"融合后的特征向量形状: {fusion_data.shape}") # torch.Size([32, 10, 192])

    # 3. 通道注意力
    attention_model = ChannelAttentionBlock(192)
    z_atten = attention_model(fusion_data)
    print(f"通道注意力层后 Z_atten : {z_atten.shape}")

    # 4. 双层 LSTM
    lstm_model = LSTMBlock(3 , 64)
    h2 = lstm_model(z_atten)
    print(f"经过 LSTM 建模， H2 : {h2.shape}")

    # 5. 周期增强模块
    period_model = PeriodEnhanceBlock(64)
    h_sp = period_model(h2 , batch_t)
    print(f"LSTM 的输出融合了周期特征后 : {h_sp.shape}")