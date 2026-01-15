import pandas as pd
import torch
from torch.utils.data import TensorDataset , DataLoader , Dataset
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from .channel_embedding import FeaturesCatBlock , create_sequences

def get_dataloader(features , targets , batch_size , shuffle = True) : 
    """
    将 numpy 转化为 torch 张量并处理
    
    :param features: 构建的 X(np.array)
    :param targets:  构建的 y(np.array)
    :param batch_size: 批次
    :param shuffle: 训练集不打乱数据
    """

    # 转化为张量
    X_tensor = torch.tensor(features , dtype = torch.float32)
    y_tensor = torch.tensor(targets , dtype=torch.float32).unsqueeze(1)
    # 封装为 dataset
    dataset = TensorDataset(X_tensor , y_tensor)
    # 创建 dataloader
    loader = DataLoader(dataset , batch_size , shuffle=True)

    return loader

# 划分训练集与验证集
def data_split(datas , train_size , val_size , test_size) : 
    train_data = datas[:train_size]
    val_data = datas[train_size : train_size + val_size]
    test_data = datas[train_size + val_size : ]
    return train_data , val_data , test_data


class SubwayDataset(Dataset) : 
    """
    构造数据集，形状为 [10 , 6]  , [1, ]  , 表示，每一个窗口的长度为 10 ，6 个特征 ; 以及 1 个对应的预测值
    """
    def __init__(self , file_path , time_step = 10):
        # 1. 数据读取
        df_data = pd.read_csv(file_path , thousands=",")
        df_target = pd.read_csv("./data/raw/一个月负荷数据总表.csv" , thousands=",")

        # 1. 特征分组，三类特征 E S R
        raws_e = ["temp","hum"	,"wind"]
        raws_s = ["power",	"cw_temp", "chw_temp"]
        raws_r = ["pax"	, "status"	 , "fan_freq"	, "pump_freq"]

        # 提取时间列，用于后续周期特征增强模块
        raw_time_data = df_data["time"].values
        # 将分钟数映射为时间步
        self.time_index = (raw_time_data // 5).astype(int)

        # 2. 读取数据
        raw_e_data = df_data[raws_e].values
        raw_s_data = df_data[raws_s].values
        raw_r_data = df_data[raws_r].values

        raw_target = df_target[["total_load"]].values

        # 3. 标准化
        self.scaler_e = StandardScaler()
        self.scaler_s = StandardScaler()
        self.scaler_r = StandardScaler()

        scaler_e_data = self.scaler_e.fit_transform(raw_e_data)
        scaler_s_data = self.scaler_s.fit_transform(raw_s_data)
        scaler_r_data = self.scaler_r.fit_transform(raw_r_data)

        self.scaler_y = StandardScaler()
        scaler_target = self.scaler_y.fit_transform(raw_target)

        # 4. 构建窗口
        self.input_seq_e = create_sequences(scaler_e_data , time_step)
        self.input_seq_s = create_sequences(scaler_s_data , time_step)
        self.input_seq_r = create_sequences(scaler_r_data , time_step)
        # print(f"E 窗口 {self.input_seq_e.shape}")   # (8630, 10, 3)
        # print(f"S 窗口 {self.input_seq_s.shape}")   # (8630, 10, 3)
        # print(f"R 窗口 {self.input_seq_r.shape}")   # (8630, 10, 4)
        # 4. 构造标签窗口
        self.labels = scaler_target[time_step:]

    def __len__(self) : 
        return len(self.input_seq_e)
    
    # 每个 index 的 dataset 代表一个窗口，由 dataloader 打包后传入 SubwayLoadModel 进行特征的融合
    def __getitem__(self, index):
        """
        返回 4 个数据，E S R Target
        """
        x_e = torch.tensor(self.input_seq_e[index]).float()
        x_s = torch.tensor(self.input_seq_s[index]).float()
        x_r = torch.tensor(self.input_seq_r[index]).float()
        target = torch.tensor(self.labels[index]).float()

        # 提取预测目标对应的时间索引 t
        t_index = torch.tensor([self.time_index[index + 10]]).float()

        return x_e , x_s , x_r , t_index , target
    

class SubwayLoadModel(nn.Module) : 
    """
    通道嵌入与特征融合模型，总流程 : 
    - 1. 共三个特征，E(环境) , S(系统) , R(工况) , 对应 3 3 4 个子特征，即通道
    - 2. 首先进行每种输入特征（对应多个通道）通过线性映射进行通道嵌入
        - 首先读取数据，按照三个分类进行读取，大小分别为 [8640 , 3] [8640 , 3] [8640 , 4]
        - 进行标准化，将三组特征进行标准化，统一量纲 ，通过 SubwayDataset 里的 StandardScaler() 实现
        - 构建窗口，以 10 为步长，对三个特征分别构建窗口，得到 [8630 , 10 , 3] [8630 , 10 , 3] [8630 , 10 , 4] 
        - 三组数据进行批次 32 打包，构建出 [32 , 10 , 3] [32 , 10 , 3] [32 , 10 , 4]
        - 将 3 个特征进行各自的线性映射，实现通道的嵌入 。得到 [32 , 10 , 64] [32 , 10 , 64] [32 , 10 , 64]
        - 嵌入后进行层归一化和激活函数处理
        - 将 嵌入特征进行拼接，生成统一的特征向量 Z(concat) ，形状为 [32 , 10 , 192]
    
    """

    def __init__(self , dim = 64 , time_step = 10) :
        super().__init__()
        self.fusion_model = FeaturesCatBlock(dim)

    def forward(self , x_e , x_s , x_r) : 
        """
        x 是从 DataLoader 传入的数据，[batch_size , 10 , num_features] , e s r 分别为 3 3 4 
        
        :param self: 说明
        :param x: 说明
        """
        # 特征嵌入与融合,执行了 linear -> LayerNorm -> Tanh -> Cat
        fusion_data = self.fusion_model(x_e , x_s , x_r)
        return fusion_data


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