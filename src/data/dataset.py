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
    def __init__(self , time_step = 10):
        # 1. 数据读取
        df_data = pd.read_csv("./data/raw/一个月数据总表.csv" , thousands=",")
        df_target = pd.read_csv("./data/raw/一个月负荷数据总表.csv" , thousands=",")

        # 6 个特征列 和 1 个标签列
        features_raws = ["passengers" , "structure_load" , "vent_load" , "temp" , "hum" , "equip_num"]
        raw_data = df_data[features_raws].values
        raw_target = df_target[["total_load"]].values

        # 2. 标准化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        scaler_data = self.scaler_x.fit_transform(raw_data)
        scaler_target = self.scaler_y.fit_transform(raw_target)

        # 3. 构建窗口
        self.input_seq = create_sequences(scaler_data , time_step)
        # 4. 构造标签窗口
        self.labels = scaler_target[time_step:]

    def __len__(self) : 
        return len(self.input_seq)
    
    # 每个 index 的 dataset 代表一个窗口，由 dataloader 打包后传入 SubwayLoadModel 进行特征的融合
    def __getitem__(self, index):
        return torch.tensor(self.input_seq[index]).float() , torch.tensor(self.labels[index]).float()
    

class SubwayLoadModel(nn.Module) : 
    """
    特征融合嵌入模型，对应专利的步骤一 (多源数据采集与融合)
    1. 特征的标准化 , 这里确保传入的数据已经进行标准化，在 SubwayDataset 中,这里特征和标签（负荷值）都进行了标准化
    2. 每种输入特征(对应多个通道) 通过线性映射进行通道嵌入 。 通过 FeaturesCatBlock 中 Linear 实现，1 -> 32 维度的映射
    3. 每个特征在嵌入后都会进行层归一化、激活函数处理。 通过 FeaturesCatBlock 中的 LayerNorm 和 Tanh 实现
    4. 通过将不同通道的嵌入特征进行拼接，生成统一的向量 。 通过 FeaturesCatBlock 中的 torch.cat 实现

    数据形状的转变 ：
    - 1. 首先加载一个月的数据, [8640 , 6] , 表示 8640 个时间点，每个时间点的 6 个特征
    - 2. 构建窗口。选择窗口的长度为 10 ，则一共有 8630 个窗口，所以数据变为 [8630 , 10 , 6] , 每个 Dataset 代表一个窗口的数据，
         为 [10 , 6]
    - 3. 把 Dataset 进行打包，一共 8630 个窗口，使用 batch_size = 32 进行批次打包，形状为 [32 , 10 , 6]
    - 4. 先不关心 batch_size , 对于每个 [10 , 6]，FeaturesCatBlock 需要对 6 个特征进行拼接，所以要切割成 6 个 [10 , 1] 的形式
    - 5. 对 6 个 [32 , 10 , 1] 进行特征拼接 ， 首先 Linear , 形状变为 [32 , 10 , 32]
    - 6. 再归一化瘀激活函数，形状不变
    - 7. 最后对 6 个特征拼接，形状变为 [32 , 10 , 32 * 6] -> [32 , 10 , 192]
    """

    def __init__(self , num_features = 6 , dim = 32) :
        super().__init__()
        self.fusion_model = FeaturesCatBlock(num_features , dim)

    def forward(self , x) : 
        """
        x 是从 DataLoader 传入的数据，[batch_size , 10 , 6]
        
        :param self: 说明
        :param x: 说明
        """
        # 把 [10 , 6] 切割成 6 个 [10 * 1] , 这样才能进行 6 个特征的融合
        x_list = [x[:, :, i : i + 1] for i in range(x.shape[-1])] 
        # 特征嵌入与融合,执行了 linear -> LayerNorm -> Tanh -> Cat
        fusion_data = self.fusion_model(x_list)
        return fusion_data


if __name__ == '__main__' : 
    # print(30 * 24 * 12)

    dataset = SubwayDataset()
    print(len(dataset)) # 8630
    x , y = dataset[0]
    # 每一个块都划分成 10 * 6 
    print(x.shape)  # torch.Size([10 , 6])
    print(y.shape)  # torch.Size([1])


    model = SubwayLoadModel(6 , 32)
    
    load_data = DataLoader(dataset , 32 , True)
    batch_x, batch_y = next(iter(load_data))
    
    fusion_data = model(batch_x)
    
    # 6. 验证结果
    print(f"输入 Batch 形状: {batch_x.shape}")    # torch.Size([32, 10, 6])
    print(f"融合后的特征向量形状: {fusion_data.shape}") # torch.Size([32, 10, 192])