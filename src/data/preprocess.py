# src/data/preprocessing.py

import numpy as np
from sklearn.preprocessing import StandardScaler

def create_sequences(features , target , time_steps) : 
    """
    从特征和目标序列中，创建时间序列样本
    
    :param features: 特征值 将连续的 time_steps 个时间点的数据作为输入(X)
    :param target: 目标值   第 (time_steps + 1) 个时间点的数据作为预测目标(y)
    :param time_steps: 时间步长，用于滑动窗口构建
    """

    X , y = [] , []
    for i in range(len(features) - time_steps) :
        X.append(features[i : i + time_steps]) 
        y.append(target[i + time_steps])

    return np.array(X) , np.array(y)

class DataPreprocessing : 
    """
    封装数据处理，包含特征 和 目标值的标准化处理
    """
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def process_train(self , features , targets) : 
        feature_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(targets.values.reshape(-1 , 1)).flatten()

        return feature_scaled , target_scaled

    def process_test(self, features, targets):
        f_scaled = self.feature_scaler.transform(features)
        t_scaled = self.target_scaler.transform(targets.values.reshape(-1, 1)).flatten()
        return f_scaled, t_scaled
    
if __name__ == "__main__":
    import pandas as pd
    import os

    # 1. 配置真实路径
    data_dir = "/Users/zhaobowen/时间序列预测/data/raw/"
    input_path = os.path.join(data_dir, "一个月数据总表.xlsx")
    target_path = os.path.join(data_dir, "一个月负荷数据总表.xlsx")

    df_input = pd.read_excel(input_path)
    df_target = pd.read_excel(target_path)

    

    
    