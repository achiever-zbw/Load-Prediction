# 加载处理完好后的数据
import pandas as pd
from src.utils.time_utils import time_to_minutes
from src.data.dataset import get_dataloader, data_split
from src.data.preprocess import create_sequences, DataPreprocessing
import os

def prepare_data_pipeline(config) : 
    """
    处理好所有数据
    
    :param config: 配置文件
    """
    # 1.加载原始数据
    input_data_file , target_data_file = config["paths"]["raw_data_input"]  , config["paths"]["raw_data_target"]
    input_data = pd.read_excel(input_data_file)
    target_data = pd.read_excel(target_data_file)["total_load"]

    # 2.特征转换 : 时间转化为分钟
    input_data["minutes"] = input_data["time"].apply(time_to_minutes)

    # 3.提取特征列 与 目标列
    features_col = ["minutes" , "passengers" , "structure_load" , "vent_load" , "temp" , "hum" , "equip_num"]
    features_data = input_data[features_col].values
    
    # 4.数据划分
    total_samples = len(features_data)
    train_ratio = config["sequence_params"]["train_split"]
    train_size = (int)(total_samples * train_ratio)

    val_ratio = config["sequence_params"]["val_split"]
    val_size = (int)(total_samples * val_ratio)

    test_size = total_samples - train_size - val_size 

    # 特征值的划分
    features_train , features_val , features_test = data_split(features_data , train_size , val_size , test_size)
    # 目标值划分
    target_train , target_val , target_test = data_split(target_data , train_size , val_size , test_size)
    
    # 5.标准化
    data_process = DataPreprocessing()
    features_train_scaled , target_train_scaled = data_process.process_train(features_train , target_train)
    features_val_scaled , target_val_scaled = data_process.process_test(features_val , target_val)
    features_test_scaled , target_test_scaled = data_process.process_test(features_test , target_test)
    
    # 滑动窗口构建序列
    time_steps = config["sequence_params"]["time_steps"]
    X_train , y_train = create_sequences(features_train_scaled , target_train_scaled , time_steps)
    X_val , y_val = create_sequences(features_val_scaled , target_val_scaled , time_steps)
    X_test , y_test = create_sequences(features_test_scaled , target_test_scaled , time_steps)

    # 创建 loader
    batch_size = config["sequence_params"]["batch_size"]
    train_loader = get_dataloader(X_train , y_train , batch_size , shuffle=True)
    val_loader = get_dataloader(X_val , y_val , batch_size , shuffle=False)
    test_loader = get_dataloader(X_test , y_test , batch_size , shuffle=False)

    return train_loader , val_loader , test_loader , data_process