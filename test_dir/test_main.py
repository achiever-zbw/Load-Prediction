import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.dataset import SubwayDataset , NoneChannelAttnDataset , DatasetProvideWeek
from torch.utils.data import DataLoader
from src.models.model import MainModel
from src.models.lstm_baseline import LSTMBaseline
from test_dir.eval import evaluate_main

checkpoint_path = "saved/checkpoints/best_model.pth"
scaler_dir = "saved/scaler"


def main():
    # 1. 设备、环境配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_step = 288

    # 2. 读取测试数据（直接使用专门的测试文件）
    df_features = pd.read_csv("data/processed/test_shifted.csv")
    df_targets = pd.read_csv("data/processed/test_target_shifted.csv")

    # raws_e = ["temp", "hum", "wind"]
    # raws_s = ["power", "cw_temp", "chw_temp"]
    # raws_r = ["pax", "status", "fan_freq", "pump_freq"]
    raws = ["temp", "hum", "wind" , "power", "cw_temp", "chw_temp" ,"pax", "status", "fan_freq", "pump_freq" ]

    print(f"测试数据加载完成: {len(df_features)} 条样本") 

    # 3. 加载训练保存的标准化
    sx = joblib.load(os.path.join(scaler_dir, "scaler_x.pkl"))
    # ss = joblib.load(os.path.join(scaler_dir, "scaler_s.pkl"))
    # sr = joblib.load(os.path.join(scaler_dir, "scaler_r.pkl"))
    sy = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))

    data_x = sx.transform(df_features[raws])
    # data_s = ss.transform(df_features[raws_s])
    # data_r = sr.transform(df_features[raws_r])
    data_y = sy.transform(df_targets[["total_load_hvac"]])

    time_index = (df_features["time"].values // 5) % 288
    day_of_week = df_features["day_of_week"].values

    # 4. 构建测试数据集（使用全部测试数据，不需要划分）
    test_dataset = DatasetProvideWeek(
        data_x=data_x , time_index=time_index , day_of_week=day_of_week , targets=data_y , time_step=288
    )

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试集样本量: {len(test_dataset)}")

    # 5. 模型初始化与权重加载
    model = MainModel(dim=64, time_step=time_step).to(device)
    print(f"采用的模型 : {model.name}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"成功加载模型权重: {checkpoint_path}")
    
    # 6. 开始评估
    evaluate_main(model, test_dataloader, device, save_dir=scaler_dir)

if __name__ == '__main__':
    main()