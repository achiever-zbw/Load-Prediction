import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.dataset import SubwayDataset , NoneChannelAttnDataset
from torch.utils.data import DataLoader
from src.models.lstm_baseline import LSTMBaseline
from src.models.model import NoneChannelAttnModel
from .eval import evaluate_nonn_channel_main

def main():
    # 1. 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "saved/checkpoints/best_none_channel_main_model.pth"
    scaler_dir = "saved/scaler/none_channel_main"
    time_step = 24

    # 2. 数据加载
    df_features = pd.read_csv("data/processed/test_shifted.csv")
    df_targets = pd.read_csv("data/processed/test_target_shifted.csv")

    raws = ["temp", "hum", "wind" , "power", "cw_temp", "chw_temp" ,"pax", "status", "fan_freq", "pump_freq" ]

    # total_len = len(df_features)
    # test_start = int(total_len * 0.8)

    # 3. 加载并应用 Scalers
    sx = joblib.load(os.path.join(scaler_dir, "scaler_x.pkl"))
    sy = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))

    data_x = sx.transform(df_features[raws])
    data_y = sy.transform(df_targets[["total_load_hvac"]])

    time_index = (df_features["time"].values // 5) % 288

    # 4. 构建测试集 
    test_dataset = NoneChannelAttnDataset(
        data_x=data_x , 
        time_index=time_index ,
        targets=data_y , 
        time_step=288
    )

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试集样本量: {len(test_dataset)}")

    # 5. 模型初始化与权重加载
    model = NoneChannelAttnModel(dim=64 , time_step=24).to(device)
    print(f"使用的模型 : {model.name}")


    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"成功加载模型权重: {checkpoint_path}")

    # 6. 开始评估
    evaluate_nonn_channel_main(model, test_dataloader, device, save_dir=scaler_dir , pic_name="None Channel")

if __name__ == '__main__':
    main()
