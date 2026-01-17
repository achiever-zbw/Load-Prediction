import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.dataset import SubwayDataset
from torch.utils.data import DataLoader
from src.models.rnn import RNNBaseline
from .eval import evaluate_lstm


def main():
    # 1. 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "saved/checkpoints/rnn_baseline_best.pth"
    scaler_dir = "saved/scaler/rnn"
    time_step = 24

    # 2. 数据加载
    df_features = pd.read_csv("data/processed/test.csv")
    df_targets = pd.read_csv("data/processed/test_target_shifted.csv")

    raws_e = ["temp", "hum", "wind"]
    raws_s = ["power", "cw_temp", "chw_temp"]
    raws_r = ["pax", "status", "fan_freq", "pump_freq"]

    # total_len = len(df_features)
    # test_start = int(total_len * 0.8)

    # 3. 加载并应用 Scalers
    se = joblib.load(os.path.join(scaler_dir, "scaler_e.pkl"))
    ss = joblib.load(os.path.join(scaler_dir, "scaler_s.pkl"))
    sr = joblib.load(os.path.join(scaler_dir, "scaler_r.pkl"))
    sy = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))

    data_e = se.transform(df_features[raws_e])
    data_s = ss.transform(df_features[raws_s])
    data_r = sr.transform(df_features[raws_r])
    data_y = sy.transform(df_targets[["total_load_hvac"]])

    time_index = (df_features["time"].values // 5) % 288

    # 4. 构建测试集 
    test_dataset = SubwayDataset(
        data_e,
        data_s,
        data_r,
        time_index,
        data_y,
        time_step=time_step
    )

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试集样本量: {len(test_dataset)}")

    # 5. 模型初始化与权重加载
    model = RNNBaseline(input_size=64 , hidden_size=128 , num_layers=2 , dropout_rate=0.2 , time_step=24).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"成功加载模型权重: {checkpoint_path}")

    # 6. 开始评估
    evaluate_lstm(model, test_dataloader, device, save_dir=scaler_dir)

if __name__ == '__main__':
    main()
