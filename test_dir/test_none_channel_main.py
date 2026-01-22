import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.dataset import SubwayDataset
from torch.utils.data import DataLoader
from src.models.model import NoneChannelAttnModel
from test_dir.eval import evaluate_main

def main():
    # 1. 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "saved/checkpoints/best_none_channel_attn_model.pth"
    scaler_dir = "saved/scaler/none_channel_attn"
    time_step = 24

    # 2. 数据加载
    df_features = pd.read_csv("data/processed/test_shifted.csv")
    df_targets = pd.read_csv("data/processed/test_load_data.csv")

    # 特征分组（与训练保持一致）
    raws_e = ["temp", "hum", "wind"]
    raws_s = ["power", "cw_temp", "chw_temp"]
    raws_r = ["pax", "status", "fan_freq", "pump_freq", "wind_shen"]

    print(f"测试数据加载完成: {len(df_features)} 条样本")

    # 3. 加载训练保存的标准化 Scaler
    scaler_e = joblib.load(os.path.join(scaler_dir, "scaler_e.pkl"))
    scaler_s = joblib.load(os.path.join(scaler_dir, "scaler_s.pkl"))
    scaler_r = joblib.load(os.path.join(scaler_dir, "scaler_r.pkl"))
    scaler_y = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))

    # 应用标准化（分组处理）
    data_e = scaler_e.transform(df_features[raws_e])
    data_s = scaler_s.transform(df_features[raws_s])
    data_r = scaler_r.transform(df_features[raws_r])
    data_y = scaler_y.transform(df_targets[["total_load"]])

    print("测试数据处理完成: 标准化 (均值=0, 标准差=1)")
    print(f"环境特征范围: [{data_e.min():.3f}, {data_e.max():.3f}]")
    print(f"系统特征范围: [{data_s.min():.3f}, {data_s.max():.3f}]")
    print(f"工况特征范围: [{data_r.min():.3f}, {data_r.max():.3f}]")
    print(f"目标范围: [{data_y.min():.3f}, {data_y.max():.3f}]")

    # 4. 时间索引处理
    time_index = (df_features["time"].values // 5) % 288

    # 5. 构建测试数据集（使用SubwayDataset，与训练保持一致）
    test_dataset = SubwayDataset(
        data_e=data_e,
        data_s=data_s,
        data_r=data_r,
        time_index=time_index,
        targets=data_y,
        time_step=24
    )

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试集样本量: {len(test_dataset)}")

    # 6. 模型初始化与权重加载
    model = NoneChannelAttnModel(dim=64, time_step=24).to(device)
    print(f"采用的模型 : {model.name}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"成功加载模型权重: {checkpoint_path}")

    # 7. 开始评估
    evaluate_main(model, test_dataloader, device, save_dir=scaler_dir)

if __name__ == '__main__':
    main()
