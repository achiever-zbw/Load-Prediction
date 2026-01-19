import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from src.models.model import MainModel
from src.data.dataset import SubwayDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score


import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import SubwayDataset
from torch.utils.data import DataLoader
from src.models.lstm_baseline import LSTMBaseline


def evaluate_lstm(model, test_loader, device, save_dir):
    """
    1. 预测推理
    2. 反序列化 (Inverse Transform)
    3. 计算物理指标 (MAE, RMSE, MAPE)
    4. 可视化

    针对 LSTM、RNN 模型，输入三个 x_e , x_s , x_r
    """
    model.eval()
    all_preds = []
    all_targets = []

    scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")
    scaler_y = joblib.load(scaler_y_path)

    print("开始模型推理")
    with torch.no_grad():
        for be, bs, br, bt, target in test_loader:
            # LSTM Baseline 只需要 3 个输入
            output = model(be.to(device), bs.to(device), br.to(device))
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.numpy())

    # 拼接并降维
    preds_norm = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    targets_norm = np.concatenate(all_targets, axis=0).reshape(-1, 1)

    # 转化为物理负荷值
    preds_real = scaler_y.inverse_transform(preds_norm).flatten()
    targets_real = scaler_y.inverse_transform(targets_norm).flatten()

    # 计算指标
    mae = mean_absolute_error(targets_real, preds_real)
    mse = mean_squared_error(targets_real, preds_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_real , preds_real)

    # 避免除以 0 的 MAPE 计算
    mape = np.mean(np.abs((targets_real - preds_real) / (targets_real + 1e-5))) * 100

    print("指标情况 :")
    print(f"MAE  (平均绝对误差): {mae:.2f} kW")
    print(f"RMSE (均方根误差):   {rmse:.2f} kW")
    print(f"MAPE (平均百分比误差): {mape:.2f} %")
    print(f"R2 分数 : {r2:.4f}")
    print(f"总预测点数: {len(targets_real)}")

    # 显示所有数据（不再限制plot_len）
    plt.figure(figsize=(20, 8))

    # 绘制真实值和预测值
    plt.plot(targets_real, label='Actual Load', color='#1f77b4', linewidth=1.2, alpha=0.8)
    plt.plot(preds_real, label='Predicted Load', color='#ff7f0e', linestyle='--', linewidth=1.2, alpha=0.8)

    # 填充误差区域
    plt.fill_between(range(len(targets_real)), targets_real, preds_real, color='gray', alpha=0.15, label='Error Region')

    plt.title(f'LSTM Baseline - Full Test Set Prediction Comparison ({len(targets_real)} points)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('Cooling Load (kW)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)

    # 添加指标文本框
    textstr = f'Evaluation Metrics:\nMAE: {mae:.2f} kW\nRMSE: {rmse:.2f} kW\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 保存结果图
    os.makedirs("saved/results/lstm_baseline", exist_ok=True)
    plt.savefig("saved/results/lstm_baseline/lstm_full_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n[OK] 完整对比曲线图已保存至 saved/results/lstm_baseline/lstm_full_comparison.png")
    plt.show()


def evaluate_main(model, test_loader, device, save_dir):
    """
    1. 预测推理
    2. 反序列化 (Inverse Transform)
    3. 计算物理指标 (MAE, RMSE, MAPE)
    4. 可视化

    针对 MainModel 接受 x, x_t, x_w (特征、日周期、周周期)
    """
    model.eval()
    all_preds = []
    all_targets = []

    scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")
    scaler_y = joblib.load(scaler_y_path)

    print("开始模型推理")
    with torch.no_grad():
        for bx, target, bt, bw in test_loader:
            # MainModel 需要 3 个输入: x, x_t (日周期), x_w (周周期)
            output = model(bx.to(device), bt.to(device), bw.to(device))
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.numpy())

    # 拼接并降维
    preds_norm = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    targets_norm = np.concatenate(all_targets, axis=0).reshape(-1, 1)

    # 转化为物理负荷值
    preds_real = scaler_y.inverse_transform(preds_norm).flatten()
    targets_real = scaler_y.inverse_transform(targets_norm).flatten()

    # 计算指标
    mae = mean_absolute_error(targets_real, preds_real)
    mse = mean_squared_error(targets_real, preds_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_real , preds_real)
    # 避免除以 0 的 MAPE 计算
    mape = np.mean(np.abs((targets_real - preds_real) / (targets_real + 1e-5))) * 100

    print("指标情况 :")
    print(f"MAE  (平均绝对误差): {mae:.2f} kW")
    print(f"RMSE (均方根误差):   {rmse:.2f} kW")
    print(f"MAPE (平均百分比误差): {mape:.2f} %")
    print(f"R2 分数 : {r2:.4f}")
    print(f"总预测点数: {len(targets_real)}")

    # 显示所有数据（不再限制plot_len）
    plt.figure(figsize=(20, 8))

    # 绘制真实值和预测值
    plt.plot(targets_real, label='Actual Load (真实值)', color='#1f77b4', linewidth=1.2, alpha=0.8)
    plt.plot(preds_real, label='Predicted Load (预测值)', color='#ff7f0e', linestyle='--', linewidth=1.2, alpha=0.8)

    # 填充误差区域
    plt.fill_between(range(len(targets_real)), targets_real, preds_real, color='gray', alpha=0.15, label='误差区域')

    plt.title(f'MainModel - Full Test Set Prediction Comparison ({len(targets_real)} points)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('Cooling Load (kW)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)

    # 添加指标文本框
    textstr = f'评估指标:\nMAE: {mae:.2f} kW\nRMSE: {rmse:.2f} kW\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 保存结果图
    os.makedirs("saved/results/", exist_ok=True)
    plt.savefig("saved/results/main_model_full_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n[OK] 完整对比曲线图已保存至 saved/results/main_model_full_comparison.png")
    plt.show()


def evaluate_nonn_channel_main(model, test_loader, device, save_dir , pic_name):
    """
    1. 预测推理
    2. 反序列化 (Inverse Transform)
    3. 计算物理指标 (MAE, RMSE, MAPE)
    4. 可视化

    不进行通道嵌入、特征加权融合的数据
    """
    model.eval()
    all_preds = []
    all_targets = []

    scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")
    scaler_y = joblib.load(scaler_y_path)

    print("开始模型推理")
    with torch.no_grad():
        for bx , target , bt in test_loader:
            
            output = model(bx.to(device), bt.to(device))
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.numpy())

    # 拼接并降维
    preds_norm = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    targets_norm = np.concatenate(all_targets, axis=0).reshape(-1, 1)

    # 转化为物理负荷值
    preds_real = scaler_y.inverse_transform(preds_norm).flatten()
    targets_real = scaler_y.inverse_transform(targets_norm).flatten()

    # 计算指标
    mae = mean_absolute_error(targets_real, preds_real)
    mse = mean_squared_error(targets_real, preds_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_real , preds_real)
    # 避免除以 0 的 MAPE 计算
    mape = np.mean(np.abs((targets_real - preds_real) / (targets_real + 1e-5))) * 100

    print("指标情况 :")
    print(f"MAE  (平均绝对误差): {mae:.2f} kW")
    print(f"RMSE (均方根误差):   {rmse:.2f} kW")
    print(f"MAPE (平均百分比误差): {mape:.2f} %")
    print(f"R2 分数 : {r2:.4f}")
    print(f"总预测点数: {len(targets_real)}")

    # 显示所有数据（不再限制plot_len）
    plt.figure(figsize=(20, 8))

    # 绘制真实值和预测值
    plt.plot(targets_real, label='Actual Load', color='#1f77b4', linewidth=1.2, alpha=0.8)
    plt.plot(preds_real, label='Predicted Load', color='#ff7f0e', linestyle='--', linewidth=1.2, alpha=0.8)

    # 填充误差区域
    plt.fill_between(range(len(targets_real)), targets_real, preds_real, color='gray', alpha=0.15, label='误差区域')

    plt.title(f'{pic_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('Cooling Load (kW)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)

    # 添加指标文本框
    textstr = f'评估指标:\nMAE: {mae:.2f} kW\nRMSE: {rmse:.2f} kW\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 保存结果图
    os.makedirs("saved/results/", exist_ok=True)
    plt.savefig("saved/results/main_model_none_channel_attn_full_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n[OK] 完整对比曲线图已保存至 saved/results/main_model_none_channel_attn_full_comparison.png")
    plt.show()

def evaluate_lstm_baseline(model, test_loader, device, save_dir , pic_name):
    """
    LSTM Baseline 测试
    """
    model.eval()
    all_preds = []
    all_targets = []

    scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")
    scaler_y = joblib.load(scaler_y_path)

    print("开始模型推理")
    with torch.no_grad():
        for bx , target , bt in test_loader:
            
            output = model(bx.to(device))
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.numpy())

    # 拼接并降维
    preds_norm = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    targets_norm = np.concatenate(all_targets, axis=0).reshape(-1, 1)

    # 转化为物理负荷值
    preds_real = scaler_y.inverse_transform(preds_norm).flatten()
    targets_real = scaler_y.inverse_transform(targets_norm).flatten()

    # 计算指标
    mae = mean_absolute_error(targets_real, preds_real)
    mse = mean_squared_error(targets_real, preds_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_real , preds_real)
    # 避免除以 0 的 MAPE 计算
    mape = np.mean(np.abs((targets_real - preds_real) / (targets_real + 1e-5))) * 100

    print("指标情况 :")
    print(f"MAE  (平均绝对误差): {mae:.2f} kW")
    print(f"RMSE (均方根误差):   {rmse:.2f} kW")
    print(f"MAPE (平均百分比误差): {mape:.2f} %")
    print(f"R2 分数 : {r2:.4f}")
    print(f"总预测点数: {len(targets_real)}")

    # 显示所有数据（不再限制plot_len）
    plt.figure(figsize=(20, 8))

    # 绘制真实值和预测值
    plt.plot(targets_real, label='Actual Load', color='#1f77b4', linewidth=1.2, alpha=0.8)
    plt.plot(preds_real, label='Predicted Load', color='#ff7f0e', linestyle='--', linewidth=1.2, alpha=0.8)

    # 填充误差区域
    plt.fill_between(range(len(targets_real)), targets_real, preds_real, color='gray', alpha=0.15, label='误差区域')

    plt.title(f'{pic_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('Cooling Load (kW)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)

    # 添加指标文本框
    textstr = f'评估指标:\nMAE: {mae:.2f} kW\nRMSE: {rmse:.2f} kW\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 保存结果图
    os.makedirs("saved/results/", exist_ok=True)
    plt.savefig("saved/results/lstm_baseline_full_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n[OK] 完整对比曲线图已保存至 saved/results/main_model_none_channel_attn_full_comparison.png")
    plt.show()

