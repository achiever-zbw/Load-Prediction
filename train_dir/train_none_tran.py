# 无Transformer模型的训练脚本 - 消融实验
# 与 MainModel 对比，验证Transformer长期依赖建模的作用
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.model import NoneTransformerModel
from src.data.dataset import SubwayDataset
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 计算指标
def calculate_metrics(predictions, targets, scaler_y):
    """
    计算评估指标: MAE, RMSE, MAPE
    """
    # 反标准化到原始尺度
    pred_original = scaler_y.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).flatten()
    target_original = scaler_y.inverse_transform(targets.cpu().numpy().reshape(-1, 1)).flatten()

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred_original - target_original))

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((pred_original - target_original) ** 2))

    # MAPE (Mean Absolute Percentage Error) - 避免除零
    mask = target_original != 0
    mape = np.mean(np.abs((target_original[mask] - pred_original[mask]) / target_original[mask])) * 100

    return mae, rmse, mape


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0

    # 加载数据集
    for batch_e, batch_s, batch_r, targets, batch_t in train_loader:
        batch_e = batch_e.to(device)
        batch_s = batch_s.to(device)
        batch_r = batch_r.to(device)
        targets = targets.to(device)
        batch_t = batch_t.to(device)

        # 前向传播
        main_output = model(batch_e, batch_s, batch_r, batch_t)
        loss_main = loss_fn(main_output, targets)

        # 反向传播
        optimizer.zero_grad()
        loss_main.backward()

        # 梯度裁剪：防止 LSTM 梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        running_loss += loss_main.item()

    return running_loss / len(train_loader)


def main():
    # 1. 配置与环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "saved/checkpoints/best_none_transformer_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. 数据准备（与 train_main.py 保持一致）
    features_file_path = "data/processed/2_month_data.csv"
    target_file_path = "data/processed/2_month_load_data.csv"

    df_features = pd.read_csv(features_file_path)
    df_targets = pd.read_csv(target_file_path)

    # 特征列（分三组，与 MainModel 一致）
    raws_e = ["temp", "hum", "wind"]
    raws_s = ["power", "cw_temp", "chw_temp"]
    raws_r = ["pax", "status", "fan_freq", "pump_freq", "wind_shen"]

    # 划分训练集 验证集
    total_len = len(df_features)
    train_size = int(0.5 * total_len)

    # 标准化处理 (StandardScaler - 均值=0, 标准差=1)
    scaler_e = StandardScaler()
    scaler_s = StandardScaler()
    scaler_r = StandardScaler()
    scaler_y = StandardScaler()

    # 注意，只在训练集上 fit ！
    scaler_e.fit(df_features[raws_e].iloc[:train_size])
    scaler_s.fit(df_features[raws_s].iloc[:train_size])
    scaler_r.fit(df_features[raws_r].iloc[:train_size])
    scaler_y.fit(df_targets[["total_load"]].iloc[:train_size])

    # 应用标准化到全部数据
    data_e = scaler_e.transform(df_features[raws_e])
    data_s = scaler_s.transform(df_features[raws_s])
    data_r = scaler_r.transform(df_features[raws_r])
    data_y = scaler_y.transform(df_targets[["total_load"]])

    # 保存 Scaler
    scaler_dir = "saved/scaler/none_transformer"
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler_e, os.path.join(scaler_dir, "scaler_e.pkl"))
    joblib.dump(scaler_s, os.path.join(scaler_dir, "scaler_s.pkl"))
    joblib.dump(scaler_r, os.path.join(scaler_dir, "scaler_r.pkl"))
    joblib.dump(scaler_y, os.path.join(scaler_dir, "scaler_y.pkl"))

    print("数据处理完成: 标准化 (均值=0, 标准差=1)")
    print(f"特征范围: [{data_e.min():.3f}, {data_e.max():.3f}]")
    print(f"特征范围: [{data_s.min():.3f}, {data_s.max():.3f}]")
    print(f"特征范围: [{data_r.min():.3f}, {data_r.max():.3f}]")
    print(f"目标范围: [{data_y.min():.3f}, {data_y.max():.3f}]")

    # 时间周期索引处理 (0-287 代表一天中 288 个 5分钟点)
    time_index = (df_features["time"].values // 5) % 288

    # 构造训练集、验证集
    train_dataset = SubwayDataset(
        data_e=data_e[:train_size], data_s=data_s[:train_size], data_r=data_r[:train_size],
        time_index=time_index[:train_size], targets=data_y[:train_size],
        time_step=48
    )

    val_dataset = SubwayDataset(
        data_e=data_e[train_size:], data_s=data_s[train_size:], data_r=data_r[train_size:],
        time_index=time_index[train_size:], targets=data_y[train_size:],
        time_step=48
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 4. model 初始化 - 使用无Transformer模型
    model = NoneTransformerModel(dim=64, time_step=48).to(device)
    print(f"采用的模型 : {model.name}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 训练
    epochs = 100
    best_val_loss = float('inf')

    # 记录训练和验证的 loss
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)

        # 验证逻辑 - 计算完整的评估指标
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_e, batch_s, batch_r, targets_batch, batch_t in val_dataloader:
                batch_e = batch_e.to(device)
                batch_s = batch_s.to(device)
                batch_r = batch_r.to(device)
                targets_batch = targets_batch.to(device)
                batch_t = batch_t.to(device)

                output = model(batch_e, batch_s, batch_r, batch_t)
                val_loss += loss_fn(output, targets_batch).item()

                # 收集预测值和真实值用于计算指标
                all_predictions.append(output)
                all_targets.append(targets_batch)

        avg_val_loss = val_loss / len(val_dataloader)

        # 计算详细评估指标 (MAE, RMSE, MAPE)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mae, rmse, mape = calculate_metrics(all_predictions, all_targets, scaler_y)

        scheduler.step(avg_val_loss)

        # 记录 loss 和指标
        train_losses.append(train_loss)
        val_losses.append(avg_val_loss)

        # 保存最优模型 (基于验证损失)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch:03d}: 保存当前最优模型 | Val Loss: {avg_val_loss:.6f} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | MAPE: {mape:.2f}%")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            print(f"  评估指标: MAE={mae:.3f} kW | RMSE={rmse:.3f} kW | MAPE={mape:.2f}%")

    # 绘制训练和验证 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('NoneTransformerModel - Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    loss_curve_path = 'saved/none_transformer_loss_curve.png'
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss 曲线已保存至: {loss_curve_path}")
    plt.show()


if __name__ == '__main__':
    main()
