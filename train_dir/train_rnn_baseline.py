"""
普通 RNN 基线模型训练脚本
用于对比实验，使用与 MainModel 相同的数据处理方式
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

from src.models.rnn import RNNBaseline
from src.data.dataset import SubwayDataset


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0

    for batch_e, batch_s, batch_r, batch_t, targets in train_loader:
        batch_e = batch_e.to(device)
        batch_s = batch_s.to(device)
        batch_r = batch_r.to(device)
        targets = targets.to(device)

        # 前向传播 (模型内部会进行特征融合，与 MainModel 相同)
        output = model(batch_e, batch_s, batch_r)
        loss = loss_fn(output, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, val_loader, loss_fn, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_e, batch_s, batch_r, batch_t, targets in val_loader:
            batch_e = batch_e.to(device)
            batch_s = batch_s.to(device)
            batch_r = batch_r.to(device)
            targets = targets.to(device)

            output = model(batch_e, batch_s, batch_r)
            val_loss += loss_fn(output, targets).item()

    return val_loss / len(val_loader)


def main():
    # 1. 配置与环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "saved/checkpoints/rnn_baseline_best.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"使用设备: {device}")

    # 2. 数据准备 (与 train_better.py 完全一致)
    features_file_path = "data/processed/一个月数据总表_10特征_shifted.csv"
    target_file_path = "data/processed/一个月总负荷数据_shifted.csv"

    df_features = pd.read_csv(features_file_path)
    df_targets = pd.read_csv(target_file_path)

    # 特征列
    raws_e = ["temp", "hum", "wind"]
    raws_s = ["power", "cw_temp", "chw_temp"]
    raws_r = ["pax", "status", "fan_freq", "pump_freq"]

    # 划分训练集/验证集
    total_len = len(df_features)
    train_size = int(0.8 * total_len)

    # 归一化处理 (与 MainModel 完全一致)
    scaler_e = StandardScaler()
    scaler_s = StandardScaler()
    scaler_r = StandardScaler()
    scaler_y = StandardScaler()

    scaler_e.fit(df_features[raws_e].iloc[:train_size])
    scaler_s.fit(df_features[raws_s].iloc[:train_size])
    scaler_r.fit(df_features[raws_r].iloc[:train_size])
    scaler_y.fit(df_targets[["total_load_hvac"]].iloc[:train_size])

    # 保存 scaler
    os.makedirs("saved/scaler/rnn", exist_ok=True)
    joblib.dump(scaler_e, "saved/scaler/rnn/scaler_e.pkl")
    joblib.dump(scaler_s, "saved/scaler/rnn/scaler_s.pkl")
    joblib.dump(scaler_r, "saved/scaler/rnn/scaler_r.pkl")
    joblib.dump(scaler_y, "saved/scaler/rnn/scaler_y.pkl")

    # 应用 transform
    data_e = scaler_e.transform(df_features[raws_e])
    data_s = scaler_s.transform(df_features[raws_s])
    data_r = scaler_r.transform(df_features[raws_r])
    data_y = scaler_y.transform(df_targets[["total_load_hvac"]])

    # 时间周期索引
    time_index = (df_features["time"].values // 5) % 288

    # 3. 创建数据集 (与 train_better.py 完全一致)
    train_dataset = SubwayDataset(
        data_e[:train_size], data_s[:train_size], data_r[:train_size],
        time_index[:train_size], data_y[:train_size], time_step=24
    )
    val_dataset = SubwayDataset(
        data_e[train_size:], data_s[train_size:], data_r[train_size:],
        time_index[train_size:], data_y[train_size:], time_step=24
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 4. 模型初始化
    model = RNNBaseline(
        input_size=64 , 
        hidden_size=128 , 
        num_layers=2 , 
        dropout_rate=0.2 , 
        time_step=24
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )

    # 5. 训练
    epochs = 200
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\n开始训练 RNN 基线模型...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)

        # 验证
        val_loss = validate(model, val_dataloader, loss_fn, device)
        scheduler.step(val_loss)

        # 记录 loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch:03d}: 保存最优模型 | Val Loss: {val_loss:.6f}")

        # 打印训练信息
        if epoch % 5 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch:03d}/{epochs}] | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

    print("=" * 60)
    print(f"训练完成! 最优验证损失: {best_val_loss:.6f}")
    print(f"模型已保存至: {save_path}")

    # 6. 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('RNN Baseline - Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_curve_path = 'saved/RNN_baseline_loss_curve.png'
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss 曲线已保存至: {loss_curve_path}")
    plt.show()


if __name__ == '__main__':
    main()
