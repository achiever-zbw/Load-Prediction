import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Subset
from src.models.model import MainModel
from src.data.dataset import SubwayDataset
import random
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def train_one_epoch(model , train_loader , loss_fn , optimizer , device) :
    model.train()
    running_loss = 0.0

    # 加载数据集
    for batch_e , batch_s , batch_r , batch_t , targets in train_loader : 
        batch_e, batch_s = batch_e.to(device), batch_s.to(device)
        batch_r, batch_t = batch_r.to(device), batch_t.to(device)
        targets = targets.to(device)

        # 前向传播
        main_output = model(batch_e , batch_s , batch_r , batch_t)
        loss_main = loss_fn(main_output , targets)

        # 反向传播
        optimizer.zero_grad()
        loss_main.backward()

        # 梯度裁剪：防止 LSTM 梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        running_loss += loss_main.item()
    
    return running_loss / len(train_loader)


def main() : 
    # 1. 配置与环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "saved/checkpoints/best_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. 数据准备
    # 使用shifted后的特征数据(8639行)与shifted负荷数据完全对齐
    features_file_path = "data/processed/一个月数据总表_10特征_shifted.csv"
    target_file_path = "data/processed/一个月总负荷数据_shifted.csv"

    df_features = pd.read_csv(features_file_path)
    df_targets = pd.read_csv(target_file_path)

    # 特征列
    raws_e = ["temp", "hum", "wind"]
    raws_s = ["power", "cw_temp", "chw_temp"]
    raws_r = ["pax", "status", "fan_freq", "pump_freq"]
    
    # 划分训练集 验证集
    total_len = len(df_features)
    train_size = int(0.8 * total_len)

    # 归一化处理
    scaler_e = StandardScaler()
    scaler_s = StandardScaler()
    scaler_r = StandardScaler()
    scaler_y = StandardScaler()

    # 注意，只在训练集上 fit ！
    scaler_e.fit(df_features[raws_e].iloc[:train_size])
    scaler_s.fit(df_features[raws_s].iloc[:train_size])
    scaler_r.fit(df_features[raws_r].iloc[:train_size])
    scaler_y.fit(df_targets[["total_load_hvac"]].iloc[:train_size])
    # 保存 Scaler (未来推理时必须使用训练时的均值和方差)
    joblib.dump(scaler_e, os.path.join("saved/scaler", "scaler_e.pkl"))
    joblib.dump(scaler_y, os.path.join("saved/scaler", "scaler_y.pkl"))
    joblib.dump(scaler_s, os.path.join("saved/scaler", "scaler_s.pkl"))
    joblib.dump(scaler_r, os.path.join("saved/scaler", "scaler_r.pkl"))

    # 应用transform到全局数据
    data_e = scaler_e.transform(df_features[raws_e])
    data_s = scaler_s.transform(df_features[raws_s])
    data_r = scaler_r.transform(df_features[raws_r])
    data_y = scaler_y.transform(df_targets[["total_load_hvac"]])

    # 时间周期索引处理 (0-287 代表一天中 288 个 5分钟点)
    time_index = (df_features["time"].values // 5) % 288

    # 5. 实例化 Dataset (此时 Dataset 接收的是 numpy 数组)
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

    # 4. model 初始化
    model = MainModel(dim=64 , time_step=24).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters() , lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer , 'min' , patience=5 , factor=0.5)

    # 训练
    epochs = 100
    best_val_loss = float('inf')

    # 记录训练和验证的 loss
    train_losses = []
    val_losses = []

    for epoch in range(1 , epochs + 1) :
        train_loss = train_one_epoch(model , train_dataloader , loss_fn , optimizer , device)

        # 验证逻辑s
        model.eval()
        val_loss = 0.0
        with torch.no_grad() :
            for be , bs , br , bt , targets in val_dataloader :
                be = be.to(device)
                bs = bs.to(device)
                br = br.to(device)
                bt = bt.to(device)
                output = model(be , bs , br , bt)
                val_loss += loss_fn(output , targets.to(device)).item()

        avg_val_loss = val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)

        # 记录 loss
        train_losses.append(train_loss)
        val_losses.append(avg_val_loss)

        # 保存最优模型
        if avg_val_loss < best_val_loss :
            best_val_loss = avg_val_loss
            torch.save(model.state_dict() , save_path)
            print(f"Epoch {epoch:03d}: 保存当前最优模型 | Val Loss: {avg_val_loss:.6f}")

        if epoch % 5 == 0 or epoch == 1 :
            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 绘制训练和验证 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    loss_curve_path = 'saved/loss_curve.png'
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss 曲线已保存至: {loss_curve_path}")
    plt.show()

if __name__ == '__main__' : 
    main()