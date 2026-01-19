# 无周期特征模型测试
import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.dataset import DatasetProvideWeek
from torch.utils.data import DataLoader
from src.models.model import NonePeriodModel
from test_dir.eval import evaluate_lstm_baseline


def main():
    # 1. 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "saved/checkpoints/best_none_period_model.pth"
    scaler_dir = "saved/scaler/none_period"
    time_step = 24

    # 2. 数据加载
    df_features = pd.read_csv("data/processed/test_shifted.csv")
    df_targets = pd.read_csv("data/processed/test_target_shifted.csv")

    raws = ["temp", "hum", "wind", "power", "cw_temp", "chw_temp",
            "pax", "status", "fan_freq", "pump_freq"]

    # 3. 加载并应用 Scalers
    sx = joblib.load(os.path.join(scaler_dir, "scaler_x.pkl"))
    sy = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))

    data_x = sx.transform(df_features[raws])
    data_y = sy.transform(df_targets[["total_load_hvac"]])

    time_index = (df_features["time"].values // 5) % 288
    day_of_week = df_features["day_of_week"].values

    # 4. 构建测试集
    test_dataset = DatasetProvideWeek(
        data_x=data_x,
        time_index=time_index,
        day_of_week=day_of_week,
        targets=data_y,
        time_step=24
    )

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试集样本量: {len(test_dataset)}")

    # 5. 模型初始化与权重加载
    model = NonePeriodModel(dim=64, time_step=24).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"成功加载模型权重: {checkpoint_path}")
    print(f"使用的模型: {model.name}")

    # 6. 开始评估
    evaluate_none_period(model, test_dataloader, device, save_dir=scaler_dir,
                        pic_name="NonePeriodModel - No Periodic Feature Enhancement")


def evaluate_none_period(model, test_loader, device, save_dir, pic_name):
    """
    评估无周期特征模型
    1. 预测推理
    2. 反序列化 (Inverse Transform)
    3. 计算物理指标 (MAE, RMSE, MAPE, R2)
    4. 可视化

    NonePeriodModel 只需要 x (特征)，不需要周期特征
    """
    model.eval()
    all_preds = []
    all_targets = []

    scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")
    scaler_y = joblib.load(scaler_y_path)

    print("开始模型推理")
    with torch.no_grad():
        for bx, target, _, _ in test_loader:  # 忽略周期特征 (bt, bw)
            # NonePeriodModel 只需要特征 x
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
    r2 = r2_score(targets_real, preds_real)

    # 避免除以 0 的 MAPE 计算
    mape = np.mean(np.abs((targets_real - preds_real) / (targets_real + 1e-5))) * 100

    print("\n" + "="*60)
    print("📊 NonePeriodModel 评估结果")
    print("="*60)
    print(f"MAE  (平均绝对误差):      {mae:.2f} kW")
    print(f"RMSE (均方根误差):        {rmse:.2f} kW")
    print(f"MAPE (平均百分比误差):    {mape:.2f} %")
    print(f"R² 分数 (决定系数):       {r2:.4f}")
    print(f"总预测点数:               {len(targets_real)}")
    print("="*60)

    # 可视化 - 显示所有数据
    plt.figure(figsize=(20, 8))

    # 绘制真实值和预测值
    plt.plot(targets_real, label='Actual Load (真实值)', color='#1f77b4',
             linewidth=1.2, alpha=0.8)
    plt.plot(preds_real, label='Predicted Load (预测值)', color='#ff7f0e',
             linestyle='--', linewidth=1.2, alpha=0.8)

    # 填充误差区域
    plt.fill_between(range(len(targets_real)), targets_real, preds_real,
                     color='gray', alpha=0.15, label='误差区域')

    plt.title(f'{pic_name} - Full Test Set ({len(targets_real)} points)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('Cooling Load (kW)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)

    # 添加指标文本框
    textstr = f'评估指标:\n' \
              f'MAE: {mae:.2f} kW\n' \
              f'RMSE: {rmse:.2f} kW\n' \
              f'MAPE: {mape:.2f}%\n' \
              f'R²: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 保存结果图
    result_dir = "saved/results/none_period"
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, "none_period_full_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 完整对比曲线图已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
