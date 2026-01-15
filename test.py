import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.get_config import load_config
from src.data.pipeline import prepare_data_pipeline
from src.models.lstm import LSTM
from src.models.fusion_model import FusionModel
from src.models.rnn import SimpleRNN
from src.models.model_config import get_model

def main():
    # 1. 加载配置与设备
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader , data_process = prepare_data_pipeline(config)
    
    # 3. 加载模型架构
    model_type = config["model_type"]
    model = get_model(model_type).to(device)
    print(f"采用模型为 : {model_type}")
    
    # 4. 加载训练好的权重参数
    save_path = config["train_params"]["save_path"]
    model.load_state_dict(torch.load(save_path))
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(targets.numpy())

    # 合并结果
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)

    # 6. 逆标准化 转化为真实的负荷值
    preds_original = data_process.target_scaler.inverse_transform(preds)
    trues_original = data_process.target_scaler.inverse_transform(trues)

    # 7. 可视化对比，实际数据与预测值比对
    plt.figure(figsize=(12, 6))
    plt.plot(trues_original[:200], label='Actual Load', color='blue', alpha=0.7)
    plt.plot(preds_original[:200], label='Predicted Load', color='red', linestyle='--')
    plt.title('Subway AC Load Prediction - Test Result')
    plt.xlabel('Time Step')
    plt.ylabel('Load (kW)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 8. 计算评估指标
    rmse = np.sqrt(np.mean((preds_original - trues_original)**2))
    mae = np.mean(np.abs(preds_original - trues_original))
    print(f"测试集评价指标: RMSE = {rmse:.2f} kW, MAE = {mae:.2f} kW")

if __name__ == "__main__":
    main()