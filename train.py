# 训练脚本
import os
import torch.nn as nn 
import pandas as pd
import numpy as np
import torch
from src.data.pipeline import prepare_data_pipeline
from src.models.lstm import LSTM
from src.models.rnn import SimpleRNN
from src.models.fusion_model import FusionModel
from src.utils.get_config import load_config

def train(model , train_loader , val_loader , config , device) : 
    """
    训练
    
    :param model: 采用的模型
    :param train_loader: 训练数据
    :param val_loader: 验证数据
    :param config: 配置
    :param device: cpu / cuda
    """

    epochs = config["train_params"]["epoch"]
    lr = config["train_params"]["learning_rate"]
    save_path = config["train_params"]["save_path"]

    # 损失函数 与 优化器
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters() , lr = lr)
    # 学习率调整，10 次loss不下降就减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    print(f"开始训练 !!")

    len_trainLoader = len(train_loader)
    len_valLoader = len(val_loader)

    # 记录所有轮次的损失
    all_train_loss , all_val_loss = [] , []

    # 最佳loss
    best_val_loss = float('inf')

    for epoch in range(1 , epochs + 1) : 
        model.train()

        running_train_loss = 0.0
        running_val_loss = 0.0

        # 训练阶段
        for inputs , targets in train_loader : 
            inputs, targets = inputs.to(device), targets.to(device)
            # 前向传播
            output = model(inputs)
            loss = loss_fc(output , targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len_trainLoader

        # 验证阶段
        model.eval()
        with torch.no_grad() : 
            for inputs , targets in val_loader : 
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                val_loss = loss_fc(output , targets)
                running_val_loss += val_loss.item()
        
        avg_val_loss = running_val_loss / len_valLoader

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 保存当前轮次的loss
        all_train_loss.append(avg_train_loss)
        all_val_loss.append(avg_val_loss)

        # 记录最佳模型
        if avg_val_loss < best_val_loss : 
            best_val_loss = avg_val_loss
            torch.save(model.state_dict() , save_path)

        if epoch % 10 == 0 or epoch == 1 : 
            print(f"Epoch {epoch} / {epochs} | Train Loss : {avg_train_loss:.6f} | Val Loss : {avg_val_loss:.6f}")

    return all_train_loss , all_val_loss

def main() : 
    # 1.环境配置
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(config["train_params"]["save_path"]), exist_ok=True)

    # 2.数据准备
    train_loader , val_loader , test_loader , data_process = prepare_data_pipeline(config)

    # # 3.模型
    # model = LSTM(
    #     input_size = 7 , hidden_size = 32 , num_layers = 2 , output_size = 1 , dropout_rate = 0.3
    # ).to(device)

    model = FusionModel(
        input_size=7 , d_model=32 , n_head=2 , num_layers=2 , output_size=1 , dropout=0.2
    ).to(device)

    # 训练
    train_loss_list , val_loss_list = train(model , train_loader , val_loader , config , device)

    print(f"训练完成，模型保存")


if __name__ == '__main__' : 
    main()

