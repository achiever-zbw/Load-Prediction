"""
普通 LSTM 基线模型
用于对比实验，使用与 MainModel 相同的特征融合方式
"""
import torch
import torch.nn as nn
from src.data.dataset import SubwayLoadModel


class LSTMBaseline(nn.Module):
    """
    简单的 LSTM 基线模型
    使用与 MainModel 相同的特征融合方式
    输入: x_e, x_s, x_r (经过特征融合后的 z_concat)
    输出: 下一步的负荷预测值
    """

    def __init__(self, dim=64, hidden_dim=128, num_layers=2, dropout=0.2, time_step=24):
        """
        Args:
            dim: 特征嵌入维度 (与 MainModel 一致，64)
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            dropout: Dropout 比例
            time_step: 时间步长
        """
        super(LSTMBaseline, self).__init__()

        self.time_step = time_step
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 特征融合模块 (与 MainModel 相同)
        self.feature_fusion = SubwayLoadModel(dim, time_step)

        # LSTM 层 (输入是融合后的特征 dim * 3 = 192)
        self.lstm = nn.LSTM(
            input_size=dim * 3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_e, x_s, x_r):
        """
        Args:
            x_e: 环境特征 [B, time_step, 3]
            x_s: 系统特征 [B, time_step, 3]
            x_r: 工况特征 [B, time_step, 4]

        Returns:
            output: 预测值 [B, 1]
        """
        # 1. 特征融合 (与 MainModel 相同)
        z_concat = self.feature_fusion(x_e, x_s, x_r)  # [B, time_step, 192]

        # 2. LSTM 前向传播
        lstm_out, _ = self.lstm(z_concat)  # [B, time_step, hidden_dim]

        # 3. 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [B, hidden_dim]

        # 4. 全连接层预测
        output = self.fc(last_output)  # [B, 1]

        return output
