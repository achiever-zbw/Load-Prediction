"""
通用模型评估器
支持不同输入参数的模型（MainModel需要4个输入，Baseline模型需要3个输入）
"""
import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, Optional
from pathlib import Path


class ModelEvaluator:
    """通用模型评估器"""

    def __init__(self,
                 model_name: str,
                 result_dir: str,
                 time_step: int = 24,
                 plot_length: int = 576):
        """
        初始化评估器

        Args:
            model_name: 模型名称（用于显示和保存结果）
            result_dir: 结果保存目录
            time_step: 时间步长
            plot_length: 绘图长度
        """
        self.model_name = model_name
        self.result_dir = Path(result_dir)
        self.time_step = time_step
        self.plot_length = plot_length

        # 创建结果目录
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _model_forward(self, model: nn.Module, batch_data: Tuple, device: torch.device):
        """
        智能前向传播 - 根据模型类型自动适配输入参数

        Args:
            model: 模型
            batch_data: 批次数据 (be, bs, br, bt, target) 或 (be, bs, br, bt, target)
            device: 设备

        Returns:
            模型输出
        """
        be, bs, br, bt, target = batch_data

        # 将数据移到设备
        be = be.to(device)
        bs = bs.to(device)
        br = br.to(device)
        bt = bt.to(device)

        try:
            # 尝试4参数输入 (MainModel)
            output = model(be, bs, br, bt)
        except TypeError as e:
            # 如果失败，尝试3参数输入 (LSTM/RNN Baseline)
            if "positional argument" in str(e) or "missing 1 required positional argument" in str(e):
                print(f"  检测到3输入模型，使用3参数调用")
                output = model(be, bs, br)
            else:
                raise e

        return output

    def evaluate(self,
                 model: nn.Module,
                 test_loader,
                 device: torch.device,
                 scaler_dir: str) -> Dict[str, float]:
        """
        评估模型

        Args:
            model: 待评估模型
            test_loader: 测试数据加载器
            device: 设备
            scaler_dir: 标准化器目录

        Returns:
            指标字典 {'mae': ..., 'rmse': ..., 'mape': ...}
        """
        model.eval()
        all_preds = []
        all_targets = []

        # 加载标准化器
        scaler_y_path = os.path.join(scaler_dir, "scaler_y.pkl")
        scaler_y = joblib.load(scaler_y_path)

        print(f"\n{'='*60}")
        print(f"开始评估模型: {self.model_name}")
        print(f"{'='*60}")

        # 推理
        with torch.no_grad():
            for batch_data in test_loader:
                output = self._model_forward(model, batch_data, device)
                target = batch_data[4]  # target是第5个元素

                all_preds.append(output.cpu().numpy())
                all_targets.append(target.numpy())

        # 拼接并降维
        preds_norm = np.concatenate(all_preds, axis=0).reshape(-1, 1)
        targets_norm = np.concatenate(all_targets, axis=0).reshape(-1, 1)

        # 转化为物理负荷值
        preds_real = scaler_y.inverse_transform(preds_norm).flatten()
        targets_real = scaler_y.inverse_transform(targets_norm).flatten()

        # 计算指标
        metrics = self._calculate_metrics(targets_real, preds_real)

        # 打印指标
        self._print_metrics(metrics)

        # 绘制对比图
        self._plot_comparison(targets_real, preds_real)

        # 保存指标到文件
        self._save_metrics(metrics)

        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        # 避免除以0的MAPE计算
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    def _print_metrics(self, metrics: Dict[str, float]):
        """打印指标"""
        print(f"\n【评估指标】")
        print(f"  MAE  (平均绝对误差): {metrics['mae']:.2f} kW")
        print(f"  RMSE (均方根误差):   {metrics['rmse']:.2f} kW")
        print(f"  MAPE (平均百分比误差): {metrics['mape']:.2f} %")

    def _plot_comparison(self, y_true: np.ndarray, y_pred: np.ndarray):
        """绘制预测对比图"""
        plot_len = min(self.plot_length, len(y_true))

        plt.figure(figsize=(15, 6))
        plt.plot(y_true[:plot_len], label='Actual Load', color='#1f77b4', linewidth=1.5)
        plt.plot(y_pred[:plot_len], label='Predicted Load', color='#ff7f0e', linestyle='--', linewidth=1.5)
        plt.fill_between(range(plot_len), y_true[:plot_len], y_pred[:plot_len], color='gray', alpha=0.2)

        plt.title(f'{self.model_name} - Subway HVAC Load Prediction', fontsize=14)
        plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
        plt.ylabel('Cooling Load (kW)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图像
        save_path = self.result_dir / 'test_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[✓] 对比曲线图已保存至: {save_path}")
        plt.close()

    def _save_metrics(self, metrics: Dict[str, float]):
        """保存指标到文本文件"""
        metrics_path = self.result_dir / 'metrics.txt'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"模型: {self.model_name}\n")
            f.write(f"时间步长: {self.time_step}\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"MAE  (平均绝对误差): {metrics['mae']:.4f} kW\n")
            f.write(f"RMSE (均方根误差):   {metrics['rmse']:.4f} kW\n")
            f.write(f"MAPE (平均百分比误差): {metrics['mape']:.4f} %\n")

        print(f"[✓] 指标已保存至: {metrics_path}")


# 向后兼容的函数接口（保持与旧代码兼容）
def evaluate(model, test_loader, device, save_dir, model_name="Model"):
    """
    兼容旧版本的evaluate函数
    自动检测模型输入参数并调用
    """
    import os
    result_dir = os.path.join("saved/results", model_name.lower().replace(" ", "_"))

    evaluator = ModelEvaluator(
        model_name=model_name,
        result_dir=result_dir
    )

    return evaluator.evaluate(model, test_loader, device, save_dir)
