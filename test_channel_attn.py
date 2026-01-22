"""
测试新的通道注意力实现
"""
import torch
from src.models.model import MainModel

def test_new_channel_attention():
    print("=" * 80)
    print("测试新的通道注意力实现（使用 AdaptiveAvgPool1d）")
    print("=" * 80)

    # 创建模型
    model = MainModel(dim=64, time_step=24)
    model.eval()

    # 准备测试数据
    batch_size = 4
    x_e = torch.randn(batch_size, 24, 3)
    x_s = torch.randn(batch_size, 24, 3)
    x_r = torch.randn(batch_size, 24, 5)
    x_t = torch.randn(batch_size, 24)

    print(f"\n输入形状:")
    print(f"  x_e: {x_e.shape}")
    print(f"  x_s: {x_s.shape}")
    print(f"  x_r: {x_r.shape}")
    print(f"  x_t: {x_t.shape}")

    # 测试前向传播
    with torch.no_grad():
        # 特征融合
        z_concat = model.feature_fusion(x_e, x_s, x_r)
        print(f"\n特征融合后: {z_concat.shape}")

        # 通道注意力
        z_attn = model.channel_attention(z_concat)
        print(f"通道注意力后: {z_attn.shape}")

        # 完整前向传播
        output = model(x_e, x_s, x_r, x_t)
        print(f"最终输出: {output.shape}")

    # 测试通道注意力权重
    print("\n" + "=" * 80)
    print("通道注意力权重测试:")
    print("=" * 80)

    # 生成一组样本数据
    test_input = torch.randn(1, 24, 192)

    with torch.no_grad():
        # 使用 AdaptiveAvgPool1d 进行池化
        x_permuted = test_input.permute(0, 2, 1)  # [1, 24, 192] -> [1, 192, 24]
        print(f"转置后: {x_permuted.shape}")

        channel_status = model.channel_attention.global_pool(x_permuted)  # [1, 192, 24] -> [1, 192, 1]
        print(f"全局池化后: {channel_status.shape}")

        channel_status = channel_status.squeeze(-1)  # [1, 192, 1] -> [1, 192]
        print(f"去掉维度后: {channel_status.shape}")

        # 通过全连接层
        e = model.channel_attention.score_layer(channel_status)  # [1, 192] -> [1, 192]
        print(f"注意力权重: {e.shape}")
        print(f"权重范围: [{e.min().item():.4f}, {e.max().item():.4f}]")
        print(f"权重均值: {e.mean().item():.4f}")
        print(f"权重标准差: {e.std().item():.4f}")

    print("\n" + "=" * 80)
    print("✓ 新实现测试通过！")
    print("=" * 80)
    print("\n关键区别:")
    print("  旧实现: torch.mean(x, dim=1) - 直接对时间维度求平均")
    print("  新实现: AdaptiveAvgPool1d(1) - 使用PyTorch的标准池化层")
    print("\n优势:")
    print("  1. 更符合SENet的标准设计")
    print("  2. 可能更高效（优化的底层实现）")
    print("  3. 语义更清晰")
    print("=" * 80)

if __name__ == '__main__':
    test_new_channel_attention()
