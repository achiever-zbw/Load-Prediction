import pandas as pd

# 读取修改后的原文件
df = pd.read_csv("data/processed/一个月数据总表_10特征.csv")

# 1. 逻辑分段：每 288 行是一天
# 第 0-287 行是周一，第 1440-1727 行是周六 (5 * 288)
monday_pax = df.iloc[0:288]['pax'].mean()
saturday_pax = df.iloc[5*288:6*288]['pax'].mean()
sunday_pax = df.iloc[6*288:7*288]['pax'].mean()

print(f"--- 采样点对比 ---")
print(f"第一天 (周一) 平均客流: {monday_pax:.2f}")
print(f"第六天 (周六) 平均客流: {saturday_pax:.2f}")
print(f"第七天 (周日) 平均客流: {sunday_pax:.2f}")
print(f"周末/工作日倍数: {saturday_pax / monday_pax:.2f} 倍")