import pandas as pd
import numpy as np

# 1. 读取原特征文件
file_path = "data/processed/2_month_data_cyclenet.csv"
df = pd.read_csv(file_path)

# 2. 根据行索引计算是第几天，进而计算是周几
# 每天有 288 个采样点 (24*60/5)
# df.index // 288 得到第 0, 1, 2... 天
# 第一天（index 0-287）为周一
df['day_count'] = df.index // 288
df['day_of_week'] = df['day_count'] % 7

# 3. 定义增强逻辑 (0=周一, 5=周六, 6=周日)
def boost_pax_by_index(row):
    if row['day_of_week'] >= 5:
        # 周末大幅增长 2.8 到 3.8 倍
        return int(row['pax'] * np.random.uniform(2.8, 3.8))
    return row['pax']

# 4. 执行修改
df['pax'] = df.apply(boost_pax_by_index, axis=1)

# 5. 验证结果
monday_pax = df[df['day_of_week'] == 0]['pax'].mean()
saturday_pax = df[df['day_of_week'] == 5]['pax'].mean()

print(f"--- 验证结果 ---")
print(f"总记录行数: {len(df)}")
print(f"总计天数: {len(df)//288} 天")
print(f"周一平均人流: {monday_pax:.2f}")
print(f"周六平均人流: {saturday_pax:.2f}")
print(f"增长倍数: {saturday_pax / monday_pax:.2f} 倍")

# 6. 保存回原文件，删除辅助列
df.drop(columns=['day_count', 'day_of_week']).to_csv(file_path, index=False)
print(f"--- 修改成功并覆盖原文件 ---")