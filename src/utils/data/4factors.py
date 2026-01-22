# 负荷的四个因素 ：客流量、渗透风（出入口渗透风、屏蔽门渗透风）、照明系统、设备散热
import os
import pandas as pd
import numpy as np
from src.utils.data.classes import *

# 1. 客流量散热
def passenger_load(people_num , q) : 
    return people_num * q

# 2. 渗透风
def wind_load(G1 , h1 , G2 , h2) : 
    return (G1 * h1 + G2 * h2) * 1.2

# 3. 照明系统
def light_load(work : bool , load) : 
    if work :
        return load
    return 0

# 4. 设备散热
def equip_load(length, is_working, mean=50.0, std=2.0):
    """
    length: 数据总长度
    is_working: 运营状态序列 (Series/Array)
    mean: 运营期间设备平均功率 (kW)
    std: 波动标准差
    """
    # 生成基础的正态分布波动
    noise = np.random.normal(0, std, length)
    
    # 运营时：基准功率 + 波动
    # 非运营时：基础弱电/应急设备能耗 (假设为基准的 10%)
    base_load = is_working * mean + (1 - is_working) * (mean * 0.1)
    
    return base_load + (is_working * noise)

def factors_make() : 
    points = 288
    q = 0.182   # 人均散热量
    G1 = 18.0
    h1 = 31.7
    G2 = 7.0
    h2 = 12.5
    light_p = 120.0    # 照明系统固定负荷

    # 1. 读取人数列，已经模拟好的
    df = pd.read_csv("data/processed/2_month_data.csv")
    length = len(df)
    # print(length)

    pass_load = passenger_load(df["pax"] , q)

    # print(pass_load.head())

    # 2. 处理渗透风数据
    # 判断运营时间段（5:00-24:00运营）
    is_working = ((df["time"] >= 300) & (df["time"] < 1440)).astype(int)

    # 模拟地铁进站的开关门效应（5分钟间隔，一开一关）
    # 创建一个开关门状态数组：奇数时间点开门，偶数时间点关门
    door_status = (np.arange(length) % 2).astype(int)  # 0=关门, 1=开门

    # 出入口渗透风：持续存在，但运营期间更大
    # 开门时增加30%（乘客进出），关门时为基础值
    g1_base = G1 * (1 + door_status * 0.3)
    g1_noise = g1_base + np.random.normal(0, 1, length)

    # 屏蔽门渗透风：只在开门时产生，关门时接近0
    # 开门时为G2，关门时为G2的5%（微小泄漏）
    g2_base = G2 * door_status + G2 * 0.05 * (1 - door_status)
    g2_noise = g2_base + np.random.normal(0, 0.5, length)

    # 计算渗透风负荷（只在运营期间）
    wd_load = wind_load(g1_noise , h1 , g2_noise , h2) 
    df["wind_shen"] = g1_noise + g2_noise
    df.to_csv("data/processed/2_month_data.csv")
    print("原文件已添加渗透风")

    # 3. 处理照明系统
    lt_load = (df["time"] >= 0).astype(int) * light_p     # 0 到 5 点内不开启照明

    # 4. 设备散热
    eq_load = equip_load(length, is_working, mean=50.0 , std=2.5)

    total_load = pass_load + wd_load + lt_load + eq_load

    # 创建新的DataFrame，保存所有负荷数据
    load_df = pd.DataFrame({
        "time": df["time"],
        "passenger_load": pass_load,  # 客流负荷 (kW)
        "wind_load": wd_load,         # 渗透风负荷 (kW)
        "light_load": lt_load,        # 照明负荷 (kW)
        "equipment_load": eq_load ,    # 设备负荷 (kW)
        "total_load": total_load
    })

    # 保存到文件
    output_path = "data/processed/2_month_load_data.csv"
    load_df.to_csv(output_path, index=False)
    print(f"负荷数据已保存到: {output_path}")
    print(f"数据形状: {load_df.shape}")
    print(f"列名: {list(load_df.columns)}")
    print("\n前5行数据:")
    print(load_df.head())

    # 测试文件也需要更改
    df_test = pd.read_csv("data/processed/test_shifted.csv")
    pass_load_test = passenger_load(df_test["pax"] , q)
    # 渗透风
    door_status_test = (np.arange(len(df_test)) % 2).astype(int)  # 0=关门, 1=开门
    g1_base = G1 * (1 + door_status_test * 0.3)
    g1_noise = g1_base + np.random.normal(0, 1, len(df_test))
    g2_base = G2 * door_status_test + G2 * 0.05 * (1 - door_status_test)
    g2_noise = g2_base + np.random.normal(0, 0.5, len(df_test))
    wd_load_test = wind_load(g1_noise , h1 , g2_noise , h2) 
    df_test["wind_shen"] = g1_noise + g2_noise
    df_test.to_csv("data/processed/test_shifted.csv")
    print("测试原文件添加渗透风列")

    # 照明系统
    lt_load_test = (df_test["time"] >= 0).astype(int) * light_p
    # 4. 设备散热
    is_working_test = ((df_test["time"] >= 300) & (df_test["time"] < 1440)).astype(int)
    eq_load_test = equip_load(len(df_test), is_working_test, mean=50.0 , std=2.5)
    # 总负荷
    total_load_test = pass_load_test + wd_load_test + lt_load_test + eq_load_test
    load_df_test = pd.DataFrame({
            "time": df_test["time"],
            "passenger_load": pass_load_test,  # 客流负荷 (kW)
            "wind_load": wd_load_test,         # 渗透风负荷 (kW)
            "light_load": lt_load_test,        # 照明负荷 (kW)
            "equipment_load": eq_load_test ,    # 设备负荷 (kW)
            "total_load": total_load_test
        })
    
    output_path_test = "data/processed/test_load_data.csv"
    load_df_test.to_csv(output_path_test, index=False)
    print("测试负荷数据保存")

if __name__ == '__main__' : 
    factors_make()