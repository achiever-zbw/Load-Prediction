import pandas as pd
from datetime import datetime
import sys
import os
from src.utils.data.classes import *

def generate_one_day_data():
    points = 288
    # 1. 生成基础物理量
    df_pax = PeopleFlow(points=points, mu_idx=144, sigma_idx=50, peak_num=2000, if_excel=False, q_each=0.182, if_load=False).make()
    df_temp = TemperatureFlow(points=points, mu_idx=144, sigma_idx=36, peak_temp=32, base_temp=25).make()
    df_hum = HumFlow(points=points, mu_idx=144, sigma_idx=36, peak_hum=70, base_hum=45).make()
    df_wind = WindSpeedFlow(points=points, mu_idx=144, sigma_idx=36, peak_wind=5.0, base_wind=2.0).make()

    # 2. 生成系统关联物理量 (注意因果链)
    # R-2 启停状态 (基于客流)
    ac_status_gen = AirConditionStatus(points)
    df_status = ac_status_gen.make(df_pax["passengers"])

    # S-1 冷机功率 (基于客流)
    chiller_gen = ChillerPowerFlow(points)
    df_power = chiller_gen.make(df_pax["passengers"])

    # S-2/S-3 水温 (基于功率)
    df_cw = CoolingWaterTempFlow(points).make(df_power["chiller_power"])
    df_chw = ChilledWaterTempFlow(points).make(df_power["chiller_power"])

    # R-3/R-4 频率 (基于状态和负荷)
    df_fan = FanFrequencyFlow(points).make(df_pax["passengers"], df_status["ac_status"])
    df_pump = PumpFrequencyFlow(points).make(df_power["chiller_power"], df_status["ac_status"])

    # 3. 按照专利要求的 10 个特征进行拼接
    # E-类: temp, hum, wind_speed
    # S-类: chiller_power, cw_temp, chw_temp
    # R-类: passengers, ac_status, fan_freq, pump_freq
    
    final_df = pd.DataFrame({
        "time": df_pax["time"].dt.total_seconds().div(60).astype(int), # 转为分钟数，便于模型计算
        "temp": df_temp["temp"],
        "hum": df_hum["hum"],
        "wind": df_wind["wind_speed"],
        "power": df_power["chiller_power"],
        "cw_temp": df_cw["cw_temp"],
        "chw_temp": df_chw["chw_temp"],
        "pax": df_pax["passengers"],
        "status": df_status["ac_status"],
        "fan_freq": df_fan["fan_freq"],
        "pump_freq": df_pump["pump_freq"]
    })

    return final_df