import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# E-1 温度
class TemperatureFlow:
    """ E-1  
    温度模拟--正态分布（整数，无随机扰动）
    """

    def __init__(self, points, mu_idx, sigma_idx, peak_temp, base_temp=25, if_excel=False):
        """
        points: 时间点数量
        mu_idx: 温度峰值索引
        sigma_idx: 峰值标准差
        peak_temp: 峰值温度
        base_temp: 基础温度
        if_excel: 是否导出 Excel
        """
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.peak_temp = peak_temp
        self.base_temp = base_temp
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)

        # 正态分布公式生成温度
        temp = self.base_temp + (self.peak_temp - self.base_temp) * np.exp(
            -(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        temp = temp.astype(int)  # 转为整数

        df = pd.DataFrame({
            "time": time_list,
            "temp": temp
        })

        if self.if_excel:
            df.to_excel("temperature.xlsx", index=False)
        return df

# E-2 湿度
class HumFlow:
    """E-2
    湿度模拟--正态分布（整数，无随机扰动）"""

    def __init__(self, points, mu_idx, sigma_idx, peak_hum, base_hum, if_excel=False):
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.peak_temp = peak_hum
        self.base_temp = base_hum
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)

        # 正态分布公式生成温度
        hum = self.base_temp + (self.peak_temp - self.base_temp) * np.exp(
            -(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        hum = hum.astype(int)  # 转为整数

        df = pd.DataFrame({
            "time": time_list,
            "hum": hum
        })

        if self.if_excel:
            df.to_excel("hum.xlsx", index=False)
        return df

# E-3 风速
class WindSpeedFlow:
    """ E-3
    风速模拟--基于基础风速与波动"""

    def __init__(self, points, mu_idx, sigma_idx, peak_wind, base_wind=2.0, if_excel=False):
        """
        points: 时间点数量
        mu_idx: 风速峰值索引
        sigma_idx: 波动范围标准差
        peak_wind: 最大风速 (m/s)
        base_wind: 基础平均风速 (m/s)
        """
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.peak_wind = peak_wind
        self.base_wind = base_wind
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)

        # 1. 生成基础正态分布趋势
        # 模拟风速随时间（如昼夜温差带来的气流变化）的波动
        wind_speed = self.base_wind + (self.peak_wind - self.base_wind) * np.exp(
            -(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        
        # 2. 加入微小的随机阵风扰动（使数据更真实，符合专利中的多源异构噪声）
        noise = np.random.normal(0, 0.2, self.points) 
        wind_speed = np.abs(wind_speed + noise)  # 确保风速不为负值
        
        # 保留两位小数（风速通常不取整，需要精度）
        wind_speed = np.round(wind_speed, 2)

        df = pd.DataFrame({
            "time": time_list,
            "wind_speed": wind_speed
        })

        if self.if_excel:
            df.to_excel("wind_speed.xlsx", index=False)
        return df

# S-1 冷机功率
class ChillerPowerFlow:
    """S-1: 冷机功率模拟"""
    def __init__(self, points, base_load=5000):
        self.points = points
        self.base_load = base_load

    def make(self, pax_series):
        """
        pax_series: 传入客流量序列，用于计算变频功率
        """
        # 基础功率 + 客流带来的变动负荷 (假设每人增加50W功耗)
        power = self.base_load + (pax_series * 50)
        # 加入运行过程中的高频随机扰动
        noise = np.random.normal(0, 200, self.points)
        chiller_power = np.abs(power + noise)
        
        return pd.DataFrame({
            "chiller_power": chiller_power.astype(float).round(2)
        })

# S-2 冷却水温度
class CoolingWaterTempFlow:
    """S-2: 冷却水温度模拟"""
    def __init__(self, points, base_temp=30.0):
        self.points = points
        self.base_temp = base_temp

    def make(self, power_series):
        """
        power_series: 传入冷机功率序列，功率越高，散热温升越高
        """
        # 冷却水温度 = 基础温度 (受环境影响) + 散热增量
        # 假设功率每增加10kW，冷却水温升 0.5度
        temp_rise = (power_series / 10000) * 0.5
        cw_temp = self.base_temp + temp_rise + np.random.normal(0, 0.1, self.points)
        
        return pd.DataFrame({
            "cw_temp": cw_temp.round(2)
        })
    
# S-3 冷冻水温度
class ChilledWaterTempFlow:
    """S-3
    冷冻水温度模拟"""
    def __init__(self, points, set_point=7.0):
        self.points = points
        self.set_point = set_point

    def make(self, power_series):
        """
        power_series: 传入冷机功率序列，高功率时冷冻水温会有小幅波动
        """
        # 负荷极高时，冷冻水温会略微偏离设定点 (由于换热滞后)
        max_p = power_series.max()
        deviation = (power_series / max_p) * 0.4
        chw_temp = self.set_point + deviation + np.random.normal(0, 0.05, self.points)
        
        return pd.DataFrame({
            "chw_temp": chw_temp.round(2)
        })
    
# R-1 客流量
class PeopleFlow:
    """ R-1
    人流量模拟--正态分布"""

    def __init__(self, points, mu_idx, sigma_idx, peak_num, if_excel, q_each, if_load):
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.peak_num = peak_num
        self.if_excel = if_excel
        self.q_each = q_each
        self.if_load = if_load

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)
        passengers = self.peak_num * np.exp(-(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        passengers = passengers.astype(int)

        load = passengers * self.q_each
        if self.if_load:
            df = pd.DataFrame({
                "time": time_list, "passengers": passengers, "passengers_load": load
            })
        else:
            df = pd.DataFrame({
                "time": time_list, "passengers": passengers
            })
        return df

    def figure(self):
        df = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["passengers"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Passengers")
        plt.grid(True)
        plt.show()

# R-2 空调设备启停
class AirConditionStatus:
    """R-2: 空调设备启停状态模拟 (0-关, 1-开)"""
    def __init__(self, points, start_idx=72, end_idx=264): 
        """
        start_idx: 开启时间点 (默认早上6点)
        end_idx: 关闭时间点 (默认晚上10点)
        """
        self.points = points
        self.start_idx = start_idx
        self.end_idx = end_idx

    def make(self, pax_series):
        # 逻辑：在运营时段内，且客流大于一定阈值时开启
        status = np.zeros(self.points)
        for i in range(self.points):
            if self.start_idx <= i <= self.end_idx or pax_series[i] > 100:
                status[i] = 1
            else:
                status[i] = 0
        
        return pd.DataFrame({"ac_status": status.astype(int)})

# R-3 风机频率
class FanFrequencyFlow:
    """R-3: 
    风机频率模拟 (Hz)"""
    def __init__(self, points, min_freq=30.0, max_freq=50.0):
        self.points = points
        self.min_freq = min_freq
        self.max_freq = max_freq

    def make(self, pax_series, status_series):
        """
        根据客流比例调节频率，若状态为关，则频率为0
        """
        max_pax = pax_series.max() if pax_series.max() > 0 else 1
        # 变频逻辑：基础频率 + 随客流波动的频率
        freq = self.min_freq + (self.max_freq - self.min_freq) * (pax_series / max_pax)
        # 加入控制波动误差
        freq = freq + np.random.normal(0, 0.5, self.points)
        # 联动启停状态
        freq = np.where(status_series == 1, freq, 0)
        
        return pd.DataFrame({"fan_freq": np.round(freq, 2)})

# R-4 水泵频率
class PumpFrequencyFlow:
    """R-3: 水泵频率模拟 (Hz)"""
    def __init__(self, points, min_freq=35.0, max_freq=50.0):
        self.points = points
        self.min_freq = min_freq
        self.max_freq = max_freq

    def make(self, power_series, status_series):
        """
        根据冷机功率调节水泵频率
        """
        max_p = power_series.max() if power_series.max() > 0 else 1
        # 变频逻辑：负载越高，频率越高
        freq = self.min_freq + (self.max_freq - self.min_freq) * (power_series / max_p)
        freq = freq + np.random.normal(0, 0.3, self.points)
        # 联动启停状态
        freq = np.where(status_series == 1, freq, 0)
        
        return pd.DataFrame({"pump_freq": np.round(freq, 2)})



class EquipHeap:
    """设备散热量"""

    def __init__(self, points, q_equip, if_excel=False):
        self.points = points
        self.q_equip = q_equip
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        equip_heap = np.full(self.points, self.q_equip)
        df_equip = pd.DataFrame({
            "time": time_list,
            "equip_heat": equip_heap
        })
        return df_equip

    def figure(self):
        df_equip = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df_equip["time"], df_equip["equip_heat"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Equip_heat (W)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class StructureLoad:
    """地铁结构热负荷模拟"""

    def __init__(self, points, q_structure, if_excel=False):
        self.points = points
        self.q_structure = q_structure
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        structure_load = np.full(self.points, self.q_structure)
        df_structure = pd.DataFrame({
            "time": time_list,
            "structure_load": structure_load
        })
        return df_structure

    def figure(self):
        df_structure = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df_structure["time"], df_structure["structure_load"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Structure Load (W)")
        plt.title("Subway Structure Heat Load Simulation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class VentilationLoad:
    """渗透风热负荷模拟--交替模式"""

    def __init__(self, points, q_vent, period=1, if_excel=False):
        self.points = points
        self.q_vent = q_vent
        self.period = period
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        # vent_load = np.array([(self.q_vent if (i // self.period) % 2 == 0 else 0) for i in range(self.points)])
        vent_load = []
        for i in range(self.points):
            if (i % 2):
                vent_load.append(0)
            else:
                vent_load.append(self.q_vent)

        df_vent = pd.DataFrame({
            "time": time_list,
            "vent_load": vent_load
        })
        return df_vent

    def figure(self):
        df_vent = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df_vent["time"], df_vent["vent_load"], marker="+")
        plt.xlabel("Time")
        plt.ylabel("Vent_load (W)")
        plt.title("Ventilation Heat Load Simulation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()






class EquipNum:
    """冷水机组数量模拟--正态分布或峰值波动（整数）"""

    def __init__(self, points, mu_idx, sigma_idx, min_num=1, max_num=5, if_excel=False):
        self.points = points
        self.mu_idx = mu_idx
        self.sigma_idx = sigma_idx
        self.min_num = min_num
        self.max_num = max_num
        self.if_excel = if_excel

    def make(self):
        time_list = [timedelta(minutes=5 * i) for i in range(self.points)]
        x = np.arange(self.points)

        # 正态分布生成机组数量
        num = self.min_num + (self.max_num - self.min_num) * np.exp(-(x - self.mu_idx) ** 2 / (2 * self.sigma_idx ** 2))
        num = np.round(num).astype(int)

        df = pd.DataFrame({
            "time": time_list,
            "equip_num": num
        })

        if self.if_excel:
            df.to_excel("equip_num.xlsx", index=False)
        return df

    def figure(self):
        df = self.make()
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["equip_num"], marker="+", color="green")
        plt.xlabel("Time")
        plt.ylabel("Chiller Number")
        plt.grid(True)
        plt.show()
