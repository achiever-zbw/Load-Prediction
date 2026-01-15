import pandas as pd
from datetime import datetime, timedelta
import os
# 确保你的路径正确
from src.utils.data.one_day_data import generate_one_day_data

def get_all_data(start_date="2025-06-01", days=30):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    all_data = []

    print("=" * 50)
    print(f"开始生成从 {start_date} 开始的 {days} 天模拟数据...")

    for day in range(days):
        current_date = start_dt + timedelta(days=day)
        # 调用你刚刚写好的 generate_one_day_data()
        # 它现在返回的是包含 10 个特征 + time 的 DataFrame
        daily_data = generate_one_day_data()
        
        # 可选：如果你想在数据中保留日期信息，可以取消下面这行的注释
        # daily_data.insert(0, 'date', current_date.strftime("%Y-%m-%d"))

        all_data.append(daily_data)

    # 合并所有天的数据
    month_data = pd.concat(all_data, ignore_index=True)

    # --- 关键修改：统一 10 个特征的列顺序 ---
    # 这里的顺序必须和你 generate_one_day_data() 返回的列名完全一致
    columns_order = [
        "time",       # 时间索引 (0, 5, 10...)
        "temp",       # E-1
        "hum",        # E-2
        "wind",       # E-3
        "power",      # S-1
        "cw_temp",    # S-2
        "chw_temp",   # S-3
        "pax",        # R-1
        "status",     # R-2
        "fan_freq",   # R-3
        "pump_freq"   # R-4
    ]
    
    # 检查生成的列是否完整，防止 merge 逻辑导致列缺失
    existing_cols = [col for col in columns_order if col in month_data.columns]
    month_data = month_data[existing_cols]

    return month_data

def main():
    # 确保输出目录存在
    output_dir = "data/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成数据
    days_to_gen = 30
    month_data = get_all_data(start_date="2025-06-01", days=days_to_gen)

    print(f"\n数据生成完成")
    print(f"总记录数: {len(month_data)} (应为 {days_to_gen} * 288 = {days_to_gen * 288})")
    print(f"特征总数: {len(month_data.columns) - 1}") # 减去 time 列
    print(f"最终列清单: {list(month_data.columns)}")

    # 保存文件
    # 建议同时保存 CSV 和 Excel，因为数据量大时 Excel 打开很慢
    csv_output = os.path.join(output_dir, "一个月数据总表_10特征.csv")
    excel_output = os.path.join(output_dir, "一个月数据总表_10特征.xlsx")

    month_data.to_csv(csv_output, index=False, encoding="utf-8-sig")
    month_data.to_excel(excel_output, index=False)
    
    print(f"\n文件已成功保存:")
    print(f"{csv_output}")

if __name__ == "__main__":
    main()