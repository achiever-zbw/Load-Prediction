# 处理时间列
def time_to_minutes(time_val) : 
    hours , minutes = map(int , time_val.split(":"))
    return hours * 60 + minutes