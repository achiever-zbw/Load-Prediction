# 获取配置信息
import yaml

def load_config(config_path = "configs/data_config.yaml") :
    with open(config_path , 'r' , encoding='utf-8') as f :
        config = yaml.safe_load(f)
    return config

# 测试
if __name__ == '__main__' : 
    config = load_config("configs/data_config.yaml")
    time_steps = config["sequence_params"]["time_steps"]

    print(f"检测到步长为 : {time_steps}")