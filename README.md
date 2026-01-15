# 文件说明

### 数据

- [一个月数据](data/processed/一个月数据总表_10特征.csv)

### 数据处理

- [原始数据标准化 与 窗口构建](src/data/dataset.py)
- [通道嵌入、加权求和、特征拼接](src/data/channel_embedding.py)

### 模型搭建
- [通道注意力层](src/models/attention.py)
- [短时序建模层](src/models/lstm.py)
- [周期特征建模](src/models/period.py)

### 测试文件
- 测试各个模型搭建的流程中数据形状 [test.py](src/models/test.py)