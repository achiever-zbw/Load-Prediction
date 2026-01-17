import torch
import torch.nn as nn 


class TaskOutPutBlock(nn.Module) :
    """
    多任务输出层
    - 主任务 : 冷负荷预测
    """

    def __init__(self , dim = 64) :
        super().__init__()
        self.main_output = nn.Linear(dim , 1)

    def forward(self , h_l) :
        """
        h_l : 经过长期依赖建模的输出
        """

        y_load = self.main_output(h_l)

        return y_load