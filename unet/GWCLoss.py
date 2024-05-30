import torch
import torch.nn as nn
import torch.nn.functional as F
def weight_gaze(heatmap):
    # 提取红色和蓝色通道
    red_channel = heatmap[:, 0, :, :]
    blue_channel = heatmap[:, 2, :, :]

    # 计算权重，红色通道越大，权重越大；蓝色通道越大，权重越小
    weights = red_channel / (red_channel + blue_channel)
    weights = 1 + weights
    # 确保权重矩阵的值在 [1, 2] 范围内
    weights = torch.clamp(weights, 1, 2)
    # 归一化权重，确保每个样本的权重之和为1
    weights = weights / torch.sum(weights, dim=(1, 2)).view(-1, 1, 1)
    return weights
class GazeWeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GazeWeightedCrossEntropyLoss, self).__init__()
        self.CE=nn.CrossEntropyLoss()
    def forward(self, input, target,heatmap):
        # 计算交叉熵损失
        weight=weight_gaze(heatmap).unsqueeze(1)
        loss = self.CE(input*weight,target)
        return loss
