import torch
import torch.nn as nn


# 伪代码示例：GLCSA模块
class GLCSA(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_att = nn.Sequential(  # 语义注意力
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att_map = self.semantic_att(x)
        return x * att_map  # 特征重加权


class CS_FPN(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample = nn.Upsample(scale_factor=2)
        self.downsample = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=2)
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 跨层特征融合

    def forward(self, x_high, x_low):
        x_up = self.upsample(x_low)
        x_fused = self.fusion_conv(torch.cat([x_high, x_up], dim=1))
        return x_fused
