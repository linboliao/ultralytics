import torch

import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Softmax
from einops import rearrange, repeat

__all__ = (
    "FrequencyAttention",
    "MultiScalConv",
    "BoundaryAttention",
)

from ultralytics.nn.modules import Conv, CBAM, ChannelAttention, A2C2f
from ultralytics.nn.modules.block import C3k2


# class FrequencyAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super().__init__()
#         self.in_channels = in_channels
#         self.reduction_ratio = reduction_ratio
#
#         self.attention_net = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels // reduction_ratio, kernel_size=1),
#             nn.BatchNorm2d(in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         original_dtype = x.dtype
#
#         x = x.to(torch.float32)
#
#         B, C, H, W = x.shape
#         x_fft = torch.fft.rfft2(x, norm='ortho')
#
#         real = x_fft.real
#         imag = x_fft.imag
#         real = real.to(original_dtype)
#         imag = imag.to(original_dtype)
#
#         combined = torch.cat([real, imag], dim=1)
#         attention_weights = self.attention_net(combined)
#
#         real_weighted = real * attention_weights
#         imag_weighted = imag * attention_weights
#         real_weighted = real_weighted.to(torch.float32)
#         imag_weighted = imag_weighted.to(torch.float32)
#         x_fft_weighted = torch.complex(real_weighted, imag_weighted)
#
#         output = torch.fft.irfft2(x_fft_weighted, s=(H, W), norm='ortho')
#
#         return output.to(original_dtype)


def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor


class FrequencyAttention(nn.Module):
    def __init__(self, c1):
        super(FrequencyAttention, self).__init__()

        down_dim = c1 // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.temperature = nn.Parameter(torch.ones(8, 1, 1))

        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 16, 1, bias=True),
            nn.BatchNorm2d(down_dim // 16),
            nn.ReLU(True),
            nn.Conv2d(down_dim // 16, down_dim, 1, bias=True),
            nn.Sigmoid())

        self.softmax = Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(down_dim)
        self.relu = nn.ReLU(True)
        self.num_heads = 8

    def forward(self, x):
        conv2 = self.conv2(x)
        b, c, h, w = conv2.shape

        q_f_2 = torch.fft.fft2(conv2.float())
        k_f_2 = torch.fft.fft2(conv2.float())
        v_f_2 = torch.fft.fft2(conv2.float())
        tepqkv = torch.fft.fft2(conv2.float())

        q_f_2 = rearrange(q_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f_2 = rearrange(k_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f_2 = rearrange(v_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f_2 = torch.nn.functional.normalize(q_f_2, dim=-1)
        k_f_2 = torch.nn.functional.normalize(k_f_2, dim=-1)
        attn_f_2 = (q_f_2 @ k_f_2.transpose(-2, -1)) * self.temperature
        attn_f_2 = custom_complex_normalization(attn_f_2, dim=-1)
        out_f_2 = torch.abs(torch.fft.ifft2(attn_f_2 @ v_f_2))
        out_f_2 = rearrange(out_f_2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l_2 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))
        out_2 = torch.cat((out_f_2, out_f_l_2), 1)

        F_2 = torch.add(out_2, x)

        return F_2


def patches(x):
    """
    处理输入张量：切割、调整大小并拼接。

    参数:
        x (torch.Tensor): 输入张量，形状为 (b, 3, h, w)。

    返回:
        torch.Tensor: 输出张量，形状为 (b, 15, h/2, w/2)。
    """
    b, c, h, w = x.shape
    assert c == 3, "输入张量应有3个通道"

    h_half = h // 2
    w_half = w // 2

    top_left = x[:, :, :h_half, :w_half]
    top_right = x[:, :, :h_half, w_half:]
    bottom_left = x[:, :, h_half:, :w_half]
    bottom_right = x[:, :, h_half:, w_half:]

    patches = [top_left, top_right, bottom_left, bottom_right]

    return patches


# class MultiScalConv(nn.Module):
#     """
#     分块处理图片降低参数
#     """
#
#     # TODO 实现多层多尺度代码
#     def __init__(self, c1, c2):
#         super().__init__()
#         self.img_conv = Conv(c1, c2, 7, 2, d=2)
#         self.patch_conv = Conv(c1, c2, 3, 1)
#         self.channel_attention = ChannelAttention(c2 * 5)
#         self.feature_conv = Conv(c2 * 5, c2, 1, 1)
#
#     def forward(self, x):
#         img_feat = self.img_conv(x)
#         patch_feats = []
#         for patch in patches(x):
#             patch_feats.append(self.patch_conv(patch))
#         feats = torch.cat(patch_feats + [img_feat], 1)
#         # return self.feature_conv(self.channel_attention(feats))
#         return self.feature_conv(feats)


# - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
# - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
# - [-1, 2, C3k2, [256, False, 0.25]]
# - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
# - [-1, 2, C3k2, [512, False, 0.25]]
# - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
# - [-1, 4, A2C2f, [512, True, 4]]
# - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
# - [-1, 4, A2C2f, [1024, True, 1]]  # 8
class MultiScalConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.img_conv = Conv(c1, c2, 5, 2, d=2)
        self.patch_conv = Conv(c1, c2, 3, 1)

    def forward(self, x):
        img_feat = self.img_conv(x)
        patch_feats = []
        for patch in patches(x):
            patch_feats.append(self.patch_conv(patch))
        return torch.cat(patch_feats + [img_feat], 1)


class CustomConv(Conv):
    def forward(self, x):
        x_split = torch.chunk(x, chunks=5, dim=1)

        parts = []
        for x_part in x_split:
            parts.append(self.act(self.bn(self.conv(x_part))))

        return torch.cat(parts, dim=1)


class CustomC3k2(C3k2):
    def forward(self, x):
        x_split = torch.chunk(x, chunks=5, dim=1)
        parts = []
        for x_part in x_split:
            y = list(self.cv1(x_part).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            parts.append(self.cv2(torch.cat(y, 1)))
        return torch.cat(parts, dim=1)


class CustomA2C2f(A2C2f):
    def forward(self, x):
        x_split = torch.chunk(x, chunks=5, dim=1)
        parts = []
        for x_part in x_split:
            y = [self.cv1(x_part)]
            y.extend(m(y[-1]) for m in self.m)
            y = self.cv2(torch.cat(y, 1))
            if self.gamma is not None:
                y = x_part + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
            parts.append(y)
        return torch.cat(parts, dim=1)


class CustomChannelAttention(ChannelAttention):
    def __init__(self, channels: int):
        super().__init__(channels * 5)
        self.feature_conv = Conv(channels * 5, channels, 1, 1)

    def forward(self, x):
        return self.feature_conv(x * self.act(self.fc(self.pool(x))))


class BoundaryAttention(nn.Module):
    """
    边界增强模块 (BEC)：一个轻量级模块，用于增强特征图中的边界信息，特别适用于医学图像分割中轮廓模糊的目标。
    该模块通过一个边界预测头生成边界概率图，并通过残差连接将边界信息与原始特征融合。
    """

    def __init__(self, c1):
        super(BoundaryAttention, self).__init__()
        self.sobel_conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_conv_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_conv_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

        self.boundary = nn.Sequential(
            Conv(c1, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Conv(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Conv(32, 1, 1),
            nn.Sigmoid()
        )

        self.channel_align = nn.Conv2d(1, c1, kernel_size=1, bias=True)
        self.shortcut = nn.Identity()

    def forward(self, x):
        gray_x = torch.mean(x, dim=1, keepdim=True)
        grad_x = self.sobel_conv_x(gray_x)
        grad_y = self.sobel_conv_y(gray_x)

        sobel_feat = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        boundary_feat = self.boundary(x)
        combined_feat = sobel_feat + boundary_feat
        aligned_feat = self.channel_align(combined_feat)
        return aligned_feat + self.shortcut(x)
