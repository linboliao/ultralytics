import warnings

warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Softmax
from einops import rearrange

from ultralytics.nn.modules import Conv, A2C2f, Detect
from ultralytics.nn.modules.block import C3k2, Proto

import math
import torch

num_patches = 4


def patches(x):
    b, c, h, w = x.shape
    assert c == 3, "输入张量应有3个通道"

    grid_size = int(math.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, f"num_patches应为平方数，当前为{num_patches}"
    assert h % grid_size == 0, f"高度{h}不能被{grid_size}整除"
    assert w % grid_size == 0, f"宽度{w}不能被{grid_size}整除"

    h_patch = h // grid_size
    w_patch = w // grid_size

    patch_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = x[:, :, i * h_patch: (i + 1) * h_patch, j * w_patch: (j + 1) * w_patch]
            patch_list.append(patch)

    return patch_list


class MultiScaleConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.patch_block = Conv(c1, c2, k, s, p, g, d, act)

    def forward(self, x):
        parts = []
        for patch in patches(x):
            parts.append(self.patch_block(patch))

        return torch.cat(parts, 1)


class MSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.patch_block = Conv(c1, c2, k, s, p, g, d, act)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 3:
            split = patches(x)
        else:
            split = torch.chunk(x, chunks=num_patches, dim=1)

        parts = []
        for patch in split:
            parts.append(self.patch_block(patch))

        return torch.cat(parts, 1)


class MSC3k2(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.patch_block = C3k2(c1, c2, n, c3k, e, g, shortcut)

    def forward(self, x):
        split = torch.chunk(x, chunks=num_patches, dim=1)
        parts = []
        for patch in split:
            parts.append(self.patch_block(patch))
        return torch.cat(parts, 1)


class MSA2C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1, residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.patch_block = A2C2f(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)

    def forward(self, x):
        split = torch.chunk(x, chunks=num_patches, dim=1)
        parts = []
        for patch in split:
            parts.append(self.patch_block(patch))
        return torch.cat(parts, 1)


class MSFusionV1(nn.Module):
    def __init__(self, c1, fusion_method='attention'):
        super().__init__()
        self.num_patches = num_patches
        self.grid_size = int(math.sqrt(num_patches))
        self.downsample = Conv(c1, c1, 3, 2)
        self.fusion_method = fusion_method

        if fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Conv2d(c1, c1 // 4, 1),
                nn.ReLU(),
                nn.Conv2d(c1 // 4, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # 先使用计算patch内空间注意力，再合并patches，最后对patch下采样
        split = torch.chunk(x, chunks=self.num_patches, dim=1)

        if self.fusion_method == 'attention':
            weighted_patches = []
            for patch in split:
                attention_weights = self.attention(patch)
                weighted_patches.append(patch * attention_weights)
            split = weighted_patches
        join = self.reconstruct(split)

        return self.downsample(join)

    def reconstruct(self, patches):
        patch_matrix = []
        patch_idx = 0

        for i in range(self.grid_size):
            row_patches = []
            for j in range(self.grid_size):
                if patch_idx < len(patches):
                    row_patches.append(patches[patch_idx])
                    patch_idx += 1
            if row_patches:
                row_reconstructed = torch.cat(row_patches, dim=3)
                patch_matrix.append(row_reconstructed)

        if patch_matrix:
            return torch.cat(patch_matrix, dim=2)
        return patches[0] if patches else None


def boundary_extractor(c1, dilation):
    return nn.Sequential(
        nn.Conv2d(c1, c1 // 4, 3, padding=dilation, dilation=dilation, groups=c1 // 4),
        nn.BatchNorm2d(c1 // 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(c1 // 4, 1, 1)
    )


class BoundaryAttention(nn.Module):
    def __init__(self, c1, scales=None):
        super().__init__()

        if scales is None:
            scales = [1, 2, 3]

        sobel_kernel = torch.tensor([
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
        ], dtype=torch.float32)

        self.sobel_conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        self.sobel_conv.weight = nn.Parameter(sobel_kernel, requires_grad=False)

        self.boundary_extractors = nn.ModuleList([
            boundary_extractor(c1, dilation) for dilation in scales
        ])
        self.fusion = nn.Conv2d(len(scales), 1, 1)

        self.channel_align = nn.Conv2d(1, c1, 1)
        self.shortcut = nn.Identity()

    def forward(self, x):
        gray_x = x.mean(dim=1, keepdim=True)
        grad_xy = self.sobel_conv(gray_x)
        sobel_feat = torch.norm(grad_xy, dim=1, keepdim=True)

        boundary_maps = [extractor(x) for extractor in self.boundary_extractors]
        boundary_feat = self.fusion(torch.cat(boundary_maps, dim=1))

        combined = sobel_feat + boundary_feat
        aligned = self.channel_align(combined)

        return aligned + self.shortcut(x)


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


class CustomSegment(Detect):
    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, ch: tuple = ()):
        super().__init__(nc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(BoundaryAttention(x), Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
