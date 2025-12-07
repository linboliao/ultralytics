import math
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Softmax
from einops import rearrange, repeat

from ultralytics.nn.modules import Conv, A2C2f, Detect, ChannelAttention
from ultralytics.nn.modules.block import C3k2, Proto

num_patches = 4
num_chunks = num_patches + 1


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


class MSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.img_block = Conv(c1, c2, k, s, p, g, d, act)
        if c1 == 3:
            self.patch_block = Conv(c1, c2, k, s // 2, p, g, d, act)
        else:
            self.patch_block = Conv(c1, c2, k, s, p, g, d, act)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 3:
            img = self.img_block(x)
            ps = patches(x)
        else:
            split = torch.chunk(x, chunks=num_chunks, dim=1)
            img = self.img_block(split[0])
            ps = split[1:]

        parts = []
        for patch in ps:
            parts.append(self.patch_block(patch))

        return torch.cat([img] + parts, 1)


class MSC3k2(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.img_block = C3k2(c1, c2, n, c3k, e, g, shortcut)
        self.patch_block = C3k2(c1, c2, n, c3k, e, g, shortcut)

    def forward(self, x):
        split = torch.chunk(x, chunks=num_chunks, dim=1)
        img = self.img_block(split[0])
        parts = []
        for patch in split[1:]:
            parts.append(self.patch_block(patch))
        return torch.cat([img] + parts, 1)


class MSA2C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1, residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.img_block = A2C2f(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.patch_block = A2C2f(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)

    def forward(self, x):
        split = torch.chunk(x, chunks=num_chunks, dim=1)
        img = self.img_block(split[0])
        parts = []
        for patch in split[1:]:
            parts.append(self.patch_block(patch))
        return torch.cat([img] + parts, 1)


class DynamicSpatialAttention(nn.Module):
    def __init__(self, c1=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c1, kernel_size ** 2, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)
        x_mean = x.mean(dim=1, keepdim=True)
        x_mean = x_mean.view(1, B, H, W)
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)
        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        return x * att


import math


class DifferentiableTopK(nn.Module):
    """
    可微分的TopK操作符，基于归一化重要性分数和软掩码生成
    参考DMS论文中的可微TopK实现[2](@ref)
    """

    def __init__(self, topk_ratio=0.3, temperature=10.0):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.temperature = temperature

    def importance_normalization(self, importance_scores):
        """
        重要性归一化：将重要性分数转换为均匀分布
        """
        B, C, H, W = importance_scores.shape
        N = H * W

        # 展平处理 (B, C, H*W)
        flat_scores = importance_scores.view(B, C, -1)

        # 计算每个元素的排名（可微近似）
        sorted_scores, indices = torch.sort(flat_scores, dim=-1, descending=True)

        # 生成均匀分布的排名分数，并明确指定为float32类型
        ranks = torch.linspace(0, 1, steps=N, device=importance_scores.device, dtype=torch.float32)  # 指定dtype
        ranks = ranks.view(1, 1, N).expand(B, C, -1)

        # 创建目标张量时也使用float32类型
        normalized_importance = torch.zeros_like(flat_scores, dtype=torch.float32)  # 指定dtype

        for b in range(B):
            for c in range(C):
                # 此时indices[b, c]和ranks[0,0]都应该是CPU上的整数和float32，不会冲突
                normalized_importance[b, c, indices[b, c]] = ranks[0, 0, 0]  # 注意：ranks[0,0]是一个标量

        return normalized_importance.view(B, C, H, W)

    def soft_mask_generation(self, importance_scores, normalized_importance):
        """
        软掩码生成：使用sigmoid函数生成连续的软掩码[2](@ref)
        """
        B, C, H, W = importance_scores.shape
        N = H * W
        k = int(self.topk_ratio * N)

        # 计算动态阈值（基于topk比例）
        flat_scores = importance_scores.view(B, C, -1)
        threshold = flat_scores.kthvalue(N - k, dim=-1).values  # 第k大的值
        threshold = threshold.view(B, C, 1, 1)

        # 使用sigmoid生成软掩码[2](@ref)
        soft_mask = torch.sigmoid(self.temperature * (importance_scores - threshold))

        return soft_mask

    def forward(self, importance_scores):
        """
        前向传播：生成可微的TopK软掩码

        Args:
            importance_scores: 重要性分数图 (B, 1, H, W)

        Returns:
            soft_mask: 软掩码 (B, 1, H, W)，值在[0,1]之间
        """
        # 归一化重要性分数
        normalized_importance = self.importance_normalization(importance_scores)

        # 生成软掩码
        soft_mask = self.soft_mask_generation(importance_scores, normalized_importance)

        return soft_mask


class SoftTopKRegionSelection(nn.Module):
    """
    基于Soft TopK的区域选择模块
    替换原有的硬性TopK选择，实现完全可微分[2](@ref)
    """

    def __init__(self, topk_ratio=0.3, temperature=10.0, min_weight=0.1):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.min_weight = min_weight
        self.differentiable_topk = DifferentiableTopK(topk_ratio, temperature)

    def forward(self, local_feat, attention_map, spatial_scale=2.0):
        """
        前向传播：基于Soft TopK的区域选择

        Args:
            local_feat: 局部特征 (B, C, H_local, W_local)
            attention_map: 注意力图 (B, 1, H_global, W_global)
            spatial_scale: 空间尺度缩放因子

        Returns:
            weighted_local_feat: 加权后的局部特征
            selection_mask: 选择掩码（软掩码）
        """
        B, C, H_global, W_global = attention_map.shape
        target_dtype = local_feat.dtype

        # 上采样注意力图到局部特征尺寸
        upsampled_attention = F.interpolate(
            attention_map,
            scale_factor=spatial_scale,
            mode='bilinear',
            align_corners=False
        )

        if self.training:
            # 训练时使用可微分的Soft TopK
            selection_mask = self.differentiable_topk(upsampled_attention)
        else:
            # 推理时可以使用硬TopK以获得更好的效率
            k = int(self.topk_ratio * H_global * W_global)
            flat_attention = upsampled_attention.view(B, -1)
            threshold = flat_attention.kthvalue(flat_attention.size(1) - k).values
            selection_mask = (upsampled_attention >= threshold.view(B, 1, 1, 1)).to(dtype=target_dtype)

        # 应用基础权重确保梯度流动[2](@ref)
        base_weight = torch.tensor(self.min_weight, dtype=target_dtype, device=local_feat.device)
        weighted_local_feat = local_feat * (selection_mask + base_weight)

        return weighted_local_feat, selection_mask


class RegionSelection(nn.Module):
    def __init__(self, topk_ratio=0.3):
        super().__init__()
        self.topk_ratio = topk_ratio

    def forward(self, local_feat, attention_map, spatial_scale=2.0):
        B, C, H_global, W_global = attention_map.shape

        upsampled_attention = F.interpolate(attention_map, scale_factor=spatial_scale, mode='bilinear', align_corners=False)
        target_dtype = local_feat.dtype
        if self.training:
            selection_mask = torch.sigmoid(10.0 * (upsampled_attention - 0.5))
        else:
            k = int(self.topk_ratio * H_global * W_global)
            flat_attention = upsampled_attention.view(B, -1)
            threshold = flat_attention.kthvalue(flat_attention.size(1) - k).values
            selection_mask = (upsampled_attention >= threshold.view(B, 1, 1, 1)).to(dtype=target_dtype)
        base_weight = torch.tensor(0.1, dtype=target_dtype, device=local_feat.device)
        weighted_local_feat = local_feat * (selection_mask + base_weight)

        return weighted_local_feat, selection_mask


class GatedFeatureFusion(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.global_proj = nn.Conv2d(c1, c2, 1)
        self.local_proj = nn.Conv2d(c1, c2, 1)

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, global_feat, weighted_local_feat):
        proj_global = self.global_proj(global_feat)
        proj_local = self.local_proj(weighted_local_feat)

        upsampled_global = F.interpolate(proj_global,
                                         size=weighted_local_feat.shape[2:],
                                         mode='bilinear',
                                         align_corners=False)

        alpha = torch.sigmoid(self.fusion_weight)
        fused_feat = alpha * upsampled_global + (1 - alpha) * proj_local

        return fused_feat


class MSFusionV1(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.downsample = Conv(c1, c1, 3, 2)
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = DynamicSpatialAttention(c1, 3)
        self.region_selector = SoftTopKRegionSelection(topk_ratio=0.3)
        self.feature_fusion = GatedFeatureFusion(c1, c1)
        self.grid_size = int(math.sqrt(num_patches))

    def forward(self, x):
        split = torch.chunk(x, chunks=num_chunks, dim=1)
        global_feat = split[0]
        local_feat = self.reconstruct(split[1:])

        attention_map = self.spatial_attention(global_feat)
        # weighted_local_feat, selection_mask = self.region_selector(local_feat, attention_map)

        fused_feat = self.feature_fusion(attention_map, local_feat)
        return self.downsample(fused_feat)

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
