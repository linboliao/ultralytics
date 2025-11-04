import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskCoefficientTripletLoss(nn.Module):
    """
    专门针对掩码系数的三元组损失
    在掩码系数空间进行对比学习，完全避免像素级计算
    """

    def __init__(self, margin=0.3, prototype_weight=0.5):
        super().__init__()
        self.margin = margin
        self.prototype_weight = prototype_weight

    def forward(self, pred_masks, target_gt_idx, fg_mask):
        """
        在掩码系数空间进行对比学习

        Args:
            pred_masks: 预测的掩码系数 (BS, N_anchors, 32)
            proto: 原型掩码 (BS, 32, H, W)
            target_gt_idx: 实例索引
            fg_mask: 前景掩码
        """
        batch_size = pred_masks.shape[0]
        total_loss = 0.0
        num_triplets = 0

        for i in range(batch_size):
            if not fg_mask[i].any():
                continue

            # 获取当前批次的有效anchor和实例标签
            valid_indices = torch.where(fg_mask[i])[0]
            instances = target_gt_idx[i][valid_indices]
            mask_coefficients = pred_masks[i][valid_indices]  # (num_valid, 32)

            # 特征归一化
            mask_coefficients = F.normalize(mask_coefficients, p=2, dim=1)

            unique_instances = instances.unique()

            for inst_id in unique_instances:
                # 当前实例的anchor
                inst_mask = (instances == inst_id)
                inst_coeffs = mask_coefficients[inst_mask]

                if len(inst_coeffs) < 2:
                    continue

                # 其他实例的anchor作为负样本
                other_inst_mask = (instances != inst_id)
                if not other_inst_mask.any():
                    continue

                other_coeffs = mask_coefficients[other_inst_mask]

                # 为当前实例的每个anchor构建三元组
                for anchor_idx in range(len(inst_coeffs)):
                    anchor = inst_coeffs[anchor_idx]

                    # 正样本：同一实例的其他anchor
                    pos_mask = torch.ones(len(inst_coeffs), dtype=torch.bool)
                    pos_mask[anchor_idx] = False
                    positives = inst_coeffs[pos_mask]

                    # 困难负样本挖掘：选择最相似的负样本
                    similarities = torch.matmul(anchor.unsqueeze(0), other_coeffs.T)
                    hardest_neg_idx = similarities.argmax()
                    negative = other_coeffs[hardest_neg_idx]

                    # 随机选择一个正样本
                    pos_idx = torch.randint(0, len(positives), (1,))
                    positive = positives[pos_idx]

                    # 计算三元组损失
                    pos_distance = F.pairwise_distance(anchor.unsqueeze(0),
                                                       positive.unsqueeze(0), p=2)
                    neg_distance = F.pairwise_distance(anchor.unsqueeze(0),
                                                       negative.unsqueeze(0), p=2)

                    loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
                    total_loss += loss
                    num_triplets += 1

        return total_loss / max(num_triplets, 1) if num_triplets > 0 else torch.tensor(0.0)