"""
自定义损失函数：Focal Loss。
用于类别不平衡场景下的多分类任务。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017).

    通过 (1 - p_t)^gamma 调制因子降低易分样本的损失贡献，
    使模型更关注难分样本，缓解类别不平衡问题。

    参数:
        gamma: 聚焦参数，越大对易分样本惩罚越强。默认 2.0。
        alpha: 类别权重张量 (num_classes,)，为 None 时各类等权。
        label_smoothing: 标签平滑系数，与 CrossEntropy 的 label_smoothing 一致。
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha, reduction='none',
            label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean()
