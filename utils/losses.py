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
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets = targets.long()

        if self.label_smoothing > 0:
            num_classes = int(logits.size(-1))
            smooth = float(self.label_smoothing) / float(num_classes)
            target_dist = torch.full_like(log_probs, smooth)
            target_dist.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - float(self.label_smoothing) + smooth,
            )
            ce = -(target_dist * log_probs).sum(dim=-1)
        else:
            ce = F.nll_loss(log_probs, targets, reduction='none')

        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_(1e-8, 1.0)
        loss = ((1.0 - p_t) ** self.gamma) * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss
            return loss.sum() / alpha_t.sum().clamp_min(1e-6)

        return loss.mean()
