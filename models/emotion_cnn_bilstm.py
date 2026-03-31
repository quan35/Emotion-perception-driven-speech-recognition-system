"""
早期探索模型：CNN + BiLSTM + Attention 情感识别模型
=================================================

项目早期的可解释 SER 探索路线，用于验证数据处理链路、训练评估协议与主线模型的整体增益来源。

设计思路：
    语音情感信息同时蕴含在 **局部声学特征**（音高突变、能量爆发）和 **全局时序模式**（语速变化、语调走向）中，因此采用三阶段级联架构：

    Mel 频谱图 → 残差SE-CNN(局部特征) → BiLSTM(时序建模) → 多头注意力(关键帧聚合) → 分类头 → 情感类别

输入: Mel 频谱图 (batch, 1, n_mels=128, time_steps)
      Mel 刻度模拟人耳对频率的非线性感知，低频分辨率高、高频分辨率低，
      与人类对语音情感的感知特性一致。
输出: 情感分类 logits (batch, num_classes=6)

总参数量: ~1.4M（全部参与训练，从零训练范式）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    通过全局平均池化 -> 两层 FC 学习通道权重 -> Sigmoid 门控重标定。
    动态学习每个特征通道的重要性，放大关键特征（如共振峰、音高相关特征），
    抑制无关特征。
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class _ResidualCNNBlock(nn.Module):
    """带残差连接 + SE 通道注意力的 CNN 块。

    残差连接：允许梯度通过快捷通道直接回传，缓解深度网络中的梯度消失问题。
    SE 注意力：全局平均池化 + 两层 FC，动态学习通道重要性权重。
    Spatial Dropout：在特征图层面随机丢弃整个通道，强迫网络学习更多样化的特征。

    当 in_ch != out_ch 时，使用 1x1 卷积做通道对齐；
    MaxPool 同时作用于主路径和残差路径，保持空间尺寸一致。

    3 层级联后的维度变化：
        输入 (B, 1, 128, T)
          → 第1层 → (B, 32, 64, T/2)
          → 第2层 → (B, 64, 32, T/4)
          → 第3层 → (B, 128, 16, T/8)
          → 展平   → (B, T/8, 2048)
    """

    def __init__(self, in_ch, out_ch, dropout=0.1, se_reduction=4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.se = _SEBlock(out_ch, reduction=se_reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        out = self.se(out)
        out = out + residual
        out = self.pool(out)
        out = self.dropout(out)
        return out


class _MultiHeadAttention(nn.Module):
    """多头自注意力时序聚合模块。

    将 BiLSTM 输出从 (batch, time, embed_dim) 聚合为 (batch, embed_dim)。

    工作流程：
      1. 自注意力变换：通过 Q/K/V 矩阵计算缩放点积注意力，
         生成上下文感知的序列表示
      2. 时间维加权聚合：从注意力分数中派生时序重要性权重，
         对上下文序列加权求和，得到 (B, embed_dim) 的句级表示

    多头机制（默认 4 头，每头 32 维）让模型在独立子空间中学习不同的
    关注模式（如音高变化、能量爆发等），避免单头注意力的表达瓶颈。
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        """
        x: (batch, time, embed_dim)
        返回: (batch, embed_dim) — 聚合后的上下文向量
        """
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.W_q(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        k = self.W_k(x).view(B, T, H, d).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)              # (B, H, T, d)
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        context = self.out_proj(context)              # (B, T, D)
        
        # 加权求和，而不是简单平均
        weights = attn.sum(dim=1).mean(dim=1) # (B, T)
        weights = F.softmax(weights, dim=1).unsqueeze(1) # (B, 1, T)
        
        weighted_context = torch.bmm(weights, context).squeeze(1)

        if return_attn:
            return weighted_context, attn, weights.squeeze(1)
        return weighted_context


class EmotionRecognizer(nn.Module):
    """早期探索模型：CNN + BiLSTM + Attention。

    项目早期用于系统可行性验证与对照分析的三阶段级联架构：
      阶段 1 - 残差 SE-CNN：从 Mel 频谱图中提取局部声学模式
      阶段 2 - BiLSTM：沿时间轴双向建模上下文依赖关系
      阶段 3 - 多头注意力：自适应聚焦并加权聚合情感最强烈的时序特征

    默认维度流：
      输入 Mel:       (B, 1, 128, T)
      CNN 输出展平:   (B, T', 2048)     # T' = T/8
      BiLSTM 输出:    (B, T', 128)      # 2 × lstm_hidden
      注意力聚合:     (B, 128)
      分类器:         128 → 64 → 6

    训练策略：
      - 高斯噪声注入：训练时在输入频谱图上叠加微小噪声（std=0.01），提升鲁棒性
      - 权重初始化：Conv 用 Kaiming、LSTM 用 Orthogonal、Linear 用 Xavier
      - LayerNorm：在 BiLSTM 和 Attention 之间稳定输入分布
    """
    def __init__(self, num_classes=6, n_mels=128,
                 cnn_channels=(32, 64, 128),
                 cnn_dropout=0.1, se_reduction=4,
                 noise_std=0.01,
                 lstm_hidden=64, lstm_layers=2, lstm_dropout=0.3,
                 num_heads=4, attn_dropout=0.1,
                 attn_dim=64, cls_hidden=64, cls_dropout=0.5):
        super().__init__()

        self.noise_std = noise_std

        # ---- 残差 CNN + SE 特征提取 ----
        self.cnn_blocks = nn.ModuleList()
        in_ch = 1
        for out_ch in cnn_channels:
            self.cnn_blocks.append(
                _ResidualCNNBlock(in_ch, out_ch,
                                  dropout=cnn_dropout, se_reduction=se_reduction)
            )
            in_ch = out_ch

        cnn_freq_out = n_mels // (2 ** len(cnn_channels))
        cnn_out_dim = cnn_channels[-1] * cnn_freq_out

        # ---- BiLSTM 时序建模 ----
        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )
        bilstm_out_dim = lstm_hidden * 2

        # ---- LayerNorm: 稳定 Attention 输入 ----
        self.ln = nn.LayerNorm(bilstm_out_dim)

        # ---- 多头注意力聚合 ----
        self.attention = _MultiHeadAttention(
            embed_dim=bilstm_out_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
        )

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """显式权重初始化: Kaiming (Conv2d)、Xavier (Linear)、正交 (LSTM)。"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)

    def forward(self, x):
        """
        x: (batch, 1, n_mels, time_steps)
        """
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        out = x
        for block in self.cnn_blocks:
            out = block(out)
        batch, channels, freq, time = out.shape

        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch, time, channels * freq)

        out, _ = self.bilstm(out)

        out = self.ln(out)

        context = self.attention(out)  # (batch, bilstm_out_dim)

        logits = self.classifier(context)
        return logits

    def forward_with_intermediates(self, x, return_attn=True):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        out = x
        for block in self.cnn_blocks:
            out = block(out)
        cnn_out = out
        batch, channels, freq, time = out.shape

        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch, time, channels * freq)

        lstm_out, _ = self.bilstm(out)
        ln_out = self.ln(lstm_out)

        if return_attn:
            context, attn, time_weights = self.attention(ln_out, return_attn=True)
        else:
            context = self.attention(ln_out)
            attn = None
            time_weights = None

        logits = self.classifier(context)
        return {
            "cnn_out": cnn_out,
            "lstm_out": lstm_out,
            "ln_out": ln_out,
            "context": context,
            "attn": attn,
            "time_weights": time_weights,
            "logits": logits,
        }

    def predict_proba(self, x):
        """返回 softmax 概率分布。"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
