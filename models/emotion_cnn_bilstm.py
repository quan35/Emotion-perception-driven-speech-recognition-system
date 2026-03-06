"""
CNN + BiLSTM + Attention 情感识别模型。
输入: Mel 频谱图 (batch, 1, n_mels, time_steps)
输出: 情感分类 logits (batch, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    全局平均池化 -> 两层 FC 学习通道权重 -> Sigmoid 门控重标定。
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

    当 in_ch != out_ch 时，使用 1x1 卷积做通道对齐；
    MaxPool 同时作用于主路径和残差路径，保持空间尺寸一致。
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
    """多头自注意力时间聚合模块。

    将 BiLSTM 输出从 (batch, time, embed_dim) 聚合为 (batch, embed_dim)。
    每个头在独立子空间中计算 scaled dot-product attention 权重，
    拼接后通过线性投影再沿时间维加权求和。
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

    def forward(self, x):
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
        
        weighted_context = torch.bmm(weights, context).squeeze(1) # (B, D)

        return weighted_context                    # (B, D) 沿时间维加权聚合


class EmotionRecognizer(nn.Module):
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

    def predict_proba(self, x):
        """返回 softmax 概率分布。"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
