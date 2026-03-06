"""
CNN + BiLSTM + Attention 情感识别模型。
输入: Mel 频谱图 (batch, 1, n_mels, time_steps)
输出: 情感分类 logits (batch, num_classes)

优化内容:
  - CNN 残差连接 + Spatial Dropout (Dropout2d)
  - 训练时高斯噪声注入 (仅 training 模式生效)
  - BiLSTM 后 LayerNorm (稳定 Attention 输入)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualCNNBlock(nn.Module):
    """带残差连接的 CNN 块。

    当 in_ch != out_ch 时，使用 1x1 卷积做通道对齐；
    MaxPool 同时作用于主路径和残差路径，保持空间尺寸一致。
    """

    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
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
        out = out + residual
        out = self.pool(out)
        out = self.dropout(out)
        return out


class EmotionRecognizer(nn.Module):
    def __init__(self, num_classes=6, n_mels=128,
                 cnn_channels=(32, 64, 128),
                 cnn_dropout=0.1,
                 noise_std=0.01,
                 lstm_hidden=64, lstm_layers=2, lstm_dropout=0.3,
                 attn_dim=64, cls_hidden=64, cls_dropout=0.5):
        super().__init__()

        self.noise_std = noise_std

        # ---- 残差 CNN 特征提取 ----
        self.cnn_blocks = nn.ModuleList()
        in_ch = 1
        for out_ch in cnn_channels:
            self.cnn_blocks.append(_ResidualCNNBlock(in_ch, out_ch, dropout=cnn_dropout))
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

        # ---- Attention 加权聚合 ----
        self.attention_fc = nn.Sequential(
            nn.Linear(bilstm_out_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, 1, n_mels, time_steps)
        """
        # 训练时高斯噪声注入
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # 残差 CNN
        out = x
        for block in self.cnn_blocks:
            out = block(out)
        batch, channels, freq, time = out.shape

        # 重排为 (batch, time, channels * freq) 送入 LSTM
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch, time, channels * freq)

        # BiLSTM
        out, _ = self.bilstm(out)

        # LayerNorm
        out = self.ln(out)

        # Attention: 计算权重并加权求和
        attn_weights = self.attention_fc(out)           # (batch, time, 1)
        attn_weights = F.softmax(attn_weights, dim=1)   # 沿时间维归一化
        context = torch.sum(out * attn_weights, dim=1)  # (batch, bilstm_out_dim)

        # 分类
        logits = self.classifier(context)
        return logits

    def predict_proba(self, x):
        """返回 softmax 概率分布。"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
