"""
CNN + BiLSTM + Attention 情感识别模型。
输入: Mel 频谱图 (batch, 1, n_mels, time_steps)
输出: 情感分类 logits (batch, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionRecognizer(nn.Module):
    def __init__(self, num_classes=6, n_mels=128,
                 cnn_channels=(32, 64, 128),
                 lstm_hidden=64, lstm_layers=2, lstm_dropout=0.3,
                 attn_dim=64, cls_hidden=64, cls_dropout=0.5):
        super().__init__()

        # ---- CNN 特征提取 ----
        layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # 经过3次 MaxPool2d(2)，频率维度 n_mels -> n_mels // 8
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
        bilstm_out_dim = lstm_hidden * 2  # 双向

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
        # CNN: (batch, C, H', T')
        out = self.cnn(x)
        batch, channels, freq, time = out.shape

        # 重排为 (batch, time, channels * freq) 送入 LSTM
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch, time, channels * freq)

        # BiLSTM: (batch, time, lstm_hidden * 2)
        out, _ = self.bilstm(out)

        # Attention: 计算权重并加权求和
        attn_weights = self.attention_fc(out)           # (batch, time, 1)
        attn_weights = F.softmax(attn_weights, dim=1)   # 沿时间维归一化
        context = torch.sum(out * attn_weights, dim=1)  # (batch, lstm_hidden * 2)

        # 分类
        logits = self.classifier(context)
        return logits

    def predict_proba(self, x):
        """返回 softmax 概率分布。"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
