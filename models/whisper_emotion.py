"""
Whisper 共享编码器 + 情感分类头。
复用 Whisper Encoder 提取的特征进行情感分类，
满足"共享特征提取层并行处理识别与情感分析任务"的要求。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


WHISPER_DIMS = {
    "tiny": 384,
    "base": 512,
    "small": 768,
    "medium": 1024,
    "large": 1280,
}


class WhisperEmotionHead(nn.Module):
    """
    从 Whisper Encoder 提取特征，经过池化后接分类头。
    Whisper Encoder 参数默认冻结，只训练分类头。
    """

    def __init__(self, whisper_model, num_classes=6, freeze_encoder=True):
        super().__init__()
        self.encoder = whisper_model.encoder
        enc_dim = self.encoder.ln_post.normalized_shape[0]

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, mel):
        """
        mel: Whisper 格式的 Mel 频谱图 (batch, 80, 3000)。
        可使用 whisper.log_mel_spectrogram() 生成。
        """
        with torch.no_grad():
            features = self.encoder(mel)  # (batch, time, enc_dim)

        features = features.permute(0, 2, 1)     # (batch, enc_dim, time)
        pooled = self.pool(features).squeeze(-1)  # (batch, enc_dim)
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, mel):
        logits = self.forward(mel)
        return F.softmax(logits, dim=-1)
