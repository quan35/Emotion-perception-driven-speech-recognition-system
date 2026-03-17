"""
特征提取模块：从预处理后的音频提取 Mel 频谱图和 MFCC 特征，
保存为 .npy 文件供训练使用，同时提供 PyTorch Dataset。
"""

import os
import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import load_config, load_audio, pad_or_trim, LABEL2ID


class FeatureExtractor:
    def __init__(self, config=None):
        self.cfg = config or load_config()
        self.sr = self.cfg["audio"]["sample_rate"]
        self.n_mels = self.cfg["audio"]["n_mels"]
        self.n_mfcc = self.cfg["audio"]["n_mfcc"]
        self.hop_length = self.cfg["audio"]["hop_length"]
        self.n_fft = self.cfg["audio"]["n_fft"]
        self.max_dur = self.cfg["audio"]["max_duration"]

    def extract_mel(self, audio):
        """提取 Mel 频谱图，返回 (n_mels, time_steps) 的 numpy 数组。"""
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def extract_mfcc(self, audio):
        """提取 MFCC 特征，返回 (n_mfcc, time_steps) 的 numpy 数组。"""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr,
            n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length,
        )
        return mfcc

    def extract_from_file(self, audio_path):
        """从音频文件提取 Mel 和 MFCC。"""
        audio, _ = load_audio(audio_path, sr=self.sr)
        target_len = int(self.sr * self.max_dur)
        audio = pad_or_trim(audio, target_len)
        return {
            "mel": self.extract_mel(audio),
            "mfcc": self.extract_mfcc(audio),
        }

    def extract_dataset(self, processed_dir, features_dir):
        """
        批量提取特征。
        processed_dir 结构: {emotion_label}/{file}.wav
        输出到 features_dir: mel/{emotion_label}/{file}.npy, mfcc/{emotion_label}/{file}.npy
        """
        mel_dir = os.path.join(features_dir, "mel")
        mfcc_dir = os.path.join(features_dir, "mfcc")
        count = 0

        for filepath in glob.glob(os.path.join(processed_dir, "**", "*.wav"), recursive=True):
            rel = os.path.relpath(filepath, processed_dir)
            stem = os.path.splitext(rel)[0]

            features = self.extract_from_file(filepath)

            mel_path = os.path.join(mel_dir, stem + ".npy")
            mfcc_path = os.path.join(mfcc_dir, stem + ".npy")
            os.makedirs(os.path.dirname(mel_path), exist_ok=True)
            os.makedirs(os.path.dirname(mfcc_path), exist_ok=True)

            np.save(mel_path, features["mel"])
            np.save(mfcc_path, features["mfcc"])
            count += 1

        print(f"特征提取完成: {count} 个文件")
        return count


class EmotionDataset(Dataset):
    """加载 Mel 频谱图特征的 PyTorch Dataset。"""

    def __init__(self, features_dir, feature_type="mel", transform=None):
        self.transform = transform
        self.samples = []

        feat_dir = os.path.join(features_dir, feature_type)
        if not os.path.isdir(feat_dir):
            raise FileNotFoundError(f"特征目录不存在: {feat_dir}")

        for label_name in sorted(os.listdir(feat_dir)):
            label_dir = os.path.join(feat_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            if label_name not in LABEL2ID:
                continue
            label_id = LABEL2ID[label_name]
            for npy_file in sorted(glob.glob(os.path.join(label_dir, "*.npy"))):
                self.samples.append((npy_file, label_id))

        print(f"EmotionDataset: 加载 {len(self.samples)} 个样本 (feature_type={feature_type})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = np.load(path).astype(np.float32)

        if self.transform:
            feat = self.transform(feat)

        mean = feat.mean()
        std = feat.std()
        if std > 0:
            feat = (feat - mean) / std

        # (1, n_mels, time_steps) -- 单通道
        feat_tensor = torch.from_numpy(feat).unsqueeze(0)
        return feat_tensor, label


class AudioAugmentation:
    """频谱图数据增强: SpecAugment (时间/频率遮蔽) + 高斯噪声注入。"""

    def __init__(self, time_mask_max=20, freq_mask_max=10, noise_std=0.01):
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.noise_std = noise_std

    def __call__(self, spec):
        spec = spec.copy()
        n_freq, n_time = spec.shape

        # 时间遮蔽 (SpecAugment)
        t = np.random.randint(0, self.time_mask_max + 1)
        if t > 0 and n_time > t:
            t0 = np.random.randint(0, n_time - t)
            spec[:, t0:t0 + t] = spec.min()

        # 频率遮蔽
        f = np.random.randint(0, self.freq_mask_max + 1)
        if f > 0 and n_freq > f:
            f0 = np.random.randint(0, n_freq - f)
            spec[f0:f0 + f, :] = spec.min()

        # 高斯噪声注入
        if self.noise_std > 0:
            noise = np.random.randn(*spec.shape).astype(spec.dtype) * self.noise_std
            spec = spec + noise

        return spec


if __name__ == "__main__":
    cfg = load_config()
    extractor = FeatureExtractor(cfg)

    processed = cfg["paths"]["processed_data"]
    features = cfg["paths"]["features"]

    for subset in ("ravdess", "casia", "tess", "esd", "emodb", "iemocap"):
        subset_dir = os.path.join(processed, subset)
        if os.path.isdir(subset_dir):
            extractor.extract_dataset(subset_dir, os.path.join(features, subset))

    print("全部特征提取完成。")
