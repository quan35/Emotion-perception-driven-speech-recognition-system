import os
import yaml
import numpy as np
import librosa
import soundfile as sf


def load_config(config_path="configs/config.yaml"):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, config_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_audio(path, sr=16000):
    audio, orig_sr = librosa.load(path, sr=sr, mono=True)
    return audio, sr


def save_audio(audio, path, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr)


def get_duration(audio, sr=16000):
    return len(audio) / sr


def pad_or_trim(audio, target_length):
    """Pad with zeros or trim audio to exact target_length samples."""
    if len(audio) >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - len(audio)), mode="constant")


EMOTION_LABELS = ["happy", "angry", "sad", "neutral", "fear", "surprise"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

EMOTION_COLORS = {
    "happy": "#FF8C00",
    "angry": "#DC143C",
    "sad": "#4169E1",
    "neutral": "#808080",
    "fear": "#8B008B",
    "surprise": "#228B22",
}

EMOTION_NAMES_ZH = {
    "happy": "高兴",
    "angry": "愤怒",
    "sad": "悲伤",
    "neutral": "中性",
    "fear": "恐惧",
    "surprise": "惊讶",
}
