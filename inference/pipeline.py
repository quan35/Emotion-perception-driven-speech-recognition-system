"""
完整推理流水线：ASR (Whisper) + SER (情感识别)。
支持文件路径和 numpy 数组输入。
"""

import os
import sys
import tempfile
import numpy as np
import torch
import whisper
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import (
    load_config, load_audio, pad_or_trim,
    EMOTION_LABELS, ID2LABEL, EMOTION_COLORS, EMOTION_NAMES_ZH,
)
from preprocessing.audio_preprocess import AudioPreprocessor
from preprocessing.feature_extract import FeatureExtractor
from models.emotion_cnn_bilstm import EmotionRecognizer
from models.whisper_emotion import WhisperEmotionHead


class ASREngine:
    """Whisper 语音识别引擎封装。"""

    def __init__(self, model_size="small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载 Whisper {model_size} 模型...")
        self.model = whisper.load_model(model_size, device=self.device)
        self.model_size = model_size

    def transcribe(self, audio_input, language=None):
        """
        语音转文本。
        audio_input: 文件路径 (str) 或 numpy 数组。
        language: 指定语言 (None 为自动检测, "zh" 中文, "en" 英文)。
        返回 dict: {"text", "segments", "language"}
        """
        if isinstance(audio_input, np.ndarray):
            audio_input = audio_input.astype(np.float32)
            if audio_input.max() > 1.0:
                audio_input = audio_input / np.max(np.abs(audio_input))
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                import soundfile as sf
                sf.write(tmp.name, audio_input, 16000)
                result = self.model.transcribe(
                    tmp.name, language=language, word_timestamps=True,
                )
            finally:
                os.unlink(tmp.name)
        else:
            result = self.model.transcribe(
                audio_input, language=language, word_timestamps=True,
            )

        return {
            "text": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
        }

    def get_whisper_model(self):
        return self.model


class EmotionAwareSpeechPipeline:
    """
    情感感知语音识别流水线。
    整合 ASR + SER，一次调用返回文字 + 情感。
    """

    def __init__(self, config=None, emotion_model_path=None,
                 use_shared_model=False, whisper_size=None):
        self.cfg = config or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = AudioPreprocessor(self.cfg)
        self.feature_extractor = FeatureExtractor(self.cfg)

        ws = whisper_size or self.cfg["model"]["whisper_size"]
        self.asr = ASREngine(model_size=ws, device=str(self.device))

        self.use_shared = use_shared_model
        if use_shared_model:
            self._load_shared_model()
        else:
            self._load_emotion_model(emotion_model_path)

    def _load_emotion_model(self, path=None):
        """加载 CNN+BiLSTM+Attention 情感模型。"""
        path = path or self.cfg["paths"]["best_emotion_model"]
        self.emotion_model = EmotionRecognizer(
            num_classes=self.cfg["emotion"]["num_classes"],
            n_mels=self.cfg["audio"]["n_mels"],
            cnn_channels=tuple(self.cfg["model"]["cnn_channels"]),
            lstm_hidden=self.cfg["model"]["lstm_hidden"],
            lstm_layers=self.cfg["model"]["lstm_layers"],
        )
        if os.path.isfile(path):
            state = torch.load(path, map_location=self.device)
            self.emotion_model.load_state_dict(state)
            print(f"已加载情感模型: {path}")
        else:
            print(f"警告: 情感模型文件不存在 ({path})，使用随机权重")
        self.emotion_model.to(self.device)
        self.emotion_model.eval()

    def _load_shared_model(self):
        """加载 Whisper 共享编码器情感模型。"""
        path = self.cfg["paths"]["best_shared_model"]
        self.emotion_model = WhisperEmotionHead(
            self.asr.get_whisper_model(),
            num_classes=self.cfg["emotion"]["num_classes"],
        ).to(self.device)

        if os.path.isfile(path):
            ckpt = torch.load(path, map_location=self.device)
            self.emotion_model.classifier.load_state_dict(ckpt["classifier_state"])
            print(f"已加载共享编码器模型: {path}")
        else:
            print(f"警告: 共享编码器模型不存在 ({path})，使用随机权重")
        self.emotion_model.eval()

    def _get_emotion_from_mel(self, audio):
        """使用 CNN+BiLSTM+Attention 模型进行情感识别。"""
        sr = self.cfg["audio"]["sample_rate"]
        target_len = int(sr * self.cfg["audio"]["max_duration"])
        audio = pad_or_trim(audio, target_len)

        mel = self.feature_extractor.extract_mel(audio)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.emotion_model.predict_proba(mel_tensor)
        return probs.cpu().numpy()[0]

    def _get_emotion_from_whisper(self, audio_path):
        """使用 Whisper 共享编码器模型进行情感识别。"""
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.emotion_model.predict_proba(mel)
        return probs.cpu().numpy()[0]

    def process(self, audio_input, language=None):
        """
        处理一段音频，返回转录文字 + 情感结果。

        audio_input: 文件路径 (str) 或 (numpy_array, sample_rate) 元组。
        返回 dict 包含:
          - text: 转录文本
          - segments: 分段信息
          - language: 检测到的语言
          - emotion: 情感标签 (英文)
          - emotion_zh: 情感标签 (中文)
          - emotion_probs: 各情感的概率分布 dict
          - emotion_color: 情感对应的颜色
          - confidence: 最大概率值
        """
        # 处理 numpy 输入
        if isinstance(audio_input, tuple):
            audio_array, sr = audio_input
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                import soundfile as sf
                if sr != 16000:
                    audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=16000)
                sf.write(tmp.name, audio_array, 16000)
                return self._process_file(tmp.name, language)
            finally:
                os.unlink(tmp.name)
        else:
            return self._process_file(audio_input, language)

    def _process_file(self, audio_path, language=None):
        # ASR
        asr_result = self.asr.transcribe(audio_path, language=language)

        # SER
        if self.use_shared:
            probs = self._get_emotion_from_whisper(audio_path)
        else:
            audio, _ = load_audio(audio_path, sr=self.cfg["audio"]["sample_rate"])
            audio = self.preprocessor.denoise(audio)
            probs = self._get_emotion_from_mel(audio)

        emotion_idx = int(np.argmax(probs))
        emotion_label = EMOTION_LABELS[emotion_idx]

        return {
            "text": asr_result["text"],
            "segments": asr_result["segments"],
            "language": asr_result["language"],
            "emotion": emotion_label,
            "emotion_zh": EMOTION_NAMES_ZH[emotion_label],
            "emotion_probs": {
                label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)
            },
            "emotion_color": EMOTION_COLORS[emotion_label],
            "confidence": float(probs[emotion_idx]),
        }
