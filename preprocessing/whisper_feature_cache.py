import os
import glob
import json
from typing import Iterable, Tuple, List, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import whisper
import soundfile as sf

from utils.audio_utils import pad_or_trim
from utils.data_policy import build_data_policy_audit, filter_supported_samples


CACHE_FORMAT_VERSION = 2
FEATURE_TYPE_POOLED = "pooled"
FEATURE_TYPE_SEQUENCE = "sequence"
FEATURE_TYPE_AUDIO = "audio"

TRAINING_MODE_LIVE_ENCODER = "live_encoder"
TRAINING_MODE_CACHED_SEQUENCE = "cached_sequence"
TRAINING_MODE_CACHED_POOLED = "cached_pooled"

WHISPER_SAMPLE_RATE = 16000
WHISPER_HOP_LENGTH = 160
WHISPER_MAX_SAMPLES = 30 * WHISPER_SAMPLE_RATE
WHISPER_MEL_FRAMES = 3000
WHISPER_ENCODED_FRAMES = 1500


def _default_cache_dir(project_root: str) -> str:
    return os.path.join(project_root, "data", "features_shared")


def _resolve_np_dtype(feature_dtype: str):
    return np.float16 if str(feature_dtype).lower() == "float16" else np.float32


def _cache_paths(out_dir: str, whisper_size: str, feature_type: str) -> Dict[str, str]:
    prefix = os.path.join(out_dir, f"whisper_{whisper_size}_{feature_type}")
    paths = {
        "features": f"{prefix}_features.npy",
        "labels": f"{prefix}_labels.npy",
        "meta": f"{prefix}_meta.json",
    }
    if feature_type == FEATURE_TYPE_SEQUENCE:
        paths["lengths"] = f"{prefix}_lengths.npy"
    return paths


def _read_cache_meta(meta_path: str) -> Dict[str, object]:
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_cache_meta(meta_path: str, meta: Dict[str, object]) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _infer_encoder_shape(encoder) -> Tuple[int, int]:
    enc_dim = int(encoder.ln_post.normalized_shape[0])
    positional_embedding = getattr(encoder, "positional_embedding", None)
    if positional_embedding is not None:
        seq_len = int(positional_embedding.shape[0])
    else:
        seq_len = WHISPER_ENCODED_FRAMES
    return seq_len, enc_dim


def _encoded_length_from_num_samples(num_samples: int) -> int:
    bounded_samples = max(1, min(int(num_samples), WHISPER_MAX_SAMPLES))
    mel_frames = min(WHISPER_MEL_FRAMES, int(np.ceil(bounded_samples / WHISPER_HOP_LENGTH)))
    return min(WHISPER_ENCODED_FRAMES, int(np.ceil(mel_frames / 2.0)))


def _target_encoded_length_from_max_duration(max_duration_seconds: Optional[float]) -> int:
    if max_duration_seconds is None:
        return WHISPER_ENCODED_FRAMES

    try:
        duration_seconds = float(max_duration_seconds)
    except (TypeError, ValueError):
        return WHISPER_ENCODED_FRAMES

    if duration_seconds <= 0:
        return WHISPER_ENCODED_FRAMES

    target_num_samples = int(round(duration_seconds * WHISPER_SAMPLE_RATE))
    return max(1, min(WHISPER_ENCODED_FRAMES, _encoded_length_from_num_samples(target_num_samples)))


def _validate_existing_cache(meta: Dict[str, object], feature_type: str, whisper_size: str) -> None:
    existing_type = meta.get("feature_type")
    if existing_type is not None and existing_type != feature_type:
        raise ValueError(
            f"缓存类型不匹配: 期望 {feature_type}，实际 {existing_type}。请设置 overwrite=True 重新生成缓存。"
        )
    existing_size = meta.get("whisper_size")
    if existing_size is not None and existing_size != whisper_size:
        raise ValueError(
            f"Whisper 尺寸不匹配: 期望 {whisper_size}，实际 {existing_size}。请设置 overwrite=True 重新生成缓存。"
        )


def _upgrade_pooled_cache_meta(
    meta_path: str,
    features_path: str,
    labels_path: str,
    whisper_size: str,
    feature_dtype: str,
) -> Dict[str, object]:
    meta = _read_cache_meta(meta_path)
    _validate_existing_cache(meta, FEATURE_TYPE_POOLED, whisper_size)

    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"features/labels size mismatch: {features.shape[0]} vs {labels.shape[0]}")
    if features.ndim != 2:
        raise ValueError(f"pooled feature cache 期望二维数组，实际 shape={features.shape}")

    upgraded = dict(meta)
    upgraded.update({
        "format_version": int(upgraded.get("format_version", CACHE_FORMAT_VERSION)),
        "feature_type": FEATURE_TYPE_POOLED,
        "whisper_size": whisper_size,
        "num_samples": int(features.shape[0]),
        "enc_dim": int(features.shape[1]),
        "feature_dtype": str(upgraded.get("feature_dtype", feature_dtype)),
    })
    _write_cache_meta(meta_path, upgraded)
    return upgraded


def build_sample_list(processed_dir: str, label2id: Dict[str, int], subsets: Iterable[str]) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []

    for subset in subsets:
        subset_dir = os.path.join(processed_dir, subset)
        if not os.path.isdir(subset_dir):
            continue

        for label_name in sorted(os.listdir(subset_dir)):
            label_dir = os.path.join(subset_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            if label_name not in label2id:
                continue

            label_id = int(label2id[label_name])
            for wav_path in sorted(glob.glob(os.path.join(label_dir, "*.wav"))):
                samples.append((wav_path, label_id))

    return samples


class WhisperPooledFeatureDataset(Dataset):
    def __init__(self, features_path: str, labels_path: str, feature_dtype=np.float16):
        if not os.path.isfile(features_path):
            raise FileNotFoundError(features_path)
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(labels_path)

        self._features_path = features_path
        self._labels_path = labels_path
        self._features = np.load(features_path, mmap_mode="r")
        self._labels = np.load(labels_path, mmap_mode="r")

        if self._features.shape[0] != self._labels.shape[0]:
            raise ValueError(
                f"features/labels size mismatch: {self._features.shape[0]} vs {self._labels.shape[0]}"
            )

        if self._features.dtype != feature_dtype:
            self._feature_dtype = self._features.dtype
        else:
            self._feature_dtype = feature_dtype

    def __len__(self) -> int:
        return int(self._features.shape[0])

    def __getitem__(self, idx: int):
        feat = np.array(self._features[idx], copy=True)
        feat_t = torch.from_numpy(feat).float()
        label_t = torch.tensor(int(self._labels[idx]), dtype=torch.long)
        return feat_t, label_t


class WhisperSequenceFeatureDataset(Dataset):
    def __init__(self, features_path: str, labels_path: str, lengths_path: str, feature_dtype=np.float16):
        if not os.path.isfile(features_path):
            raise FileNotFoundError(features_path)
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(labels_path)
        if not os.path.isfile(lengths_path):
            raise FileNotFoundError(lengths_path)

        self._features = np.load(features_path, mmap_mode="r")
        self._labels = np.load(labels_path, mmap_mode="r")
        self._lengths = np.load(lengths_path, mmap_mode="r")

        if self._features.shape[0] != self._labels.shape[0] or self._features.shape[0] != self._lengths.shape[0]:
            raise ValueError(
                "sequence cache size mismatch: "
                f"features={self._features.shape[0]}, labels={self._labels.shape[0]}, lengths={self._lengths.shape[0]}"
            )
        if self._features.ndim != 3:
            raise ValueError(f"sequence feature cache 期望三维数组，实际 shape={self._features.shape}")

        self._feature_dtype = self._features.dtype if self._features.dtype != feature_dtype else feature_dtype
        self._seq_len = int(self._features.shape[1])

    def __len__(self) -> int:
        return int(self._features.shape[0])

    def __getitem__(self, idx: int):
        feat = np.array(self._features[idx], copy=True)
        feat_t = torch.from_numpy(feat).float()
        label_t = torch.tensor(int(self._labels[idx]), dtype=torch.long)

        valid_len = int(self._lengths[idx])
        valid_len = max(1, min(valid_len, self._seq_len))
        attention_mask = torch.arange(self._seq_len, dtype=torch.long) < valid_len
        return feat_t, label_t, attention_mask


class _AudioPathDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        wav_path, label = self.samples[idx]
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr), int(label)


def collate_whisper_audio_batch(batch):
    audios, srs, labels = zip(*batch)
    return list(audios), list(srs), torch.tensor(labels, dtype=torch.long)


def _build_whisper_dataloader(
    samples: List[Tuple[str, int]],
    batch_size: int,
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    pin_memory = device.type == "cuda"
    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
        "collate_fn": collate_whisper_audio_batch,
        "persistent_workers": bool(num_workers) if int(num_workers) > 0 else False,
    }
    if int(num_workers) > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(_AudioPathDataset(samples), **dl_kwargs)


def _build_mel_batch(audios: List[np.ndarray], srs: List[int], device: torch.device) -> Tuple[torch.Tensor, List[int]]:
    amp_enabled = device.type == "cuda"
    mels = []
    encoded_lengths = []

    for audio, sr in zip(audios, srs):
        if sr != WHISPER_SAMPLE_RATE:
            raise ValueError(f"Unexpected sample_rate={sr}; expected {WHISPER_SAMPLE_RATE}")
        encoded_lengths.append(_encoded_length_from_num_samples(len(audio)))

        audio_t = torch.from_numpy(audio)
        if amp_enabled:
            audio_t = audio_t.to(device)
        audio_t = whisper.pad_or_trim(audio_t)
        mel = whisper.log_mel_spectrogram(audio_t)
        mels.append(mel)

    mel_batch = torch.stack(mels, dim=0)
    if mel_batch.device != device:
        mel_batch = mel_batch.to(device)
    return mel_batch, encoded_lengths


def prepare_audio_for_whisper_ser(
    audio: np.ndarray,
    sr: int,
    preprocessor=None,
    apply_denoise: bool = False,
) -> np.ndarray:
    """将原始音频整理成与共享模型训练尽量一致的输入。

    该函数主要服务于推理和 UI-style 评估：
      1. 可选降噪
      2. 静音切除
      3. 峰值归一化
      4. 按训练配置裁剪/补零到固定时长

    训练侧 `live_encoder` 已经直接读取 `data/processed` 中的预处理结果，
    因此不会重复经过上述步骤；但训练与推理会共用后续的 mel + mask 构造逻辑。
    """
    if sr != WHISPER_SAMPLE_RATE:
        raise ValueError(f"Unexpected sample_rate={sr}; expected {WHISPER_SAMPLE_RATE}")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if preprocessor is None:
        return audio

    working_audio = audio
    fallback_audio = audio

    if apply_denoise:
        working_audio = preprocessor.denoise(working_audio)
        fallback_audio = working_audio

    trimmed_audio = preprocessor.remove_silence(working_audio)
    min_samples = int(preprocessor.sr * preprocessor.min_dur)
    if len(trimmed_audio) >= max(1, min_samples):
        working_audio = trimmed_audio
    else:
        working_audio = fallback_audio

    working_audio = preprocessor.normalize(working_audio)
    working_audio = pad_or_trim(working_audio, preprocessor.target_samples)
    return working_audio.astype(np.float32, copy=False)


def build_attention_mask_from_encoded_lengths(
    encoded_lengths: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    encoded_lengths = [max(1, min(int(length), WHISPER_ENCODED_FRAMES)) for length in encoded_lengths]
    return torch.arange(WHISPER_ENCODED_FRAMES, device=device).unsqueeze(0) < torch.tensor(
        encoded_lengths, device=device
    ).unsqueeze(1)


def build_whisper_mel_batch(
    audios: List[np.ndarray],
    srs: List[int],
    device: torch.device,
    encoded_lengths: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mel_batch, inferred_lengths = _build_mel_batch(audios, srs, device)
    if encoded_lengths is None:
        effective_lengths = inferred_lengths
    else:
        if len(encoded_lengths) != len(inferred_lengths):
            raise ValueError(
                "encoded_lengths 数量与 batch 大小不一致: "
                f"{len(encoded_lengths)} != {len(inferred_lengths)}"
            )
        effective_lengths = [int(length) for length in encoded_lengths]

    attention_mask = build_attention_mask_from_encoded_lengths(effective_lengths, device)
    return mel_batch, attention_mask


def build_whisper_ser_batch_from_raw_audio(
    audios: List[np.ndarray],
    srs: List[int],
    device: torch.device,
    preprocessor,
    apply_denoise: bool = False,
    apply_denoise_flags: Optional[Sequence[bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(audios) != len(srs):
        raise ValueError(f"audios/srs batch 大小不一致: {len(audios)} != {len(srs)}")

    if apply_denoise_flags is None:
        denoise_flags = [bool(apply_denoise)] * len(audios)
    else:
        if len(apply_denoise_flags) != len(audios):
            raise ValueError(
                "apply_denoise_flags 数量与 batch 大小不一致: "
                f"{len(apply_denoise_flags)} != {len(audios)}"
            )
        denoise_flags = [bool(flag) for flag in apply_denoise_flags]

    prepared_audios = [
        prepare_audio_for_whisper_ser(
            audio,
            sr,
            preprocessor=preprocessor,
            apply_denoise=denoise_flags[idx],
        )
        for idx, (audio, sr) in enumerate(zip(audios, srs))
    ]
    prepared_srs = [WHISPER_SAMPLE_RATE] * len(prepared_audios)
    return build_whisper_mel_batch(prepared_audios, prepared_srs, device)


def build_whisper_ser_input_from_raw_audio(
    audio: np.ndarray,
    sr: int,
    device: torch.device,
    preprocessor,
    apply_denoise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return build_whisper_ser_batch_from_raw_audio(
        audios=[audio],
        srs=[sr],
        device=device,
        preprocessor=preprocessor,
        apply_denoise=apply_denoise,
    )


def extract_whisper_pooled_features(
    samples: List[Tuple[str, int]],
    whisper_size: str,
    device: torch.device,
    out_dir: str,
    batch_size: int = 32,
    feature_dtype: str = "float16",
    overwrite: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> Tuple[str, str, str]:
    """Extract pooled Whisper encoder features and cache to disk."""

    os.makedirs(out_dir, exist_ok=True)
    paths = _cache_paths(out_dir, whisper_size, FEATURE_TYPE_POOLED)

    if not overwrite and os.path.isfile(paths["features"]) and os.path.isfile(paths["labels"]) and os.path.isfile(paths["meta"]):
        _upgrade_pooled_cache_meta(
            meta_path=paths["meta"],
            features_path=paths["features"],
            labels_path=paths["labels"],
            whisper_size=whisper_size,
            feature_dtype=feature_dtype,
        )
        return paths["features"], paths["labels"], paths["meta"]

    whisper_model = whisper.load_model(whisper_size, device=device)
    encoder = whisper_model.encoder
    encoder.eval()

    seq_len, enc_dim = _infer_encoder_shape(encoder)
    n = len(samples)
    np_dtype = _resolve_np_dtype(feature_dtype)

    features_mm = np.lib.format.open_memmap(paths["features"], mode="w+", dtype=np_dtype, shape=(n, enc_dim))
    labels_mm = np.lib.format.open_memmap(paths["labels"], mode="w+", dtype=np.int64, shape=(n,))
    pool = nn.AdaptiveAvgPool1d(1)
    loader = _build_whisper_dataloader(samples, batch_size, device, num_workers, prefetch_factor)

    with torch.no_grad():
        amp_enabled = device.type == "cuda"
        idx = 0
        for audios, srs, labels in loader:
            mel_batch, _ = _build_mel_batch(audios, srs, device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                feats = encoder(mel_batch)
            feats = pool(feats.permute(0, 2, 1)).squeeze(-1)
            feats = feats.detach().cpu().numpy().astype(np_dtype, copy=False)

            bsz = int(labels.shape[0])
            features_mm[idx : idx + bsz] = feats
            labels_mm[idx : idx + bsz] = labels.numpy().astype(np.int64, copy=False)

            idx += bsz
            if idx % 200 == 0 or idx == n:
                print(f"  已提取 {idx}/{n}")

    del features_mm
    del labels_mm

    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "feature_type": FEATURE_TYPE_POOLED,
        "whisper_size": whisper_size,
        "num_samples": n,
        "seq_len": seq_len,
        "enc_dim": enc_dim,
        "feature_dtype": feature_dtype,
    }
    _write_cache_meta(paths["meta"], meta)

    del encoder
    del whisper_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return paths["features"], paths["labels"], paths["meta"]


def extract_whisper_sequence_features(
    samples: List[Tuple[str, int]],
    whisper_size: str,
    device: torch.device,
    out_dir: str,
    batch_size: int = 16,
    feature_dtype: str = "float16",
    overwrite: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    max_duration_seconds: Optional[float] = None,
) -> Tuple[str, str, str, str]:
    """Extract sequence Whisper encoder features and cache to disk."""

    os.makedirs(out_dir, exist_ok=True)
    paths = _cache_paths(out_dir, whisper_size, FEATURE_TYPE_SEQUENCE)
    n = len(samples)
    np_dtype = _resolve_np_dtype(feature_dtype)
    requested_dtype = np.dtype(np_dtype).name
    target_seq_len = _target_encoded_length_from_max_duration(max_duration_seconds)

    if (
        not overwrite
        and os.path.isfile(paths["features"])
        and os.path.isfile(paths["labels"])
        and os.path.isfile(paths["lengths"])
        and os.path.isfile(paths["meta"])
    ):
        meta = _read_cache_meta(paths["meta"])
        existing_feature_type = str(meta.get("feature_type", "")).strip().lower()
        existing_whisper_size = str(meta.get("whisper_size", "")).strip().lower()
        existing_seq_len = int(meta.get("seq_len", meta.get("cache_seq_len", -1)))
        existing_num_samples = int(meta.get("num_samples", -1))
        existing_dtype = str(meta.get("feature_dtype", "")).strip().lower()

        if (
            existing_feature_type == FEATURE_TYPE_SEQUENCE
            and existing_whisper_size == str(whisper_size).strip().lower()
            and existing_seq_len == target_seq_len
            and existing_num_samples == n
            and existing_dtype == requested_dtype
        ):
            return paths["features"], paths["labels"], paths["lengths"], paths["meta"]

        print(
            "检测到旧版或不兼容的 cached_sequence 缓存，"
            "将按当前 max_duration 与 dtype 重新生成。"
        )

    whisper_model = whisper.load_model(whisper_size, device=device)
    encoder = whisper_model.encoder
    encoder.eval()

    full_encoder_seq_len, enc_dim = _infer_encoder_shape(encoder)
    cache_seq_len = min(full_encoder_seq_len, target_seq_len)
    estimated_feature_bytes = n * cache_seq_len * enc_dim * np.dtype(np_dtype).itemsize
    estimated_total_bytes = estimated_feature_bytes + (n * 8) + (n * 4)

    print(
        f"cached_sequence 目标序列长度: {cache_seq_len}/{full_encoder_seq_len} 帧 | "
        f"预计缓存占用: {estimated_total_bytes / (1024 ** 3):.2f} GiB"
    )

    features_mm = np.lib.format.open_memmap(
        paths["features"],
        mode="w+",
        dtype=np_dtype,
        shape=(n, cache_seq_len, enc_dim),
    )
    labels_mm = np.lib.format.open_memmap(paths["labels"], mode="w+", dtype=np.int64, shape=(n,))
    lengths_mm = np.lib.format.open_memmap(paths["lengths"], mode="w+", dtype=np.int32, shape=(n,))
    loader = _build_whisper_dataloader(samples, batch_size, device, num_workers, prefetch_factor)

    with torch.no_grad():
        amp_enabled = device.type == "cuda"
        idx = 0
        for audios, srs, labels in loader:
            mel_batch, encoded_lengths = _build_mel_batch(audios, srs, device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                feats = encoder(mel_batch)
            feats = feats[:, :cache_seq_len, :]
            feats = feats.detach().cpu().numpy().astype(np_dtype, copy=False)
            encoded_lengths = [min(int(length), cache_seq_len) for length in encoded_lengths]

            bsz = int(labels.shape[0])
            features_mm[idx : idx + bsz] = feats
            labels_mm[idx : idx + bsz] = labels.numpy().astype(np.int64, copy=False)
            lengths_mm[idx : idx + bsz] = np.asarray(encoded_lengths, dtype=np.int32)

            idx += bsz
            if idx % 100 == 0 or idx == n:
                print(f"  已提取 {idx}/{n}")

    del features_mm
    del labels_mm
    del lengths_mm

    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "feature_type": FEATURE_TYPE_SEQUENCE,
        "whisper_size": whisper_size,
        "num_samples": n,
        "seq_len": cache_seq_len,
        "full_encoder_seq_len": full_encoder_seq_len,
        "enc_dim": enc_dim,
        "feature_dtype": requested_dtype,
        "max_duration_seconds": float(max_duration_seconds) if max_duration_seconds is not None else None,
    }
    _write_cache_meta(paths["meta"], meta)

    del encoder
    del whisper_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return paths["features"], paths["labels"], paths["lengths"], paths["meta"]


def prepare_whisper_feature_dataset(
    cfg: dict,
    device: torch.device,
    subsets: Iterable[str] = ("ravdess", "casia", "tess", "esd", "emodb", "iemocap"),
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
    feature_dtype: str = "float16",
    overwrite: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    samples: Optional[List[Tuple[str, int]]] = None,
) -> Tuple[Dataset, Dict]:
    """Legacy helper that always returns pooled Whisper features."""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = cfg["paths"]["processed_data"]
    cache_dir = cache_dir or _default_cache_dir(project_root)
    whisper_size = cfg["model"]["whisper_size"]

    from utils.audio_utils import LABEL2ID

    if samples is None:
        raw_samples = build_sample_list(processed_dir, LABEL2ID, subsets=subsets)
        samples, data_policy_audit = filter_supported_samples(raw_samples, cfg)
    else:
        samples = list(samples)
        data_policy_audit = build_data_policy_audit(samples, cfg)
    print(f"共 {len(samples)} 个样本")

    features_path, labels_path, meta_path = extract_whisper_pooled_features(
        samples=samples,
        whisper_size=whisper_size,
        device=device,
        out_dir=cache_dir,
        batch_size=batch_size,
        feature_dtype=feature_dtype,
        overwrite=overwrite,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    meta = _read_cache_meta(meta_path)
    meta["data_policy_audit"] = data_policy_audit
    dataset = WhisperPooledFeatureDataset(features_path, labels_path, feature_dtype=_resolve_np_dtype(feature_dtype))
    return dataset, meta


def prepare_whisper_training_data(
    cfg: dict,
    device: torch.device,
    subsets: Iterable[str] = ("ravdess", "casia", "tess", "esd", "emodb", "iemocap"),
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
    feature_dtype: str = "float16",
    overwrite: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    samples: Optional[List[Tuple[str, int]]] = None,
) -> Tuple[Dataset, Dict]:
    """Prepare Whisper training data according to shared_model.training_mode."""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = cfg["paths"]["processed_data"]
    cache_dir = cache_dir or _default_cache_dir(project_root)
    whisper_size = cfg["model"]["whisper_size"]
    training_mode = str(
        cfg.get("shared_model", {}).get("training_mode", TRAINING_MODE_LIVE_ENCODER)
    ).strip().lower()

    from utils.audio_utils import LABEL2ID

    if samples is None:
        raw_samples = build_sample_list(processed_dir, LABEL2ID, subsets=subsets)
        samples, data_policy_audit = filter_supported_samples(raw_samples, cfg)
    else:
        samples = list(samples)
        data_policy_audit = build_data_policy_audit(samples, cfg)
    print(f"共 {len(samples)} 个样本")

    if training_mode == TRAINING_MODE_LIVE_ENCODER:
        return _AudioPathDataset(samples), {
            "format_version": CACHE_FORMAT_VERSION,
            "feature_type": FEATURE_TYPE_AUDIO,
            "training_mode": TRAINING_MODE_LIVE_ENCODER,
            "whisper_size": whisper_size,
            "num_samples": len(samples),
            "data_policy_audit": data_policy_audit,
        }

    if training_mode == TRAINING_MODE_CACHED_SEQUENCE:
        features_path, labels_path, lengths_path, meta_path = extract_whisper_sequence_features(
            samples=samples,
            whisper_size=whisper_size,
            device=device,
            out_dir=cache_dir,
            batch_size=batch_size,
            feature_dtype=feature_dtype,
            overwrite=overwrite,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            max_duration_seconds=cfg.get("audio", {}).get("max_duration"),
        )
        meta = _read_cache_meta(meta_path)
        meta["training_mode"] = TRAINING_MODE_CACHED_SEQUENCE
        meta["data_policy_audit"] = data_policy_audit
        dataset = WhisperSequenceFeatureDataset(
            features_path,
            labels_path,
            lengths_path,
            feature_dtype=_resolve_np_dtype(feature_dtype),
        )
        return dataset, meta

    if training_mode == TRAINING_MODE_CACHED_POOLED:
        features_path, labels_path, meta_path = extract_whisper_pooled_features(
            samples=samples,
            whisper_size=whisper_size,
            device=device,
            out_dir=cache_dir,
            batch_size=batch_size,
            feature_dtype=feature_dtype,
            overwrite=overwrite,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        meta = _read_cache_meta(meta_path)
        meta["training_mode"] = TRAINING_MODE_CACHED_POOLED
        meta["data_policy_audit"] = data_policy_audit
        dataset = WhisperPooledFeatureDataset(
            features_path,
            labels_path,
            feature_dtype=_resolve_np_dtype(feature_dtype),
        )
        return dataset, meta

    raise ValueError(f"不支持的 shared_model.training_mode: {training_mode}")
