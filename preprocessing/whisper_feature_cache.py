import os
import glob
import json
from typing import Iterable, Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import whisper
import soundfile as sf


def _default_cache_dir(project_root: str) -> str:
    return os.path.join(project_root, "data", "features_shared")


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

        # Use mmap_mode to avoid loading the full array into memory.
        self._features = np.load(features_path, mmap_mode="r")
        self._labels = np.load(labels_path, mmap_mode="r")

        if self._features.shape[0] != self._labels.shape[0]:
            raise ValueError(
                f"features/labels size mismatch: {self._features.shape[0]} vs {self._labels.shape[0]}"
            )

        if self._features.dtype != feature_dtype:
            # Still usable; we just keep track for conversion in __getitem__.
            self._feature_dtype = self._features.dtype
        else:
            self._feature_dtype = feature_dtype

    def __len__(self) -> int:
        return int(self._features.shape[0])

    def __getitem__(self, idx: int):
        # When loaded with mmap_mode='r', slices can be non-writable.
        # Copy to get a writable, contiguous array before converting to torch tensor.
        feat = np.array(self._features[idx], copy=True)
        # Convert to float32 for training stability.
        feat_t = torch.from_numpy(feat).float()
        label_t = torch.tensor(int(self._labels[idx]), dtype=torch.long)
        return feat_t, label_t


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


def _collate_audio(batch):
    audios, srs, labels = zip(*batch)
    return list(audios), list(srs), torch.tensor(labels, dtype=torch.long)


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
    """Extract pooled Whisper encoder features and cache to disk.

    Saves:
      - features: .npy (N, enc_dim) (float16 by default)
      - labels:   .npy (N,) (int64)
      - meta:     .json

    Returns (features_path, labels_path, meta_path).
    """

    os.makedirs(out_dir, exist_ok=True)

    features_path = os.path.join(out_dir, f"whisper_{whisper_size}_pooled_features.npy")
    labels_path = os.path.join(out_dir, f"whisper_{whisper_size}_pooled_labels.npy")
    meta_path = os.path.join(out_dir, f"whisper_{whisper_size}_pooled_meta.json")

    if (
        not overwrite
        and os.path.isfile(features_path)
        and os.path.isfile(labels_path)
        and os.path.isfile(meta_path)
    ):
        return features_path, labels_path, meta_path

    # Load Whisper once.
    whisper_model = whisper.load_model(whisper_size, device=device)
    encoder = whisper_model.encoder
    encoder.eval()

    enc_dim = int(encoder.ln_post.normalized_shape[0])
    n = len(samples)

    np_dtype = np.float16 if feature_dtype == "float16" else np.float32

    # Memmap for incremental writing without holding everything in RAM.
    features_mm = np.lib.format.open_memmap(features_path, mode="w+", dtype=np_dtype, shape=(n, enc_dim))
    labels_mm = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(n,))

    pool = nn.AdaptiveAvgPool1d(1)

    ds = _AudioPathDataset(samples)
    pin_memory = device.type == "cuda"
    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
        "collate_fn": _collate_audio,
        "persistent_workers": bool(num_workers) if int(num_workers) > 0 else False,
    }
    if int(num_workers) > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(ds, **dl_kwargs)

    with torch.no_grad():
        amp_enabled = device.type == "cuda"
        idx = 0
        for audios, srs, labels in loader:
            mels = []
            for audio, sr in zip(audios, srs):
                # processed wav should already be 16k/mono; keep a safe fallback.
                if sr != 16000:
                    # Slow path: resample would require extra deps; rely on prior preprocessing.
                    raise ValueError(f"Unexpected sample_rate={sr} for {idx}; expected 16000")
                audio_t = torch.from_numpy(audio)
                if amp_enabled:
                    audio_t = audio_t.to(device)
                audio_t = whisper.pad_or_trim(audio_t)
                mel = whisper.log_mel_spectrogram(audio_t)
                mels.append(mel)

            mel_batch = torch.stack(mels, dim=0)
            if mel_batch.device != device:
                mel_batch = mel_batch.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                feats = encoder(mel_batch)  # (B, time, enc_dim)
            feats = pool(feats.permute(0, 2, 1)).squeeze(-1)  # (B, enc_dim)
            feats = feats.detach().cpu().numpy().astype(np_dtype, copy=False)

            bsz = int(labels.shape[0])
            features_mm[idx : idx + bsz] = feats
            labels_mm[idx : idx + bsz] = labels.numpy().astype(np.int64, copy=False)

            idx += bsz
            if idx % 200 == 0 or idx == n:
                print(f"  已提取 {idx}/{n}")

    # Flush memmaps.
    del features_mm
    del labels_mm

    meta = {
        "whisper_size": whisper_size,
        "num_samples": n,
        "enc_dim": enc_dim,
        "feature_dtype": feature_dtype,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    del encoder
    del whisper_model
    torch.cuda.empty_cache()

    return features_path, labels_path, meta_path


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
) -> Tuple[Dataset, Dict]:
    """Build dataset of pooled Whisper encoder features with on-disk caching."""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = cfg["paths"]["processed_data"]

    cache_dir = cache_dir or _default_cache_dir(project_root)
    whisper_size = cfg["model"]["whisper_size"]

    from utils.audio_utils import LABEL2ID

    samples = build_sample_list(processed_dir, LABEL2ID, subsets=subsets)
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

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dataset = WhisperPooledFeatureDataset(features_path, labels_path, feature_dtype=np.float16)
    return dataset, meta
