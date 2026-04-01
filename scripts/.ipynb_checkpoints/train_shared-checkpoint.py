#!/usr/bin/env python3
"""脚本化主线训练入口：Whisper + Transformer Emotion Head。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.whisper_emotion import (
    DEFAULT_SHARED_MODEL_CONFIG,
    build_shared_model_from_config,
    create_shared_checkpoint,
)
from preprocessing.feature_extract import AudioAugmentation
from preprocessing.whisper_feature_cache import (
    TRAINING_MODE_LIVE_ENCODER,
    build_sample_list,
    build_whisper_mel_batch,
    collate_whisper_audio_batch,
    prepare_whisper_training_data,
)
from utils.audio_utils import LABEL2ID, EMOTION_LABELS, load_config
from utils.data_policy import filter_supported_samples
from utils.losses import FocalLoss
from utils.split_utils import (
    audit_subset_groups,
    build_group_holdout_folds,
    infer_subset_from_path,
    speaker_group_split,
    validate_subset_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Whisper + Transformer Emotion Head 正式主线。")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="配置文件路径。")
    parser.add_argument("--norm", choices=("derf", "dyt"), help="正式主线的 norm-free 变体。")
    parser.add_argument("--seed", type=int, help="覆盖 training.seed。")
    parser.add_argument(
        "--profile",
        choices=("auto", "cpu_preflight", "cuda_4090_mainline"),
        default="auto",
        help="运行 profile；auto 会根据 CUDA 自动选择。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="设备选择；默认 auto。",
    )
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数。")
    parser.add_argument("--audit-only", action="store_true", help="只执行数据审计与 split，不启动训练。")
    parser.add_argument("--smoke", action="store_true", help="执行短程 smoke run。")
    parser.add_argument(
        "--smoke-target-per-subset",
        type=int,
        help="smoke 模式下每个主子集的最大样本数。默认读取 runtime.cpu_preflight.smoke_target_per_subset。",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("请求使用 CUDA，但当前环境不可用")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_profile(cfg: dict, requested_profile: str, device: torch.device) -> str:
    if requested_profile != "auto":
        return requested_profile
    runtime_cfg = cfg.get("runtime", {})
    default_profile = str(runtime_cfg.get("default_profile", "auto")).strip().lower()
    if default_profile in {"cpu_preflight", "cuda_4090_mainline"}:
        if default_profile == "cuda_4090_mainline" and device.type != "cuda":
            return "cpu_preflight"
        return default_profile
    return "cuda_4090_mainline" if device.type == "cuda" else "cpu_preflight"


def apply_runtime_profile(cfg: dict, profile: str, device: torch.device, args: argparse.Namespace) -> Dict[str, Any]:
    runtime_cfg = cfg.get("runtime", {})
    profile_cfg = dict(runtime_cfg.get(profile, {}))
    training_cfg = cfg.setdefault("training", {})
    model_cfg = cfg.setdefault("model", {})
    configured_whisper_size = str(model_cfg.get("whisper_size", ""))
    effective_whisper_size = configured_whisper_size
    whisper_size_override = profile_cfg.get("whisper_size_override")
    if whisper_size_override:
        effective_whisper_size = str(whisper_size_override).strip()
        model_cfg["whisper_size"] = effective_whisper_size

    override_keys = (
        "whisper_feature_batch_size",
        "whisper_feature_num_workers",
        "whisper_feature_prefetch_factor",
        "live_encoder_batch_size",
        "live_encoder_eval_batch_size",
        "live_encoder_num_workers",
        "live_encoder_eval_num_workers",
        "live_encoder_prefetch_factor",
        "live_encoder_eval_prefetch_factor",
        "live_encoder_persistent_workers",
        "live_encoder_eval_persistent_workers",
    )
    for key in override_keys:
        if key in profile_cfg:
            training_cfg[key] = profile_cfg[key]

    smoke_enabled = bool(args.smoke)
    if profile == "cpu_preflight":
        smoke_enabled = True

    if args.epochs is not None:
        training_cfg["epochs"] = int(args.epochs)
    elif profile == "cpu_preflight" and "max_epochs" in profile_cfg:
        training_cfg["epochs"] = min(int(training_cfg.get("epochs", 1)), int(profile_cfg["max_epochs"]))

    smoke_target = args.smoke_target_per_subset
    if smoke_target is None and smoke_enabled:
        smoke_target = profile_cfg.get("smoke_target_per_subset")
    if smoke_target is not None:
        smoke_target = int(smoke_target)
        subset_targets = dict(training_cfg.get("subset_epoch_targets", {}))
        for subset, value in list(subset_targets.items()):
            subset_targets[subset] = min(int(value), smoke_target)
        training_cfg["subset_epoch_targets"] = subset_targets
        training_cfg["subset_epoch_caps"] = dict(subset_targets)

    return {
        "profile": profile,
        "device": str(device),
        "smoke_enabled": smoke_enabled,
        "smoke_target_per_subset": smoke_target,
        "configured_whisper_size": configured_whisper_size,
        "effective_whisper_size": effective_whisper_size,
    }


def enforce_mainline_shared_cfg(cfg: dict, norm: Optional[str]) -> None:
    shared_cfg = cfg.setdefault("shared_model", {})
    shared_cfg["variant"] = "transformer_head"
    shared_cfg["training_mode"] = TRAINING_MODE_LIVE_ENCODER
    shared_cfg["pooling"] = "attention"
    shared_cfg["freeze_strategy"] = "unfreeze_last_2"
    if norm is not None:
        shared_cfg["norm"] = str(norm).strip().lower()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_training_value(cfg: dict, key: str, default=None):
    return cfg.get("training", {}).get(key, default)


def resolve_subset_target(subset: str, raw_count: int, subset_caps: Dict[str, int]) -> int:
    configured_cap = int(subset_caps.get(subset, raw_count))
    if configured_cap <= 0:
        return int(raw_count)
    return int(min(raw_count, configured_cap))


def compress_holdout_plan(plan: dict) -> dict:
    compact = dict(plan)
    compact["folds"] = [
        {key: value for key, value in fold.items() if not key.endswith("_indices")}
        for fold in plan.get("folds", [])
    ]
    return compact


def compute_subset_sampling_distribution(
    labels: Sequence[int],
    num_classes: int,
    sampling_mode: str,
    balance_power: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    subset_class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    raw_count = int(labels.size)
    if raw_count == 0:
        return np.asarray([], dtype=np.float64), subset_class_counts, np.zeros(num_classes, dtype=np.float64)

    if sampling_mode == "balanced_class_aware":
        class_weights = np.zeros(num_classes, dtype=np.float64)
        nonzero = subset_class_counts > 0
        class_weights[nonzero] = np.power(subset_class_counts[nonzero], -float(balance_power))
        sample_weights = class_weights[labels]
    else:
        sample_weights = np.ones(raw_count, dtype=np.float64)

    sample_probs = sample_weights / sample_weights.sum()
    class_probs = np.zeros(num_classes, dtype=np.float64)
    for class_idx in range(num_classes):
        class_probs[class_idx] = float(sample_probs[labels == class_idx].sum())
    return sample_probs, subset_class_counts, class_probs


def compute_virtual_subset_class_counts(
    subset_label_map: Dict[str, np.ndarray],
    subset_caps: Dict[str, int],
    subset_order: Sequence[str],
    num_classes: int,
    sampling_mode: str,
    balance_power: float,
    with_replacement: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    class_counts = np.zeros(num_classes, dtype=np.float64)
    raw_subset_counts = {}
    effective_subset_counts = {}
    oversampled_subset_counts = {}

    for subset in subset_order:
        labels = np.asarray(subset_label_map.get(subset, np.asarray([], dtype=np.int64)), dtype=np.int64)
        raw_count = int(labels.size)
        target_count = resolve_subset_target(subset, raw_count, subset_caps)
        effective_count = target_count if (with_replacement or target_count <= raw_count) else raw_count
        raw_subset_counts[subset] = raw_count
        effective_subset_counts[subset] = int(effective_count)
        oversampled_subset_counts[subset] = int(max(0, effective_count - raw_count))

        if raw_count == 0 or effective_count <= 0:
            continue

        _, subset_class_counts, class_probs = compute_subset_sampling_distribution(
            labels,
            num_classes,
            sampling_mode,
            balance_power,
        )
        if sampling_mode == "uniform" or not with_replacement:
            scaled_counts = subset_class_counts * (float(min(effective_count, raw_count)) / float(raw_count))
            if effective_count > raw_count and with_replacement:
                scaled_counts += class_probs * float(effective_count - raw_count)
            class_counts += scaled_counts
        else:
            class_counts += class_probs * float(effective_count)

    return class_counts, {
        "subset_targets": {subset: int(resolve_subset_target(subset, raw_subset_counts.get(subset, 0), subset_caps)) for subset in subset_order},
        "raw_subset_counts": raw_subset_counts,
        "effective_subset_counts": effective_subset_counts,
        "oversampled_subset_counts": oversampled_subset_counts,
        "effective_train_samples_per_epoch": int(sum(effective_subset_counts.values())),
        "effective_esd_samples_per_epoch": int(effective_subset_counts.get("esd", 0)),
        "sampling_mode": sampling_mode,
        "sampling_with_replacement": bool(with_replacement),
        "class_balance_power": float(balance_power),
    }


def compute_class_weights(class_counts: np.ndarray, num_classes: int) -> np.ndarray:
    clipped = np.maximum(np.asarray(class_counts, dtype=np.float64), 1e-6)
    weights = 1.0 / clipped
    weights = weights / weights.sum() * num_classes
    return weights.astype(np.float32)


def format_subset_counts(counts: Dict[str, int], subset_order: Sequence[str]) -> str:
    entries = [f"{subset}={int(counts.get(subset, 0))}" for subset in subset_order if int(counts.get(subset, 0)) > 0]
    return ", ".join(entries) if entries else "none"


def sample_subset_relative_indices(
    relative_indices: np.ndarray,
    relative_labels: np.ndarray,
    target_count: int,
    num_classes: int,
    sampling_mode: str,
    balance_power: float,
    rng: np.random.Generator,
    with_replacement: bool,
) -> np.ndarray:
    relative_indices = np.asarray(relative_indices, dtype=np.int64)
    relative_labels = np.asarray(relative_labels, dtype=np.int64)
    raw_count = int(len(relative_indices))
    if raw_count == 0 or target_count <= 0:
        return np.asarray([], dtype=np.int64)

    sample_probs, _, _ = compute_subset_sampling_distribution(
        relative_labels,
        num_classes,
        sampling_mode,
        balance_power,
    )

    if target_count <= raw_count:
        selected = rng.choice(relative_indices, size=target_count, replace=False, p=sample_probs)
        return np.asarray(selected, dtype=np.int64)

    selected_parts = [relative_indices.copy()]
    if with_replacement and target_count > raw_count:
        extra = rng.choice(relative_indices, size=target_count - raw_count, replace=True, p=sample_probs)
        selected_parts.append(np.asarray(extra, dtype=np.int64))

    return np.concatenate(selected_parts).astype(np.int64, copy=False)


def build_epoch_train_relative_indices(
    epoch: int,
    seed: int,
    subset_relative_indices: Dict[str, np.ndarray],
    subset_relative_labels: Dict[str, np.ndarray],
    subset_caps: Dict[str, int],
    subset_order: Sequence[str],
    num_classes: int,
    sampling_mode: str,
    balance_power: float,
    with_replacement: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    selected_parts = []
    effective_subset_counts = {}
    oversampled_subset_counts = {}

    for subset_offset, subset in enumerate(subset_order):
        relative_indices = subset_relative_indices.get(subset, np.asarray([], dtype=np.int64))
        relative_labels = subset_relative_labels.get(subset, np.asarray([], dtype=np.int64))
        raw_count = int(len(relative_indices))
        target_count = resolve_subset_target(subset, raw_count, subset_caps)

        subset_rng = np.random.default_rng(int(seed) * 1009 + int(epoch) * 97 + subset_offset * 17)
        selected = sample_subset_relative_indices(
            relative_indices,
            relative_labels,
            target_count,
            num_classes,
            sampling_mode,
            balance_power,
            subset_rng,
            with_replacement,
        )
        effective_subset_counts[subset] = int(len(selected))
        oversampled_subset_counts[subset] = int(max(0, len(selected) - raw_count))
        if len(selected) > 0:
            selected_parts.append(selected)

    if selected_parts:
        combined = np.concatenate(selected_parts).astype(np.int64, copy=False)
        shuffle_rng = np.random.default_rng(int(seed) * 2003 + int(epoch))
        shuffled = shuffle_rng.permutation(combined)
    else:
        shuffled = np.asarray([], dtype=np.int64)

    return shuffled.tolist(), {
        "effective_subset_counts": effective_subset_counts,
        "oversampled_subset_counts": oversampled_subset_counts,
        "effective_train_samples": int(len(shuffled)),
        "effective_esd_samples": int(effective_subset_counts.get("esd", 0)),
    }


def build_criterion(class_weights_tensor: torch.Tensor, class_weights: np.ndarray, cfg: dict):
    label_smoothing = float(get_training_value(cfg, "live_encoder_label_smoothing", get_training_value(cfg, "label_smoothing", 0.0)))
    alpha = [float(value) for value in class_weights.tolist()]
    if bool(get_training_value(cfg, "live_encoder_focal_loss", False)):
        gamma = float(get_training_value(cfg, "live_encoder_focal_gamma", 2.0))
        criterion = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing).to(class_weights_tensor.device)
        return criterion, "focal_loss", {
            "gamma": float(gamma),
            "label_smoothing": float(label_smoothing),
            "alpha": alpha,
            "weighted": True,
        }

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing).to(class_weights_tensor.device)
    return criterion, "cross_entropy", {
        "label_smoothing": float(label_smoothing),
        "weight": alpha,
        "weighted": True,
    }


def build_optimizer(model, cfg: dict):
    default_lr = float(get_training_value(cfg, "learning_rate", 2e-4))
    head_lr = float(get_training_value(cfg, "live_encoder_head_learning_rate", get_training_value(cfg, "head_learning_rate", default_lr)))
    encoder_lr = float(get_training_value(cfg, "live_encoder_encoder_learning_rate", get_training_value(cfg, "encoder_learning_rate", max(1e-5, head_lr * 0.1))))
    weight_decay = float(get_training_value(cfg, "live_encoder_weight_decay", get_training_value(cfg, "weight_decay", 1e-4)))

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"name": "head", "params": head_params, "lr": head_lr})
    if encoder_params:
        param_groups.append({"name": "encoder", "params": encoder_params, "lr": encoder_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    return optimizer, base_lrs


def apply_warmup(optimizer, base_lrs: Sequence[float], epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0:
        return
    scale = min(1.0, float(epoch) / float(warmup_epochs))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * scale


def format_group_lrs(optimizer) -> str:
    return ", ".join(
        f"{group.get('name', f'group_{idx}')}={group['lr']:.2e}"
        for idx, group in enumerate(optimizer.param_groups)
    )


def apply_live_encoder_augmentation(mel_batch: torch.Tensor, augmenter: Optional[AudioAugmentation]) -> torch.Tensor:
    if augmenter is None:
        return mel_batch

    augmented = mel_batch.clone()
    batch_size, n_freq, n_time = augmented.shape
    for sample_idx in range(batch_size):
        sample = augmented[sample_idx]
        time_mask_max = int(getattr(augmenter, "time_mask_max", 0))
        freq_mask_max = int(getattr(augmenter, "freq_mask_max", 0))

        t = np.random.randint(0, time_mask_max + 1)
        if t > 0 and n_time > t:
            t0 = np.random.randint(0, n_time - t + 1)
            sample[:, t0:t0 + t] = sample.min()

        f = np.random.randint(0, freq_mask_max + 1)
        if f > 0 and n_freq > f:
            f0 = np.random.randint(0, n_freq - f + 1)
            sample[f0:f0 + f, :] = sample.min()

    noise_std = float(getattr(augmenter, "noise_std", 0.0))
    if noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented


def move_batch_to_device(batch, device: torch.device, augmenter: Optional[AudioAugmentation], is_training: bool = False):
    audios, srs, labels = batch
    labels = labels.to(device, non_blocking=True)
    mel_batch, attention_mask = build_whisper_mel_batch(audios, srs, device)
    if is_training:
        mel_batch = apply_live_encoder_augmentation(mel_batch, augmenter)
    return {
        "mel": mel_batch,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_epoch(model, loader, criterion, optimizer, device: torch.device, augmenter: Optional[AudioAugmentation], scaler=None, grad_accum_steps: int = 1):
    model.train(True)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    use_amp = device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch_inputs = move_batch_to_device(batch, device, augmenter, is_training=True)
        labels = batch_inputs["labels"]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(mel=batch_inputs["mel"], attention_mask=batch_inputs["attention_mask"])
            loss = criterion(logits, labels)

        batch_size = int(labels.size(0))
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

        loss_for_backward = loss / grad_accum_steps
        if scaler is not None and use_amp:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (step % grad_accum_steps == 0) or (step == len(loader))
        if should_step:
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def evaluate_epoch(model, loader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_labels = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            batch_inputs = move_batch_to_device(batch, device, augmenter=None, is_training=False)
            labels = batch_inputs["labels"]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(mel=batch_inputs["mel"], attention_mask=batch_inputs["attention_mask"])
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            batch_size = int(labels.size(0))
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_count += batch_size
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "uar": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def evaluate_subset_loaders(model, loaders_by_subset, criterion, device: torch.device):
    results = {}
    for subset, loader in loaders_by_subset.items():
        metrics = evaluate_epoch(model, loader, criterion, device)
        results[subset] = {
            "samples": int(len(loader.dataset)),
            "loss": float(metrics["loss"]),
            "acc": float(metrics["acc"]),
            "macro_f1": float(metrics["macro_f1"]),
            "uar": float(metrics["uar"]),
        }
    return results


def compute_subset_mean_metric(metrics_by_subset: Dict[str, Dict[str, float]], metric_key: str = "uar", subset_order: Optional[Sequence[str]] = None) -> float:
    subset_order = list(subset_order or metrics_by_subset.keys())
    values = [float(metrics_by_subset[subset][metric_key]) for subset in subset_order if subset in metrics_by_subset]
    if not values:
        return float("nan")
    return float(np.mean(values))


def sanitize_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def build_experiment_name(model_variant: str, training_mode: str, model_shared_config: dict) -> str:
    parts = ["shared", sanitize_token(model_variant), sanitize_token(training_mode)]
    parts.extend([
        sanitize_token(model_shared_config.get("pooling", "attention")),
        sanitize_token(model_shared_config.get("norm", "derf")),
        sanitize_token(model_shared_config.get("freeze_strategy", "unfreeze_last_2")),
    ])
    return "_".join(parts)


def decorate_experiment_stem(experiment_stem: str, runtime_meta: Dict[str, Any]) -> str:
    suffixes = []
    profile = sanitize_token(runtime_meta.get("profile", ""))
    if profile and profile != "cuda_4090_mainline":
        suffixes.append(profile)
    if bool(runtime_meta.get("smoke_enabled")):
        suffixes.append("smoke")
    return experiment_stem if not suffixes else f"{experiment_stem}_{'_'.join(suffixes)}"


def build_run_name(experiment_stem: str, seed: int) -> str:
    return f"{experiment_stem}_seed{int(seed)}"


def compute_subset_mean_from_summary(summary: dict, summary_key: str) -> Optional[float]:
    if summary_key in summary:
        return float(summary[summary_key])
    if summary_key == "selected_val_subset_mean_uar":
        metrics = summary.get("per_subset_val_metrics", {})
    elif summary_key == "test_subset_mean_uar":
        metrics = summary.get("per_subset_test_metrics", {})
    else:
        return None
    values = [float(item.get("uar", float("nan"))) for item in metrics.values() if "uar" in item]
    if not values:
        return None
    return float(np.mean(values))


AGGREGATE_METRIC_KEYS = (
    "best_val_loss",
    "best_val_acc",
    "selected_val_uar",
    "selected_val_subset_mean_uar",
    "test_acc",
    "test_subset_mean_uar",
    "macro_f1",
    "weighted_f1",
    "uar",
)


def aggregate_experiment_summaries(experiment_stem: str, seeds: Sequence[int], ckpt_dir: Path, protocol_version: int) -> Optional[dict]:
    loaded = []
    for seed_value in seeds:
        summary_path = ckpt_dir / f"{build_run_name(experiment_stem, seed_value)}_summary.json"
        if not summary_path.is_file():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if int(summary.get("protocol_version", -1)) != int(protocol_version):
            continue
        loaded.append((int(seed_value), summary_path, summary))

    if not loaded:
        return None

    aggregate_mean_std = {}
    aggregate_summary = {
        "protocol_version": int(protocol_version),
        "experiment_stem": experiment_stem,
        "num_runs": int(len(loaded)),
        "num_seeds": int(len(loaded)),
        "seeds": [seed_value for seed_value, _, _ in loaded],
        "source_summary_filenames": [summary_path.name for _, summary_path, _ in loaded],
        "aggregate_mean_std": aggregate_mean_std,
        "included_subsets": loaded[0][2].get("included_subsets", []),
        "auxiliary_subsets": loaded[0][2].get("auxiliary_subsets", []),
    }

    for key in AGGREGATE_METRIC_KEYS:
        values = []
        for _, _, summary in loaded:
            if key in summary:
                values.append(float(summary[key]))
                continue
            derived = compute_subset_mean_from_summary(summary, key)
            if derived is not None:
                values.append(float(derived))
        if not values:
            continue
        aggregate_mean_std[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
        aggregate_summary[key] = float(np.mean(values))

    aggregate_path = ckpt_dir / f"{experiment_stem}_aggregate.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate_summary, handle, ensure_ascii=False, indent=2)
    return aggregate_summary


def build_subset_eval_loaders(dataset, sample_subsets: np.ndarray, global_indices: Sequence[int], subset_order: Sequence[str], loader_kwargs: dict):
    loaders = {}
    subset_counts = {}
    for subset in subset_order:
        subset_indices = [int(idx) for idx in global_indices if sample_subsets[idx] == subset]
        subset_counts[subset] = int(len(subset_indices))
        if subset_indices:
            loaders[subset] = DataLoader(Subset(dataset, subset_indices), shuffle=False, **loader_kwargs)
    return loaders, subset_counts


def cap_eval_indices_by_subset(
    global_indices: Sequence[int],
    sample_subsets: np.ndarray,
    subset_order: Sequence[str],
    per_subset_limit: Optional[int],
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    indices = [int(idx) for idx in global_indices]
    raw_subset_counts = {}
    effective_subset_counts = {}

    if per_subset_limit is None or int(per_subset_limit) <= 0:
        for subset in subset_order:
            subset_count = int(sum(1 for idx in indices if sample_subsets[idx] == subset))
            raw_subset_counts[subset] = subset_count
            effective_subset_counts[subset] = subset_count
        return indices, {
            "applied": False,
            "limit_per_subset": None,
            "raw_subset_counts": raw_subset_counts,
            "effective_subset_counts": effective_subset_counts,
        }

    selected_indices = []
    limit = int(per_subset_limit)
    for subset_offset, subset in enumerate(subset_order):
        subset_indices = np.asarray([int(idx) for idx in indices if sample_subsets[idx] == subset], dtype=np.int64)
        raw_subset_counts[subset] = int(len(subset_indices))
        if len(subset_indices) > limit:
            subset_rng = np.random.default_rng(int(seed) * 4001 + subset_offset * 131 + limit)
            subset_indices = np.sort(subset_rng.choice(subset_indices, size=limit, replace=False))
        effective_subset_counts[subset] = int(len(subset_indices))
        selected_indices.extend(subset_indices.tolist())

    return selected_indices, {
        "applied": True,
        "limit_per_subset": limit,
        "raw_subset_counts": raw_subset_counts,
        "effective_subset_counts": effective_subset_counts,
    }


def save_curves(history: Dict[str, List[float]], output_path: Path, experiment_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    epochs_range = range(1, len(history["val_loss"]) + 1)
    axes[0, 0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs_range, history["val_loss"], label="Validation")
    axes[0, 0].set_title(f"Loss - {experiment_name}")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_range, history["train_acc"], label="Train")
    axes[0, 1].plot(epochs_range, history["val_acc"], label="Validation")
    axes[0, 1].set_title(f"Accuracy - {experiment_name}")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs_range, history["val_macro_f1"], label="Validation Macro-F1", color="#2E8B57")
    axes[1, 0].set_title(f"Validation Macro-F1 - {experiment_name}")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs_range, history["val_uar"], label="Validation UAR", color="#DC143C")
    axes[1, 1].plot(epochs_range, history["val_subset_mean_uar"], label="Validation Subset Mean UAR", color="#1E90FF")
    axes[1, 1].set_title(f"Validation UAR - {experiment_name}")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(all_labels: Sequence[int], all_preds: Sequence[int], output_path: Path, experiment_name: str) -> None:
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EMOTION_LABELS))))
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=EMOTION_LABELS).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {experiment_name}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(str(args.config))
    device = resolve_device(args.device)
    runtime_meta = apply_runtime_profile(cfg, resolve_profile(cfg, args.profile, device), device, args)
    enforce_mainline_shared_cfg(cfg, args.norm)

    seed = int(args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42))
    cfg.setdefault("training", {})["seed"] = seed
    set_seed(seed)

    shared_cfg = cfg["shared_model"]
    training_mode = TRAINING_MODE_LIVE_ENCODER
    protocol_version = int(get_training_value(cfg, "protocol_version", 2))
    whisper_size = cfg["model"]["whisper_size"]
    num_classes = int(cfg["emotion"]["num_classes"])
    seed_sweep = [int(value) for value in get_training_value(cfg, "seed_sweep", [seed])]

    default_main_subsets = ("ravdess", "casia", "esd", "emodb", "iemocap")
    default_auxiliary_subsets = ("tess",)
    main_subsets = tuple(str(subset).strip().lower() for subset in get_training_value(cfg, "main_subsets", default_main_subsets))
    auxiliary_subsets = tuple(str(subset).strip().lower() for subset in get_training_value(cfg, "auxiliary_subsets", default_auxiliary_subsets))
    requested_subset_set = set(main_subsets) | set(auxiliary_subsets)
    subsets = tuple(subset for subset in ("ravdess", "casia", "tess", "esd", "emodb", "iemocap") if subset in requested_subset_set)

    feature_bs = int(get_training_value(cfg, "whisper_feature_batch_size", 8))
    feature_workers = int(get_training_value(cfg, "whisper_feature_num_workers", 0))
    feature_prefetch = int(get_training_value(cfg, "whisper_feature_prefetch_factor", 2))
    cache_dtype = str(shared_cfg.get("cache_feature_dtype", "float16"))

    raw_samples = build_sample_list(cfg["paths"]["processed_data"], LABEL2ID, subsets=subsets)
    samples, data_policy_audit = filter_supported_samples(raw_samples, cfg)

    dataset, data_meta = prepare_whisper_training_data(
        cfg=cfg,
        device=device,
        subsets=subsets,
        batch_size=feature_bs,
        feature_dtype=cache_dtype,
        overwrite=False,
        num_workers=feature_workers,
        prefetch_factor=feature_prefetch,
        samples=samples,
    )
    data_meta["data_policy_audit"] = data_policy_audit

    sample_paths = [wav_path for wav_path, _ in samples]
    sample_labels = np.asarray([label for _, label in samples], dtype=np.int64)
    sample_subsets = np.asarray([infer_subset_from_path(path) for path in sample_paths], dtype=object)

    train_ratio = float(cfg["training"]["train_ratio"])
    val_ratio = float(cfg["training"]["val_ratio"])
    test_ratio = float(cfg["training"]["test_ratio"])

    dataset_audit = audit_subset_groups(sample_paths, train_ratio, val_ratio, test_ratio, subsets=subsets)
    main_subset_audit = validate_subset_groups(sample_paths, train_ratio, val_ratio, test_ratio, subsets=main_subsets)

    auxiliary_eval_plan = {}
    for subset in auxiliary_subsets:
        try:
            auxiliary_eval_plan[subset] = compress_holdout_plan(build_group_holdout_folds(sample_paths, subset))
        except ValueError as exc:
            auxiliary_eval_plan[subset] = {"subset": subset, "error": str(exc)}

    train_indices, val_indices, test_indices, split_meta = speaker_group_split(
        sample_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        include_subsets=main_subsets,
    )
    split_meta.update({
        "protocol_version": int(protocol_version),
        "seed": int(seed),
        "seed_sweep": list(seed_sweep),
        "main_subsets": list(main_subsets),
        "auxiliary_subsets": list(auxiliary_subsets),
        "dataset_audit": dataset_audit["subset_audit"],
        "dataset_normalization_warnings": dataset_audit["normalization_warnings"],
        "auxiliary_eval_plan": auxiliary_eval_plan,
        "data_policy_audit": data_policy_audit,
    })

    subset_caps = dict(get_training_value(cfg, "subset_epoch_targets", {}))
    subset_sampling_mode = str(get_training_value(cfg, "subset_sampling_mode", "balanced_class_aware")).strip().lower()
    subset_sampling_with_replacement = bool(get_training_value(cfg, "subset_sampling_with_replacement", False))
    subset_class_balance_power = float(get_training_value(cfg, "subset_class_balance_power", 0.5))

    eval_subset_limit = runtime_meta.get("smoke_target_per_subset") if runtime_meta.get("smoke_enabled") else None
    effective_val_indices, val_runtime_eval_sampling = cap_eval_indices_by_subset(
        val_indices,
        sample_subsets,
        main_subsets,
        eval_subset_limit,
        seed + 11,
    )
    effective_test_indices, test_runtime_eval_sampling = cap_eval_indices_by_subset(
        test_indices,
        sample_subsets,
        main_subsets,
        eval_subset_limit,
        seed + 29,
    )

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, effective_val_indices)
    test_set = Subset(dataset, effective_test_indices)

    train_sample_labels = sample_labels[train_indices]
    train_sample_subsets = sample_subsets[train_indices]
    raw_train_class_counts = np.bincount(train_sample_labels, minlength=num_classes).astype(np.int64)

    train_relative_indices_by_subset = {}
    train_relative_labels_by_subset = {}
    raw_train_subset_counts = {}
    for subset in main_subsets:
        relative_indices = np.asarray([idx for idx, sample_subset in enumerate(train_sample_subsets) if sample_subset == subset], dtype=np.int64)
        train_relative_indices_by_subset[subset] = relative_indices
        train_relative_labels_by_subset[subset] = train_sample_labels[relative_indices]
        raw_train_subset_counts[subset] = int(len(relative_indices))

    virtual_train_class_counts, train_sampling_meta = compute_virtual_subset_class_counts(
        train_relative_labels_by_subset,
        subset_caps,
        main_subsets,
        num_classes,
        subset_sampling_mode,
        subset_class_balance_power,
        subset_sampling_with_replacement,
    )
    class_weights = compute_class_weights(virtual_train_class_counts, num_classes)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    class_weights_dict = {label: float(weight) for label, weight in zip(cfg["emotion"]["labels"], class_weights)}
    virtual_train_class_counts_dict = {label: float(count) for label, count in zip(cfg["emotion"]["labels"], virtual_train_class_counts)}

    base_batch_size = int(get_training_value(cfg, "batch_size", 16))
    train_batch_size = int(get_training_value(cfg, "live_encoder_batch_size", min(base_batch_size, 16 if device.type == "cuda" else 4)))
    eval_batch_size = int(get_training_value(cfg, "live_encoder_eval_batch_size", train_batch_size))
    train_num_workers = int(get_training_value(cfg, "live_encoder_num_workers", 0))
    eval_num_workers = int(get_training_value(cfg, "live_encoder_eval_num_workers", train_num_workers))
    train_prefetch_factor = int(get_training_value(cfg, "live_encoder_prefetch_factor", 2))
    eval_prefetch_factor = int(get_training_value(cfg, "live_encoder_eval_prefetch_factor", train_prefetch_factor))
    train_persistent_workers = bool(get_training_value(cfg, "live_encoder_persistent_workers", train_num_workers > 0))
    eval_persistent_workers = bool(get_training_value(cfg, "live_encoder_eval_persistent_workers", eval_num_workers > 0))

    pin_memory = device.type == "cuda"
    train_loader_kwargs = {
        "batch_size": train_batch_size,
        "pin_memory": pin_memory,
        "num_workers": train_num_workers,
        "collate_fn": collate_whisper_audio_batch,
    }
    eval_loader_kwargs = {
        "batch_size": eval_batch_size,
        "pin_memory": pin_memory,
        "num_workers": eval_num_workers,
        "collate_fn": collate_whisper_audio_batch,
    }
    if train_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = train_persistent_workers
        train_loader_kwargs["prefetch_factor"] = train_prefetch_factor
    if eval_num_workers > 0:
        eval_loader_kwargs["persistent_workers"] = eval_persistent_workers
        eval_loader_kwargs["prefetch_factor"] = eval_prefetch_factor

    def build_epoch_train_loader(epoch: int):
        effective_indices, epoch_sampling = build_epoch_train_relative_indices(
            epoch,
            seed,
            train_relative_indices_by_subset,
            train_relative_labels_by_subset,
            subset_caps,
            main_subsets,
            num_classes,
            subset_sampling_mode,
            subset_class_balance_power,
            subset_sampling_with_replacement,
        )
        effective_train_set = Subset(train_set, effective_indices)
        loader = DataLoader(effective_train_set, shuffle=False, **train_loader_kwargs)
        return loader, epoch_sampling

    val_loader = DataLoader(val_set, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **eval_loader_kwargs)
    val_loaders_by_subset, val_subset_eval_counts = build_subset_eval_loaders(dataset, sample_subsets, effective_val_indices, main_subsets, eval_loader_kwargs)
    test_loaders_by_subset, test_subset_eval_counts = build_subset_eval_loaders(dataset, sample_subsets, effective_test_indices, main_subsets, eval_loader_kwargs)

    audit_payload = {
        "protocol_version": int(protocol_version),
        "seed": int(seed),
        "device": str(device),
        "runtime_profile": runtime_meta,
        "shared_model": {
            "variant": shared_cfg["variant"],
            "training_mode": shared_cfg["training_mode"],
            "pooling": shared_cfg["pooling"],
            "freeze_strategy": shared_cfg["freeze_strategy"],
            "norm": shared_cfg["norm"],
        },
        "data_meta": data_meta,
        "data_policy_audit": data_policy_audit,
        "dataset_audit": dataset_audit,
        "main_subset_audit": main_subset_audit,
        "auxiliary_eval_plan": auxiliary_eval_plan,
        "split_meta": split_meta,
        "runtime_eval_sampling": {
            "val": val_runtime_eval_sampling,
            "test": test_runtime_eval_sampling,
        },
        "subset_sampling_mode": subset_sampling_mode,
        "subset_sampling_with_replacement": subset_sampling_with_replacement,
        "subset_class_balance_power": subset_class_balance_power,
        "subset_epoch_targets": train_sampling_meta["subset_targets"],
        "raw_subset_train_samples": train_sampling_meta["raw_subset_counts"],
        "effective_subset_samples_per_epoch": train_sampling_meta["effective_subset_counts"],
        "oversampled_subset_samples_per_epoch": train_sampling_meta["oversampled_subset_counts"],
        "class_weights": class_weights_dict,
        "virtual_train_class_counts": virtual_train_class_counts_dict,
        "val_subset_eval_counts": val_subset_eval_counts,
        "test_subset_eval_counts": test_subset_eval_counts,
    }
    print(json.dumps(audit_payload, ensure_ascii=False, indent=2))
    if args.audit_only:
        return 0

    print(json.dumps({
        "loading_whisper_size": whisper_size,
        "device": str(device),
    }, ensure_ascii=False), flush=True)
    shared_whisper_model = whisper.load_model(whisper_size, device=str(device))
    model = build_shared_model_from_config(shared_whisper_model, cfg).to(device)

    criterion, criterion_name, criterion_config = build_criterion(class_weights_tensor, class_weights, cfg)
    optimizer, base_lrs = build_optimizer(model, cfg)
    scheduler_patience = int(get_training_value(cfg, "live_encoder_scheduler_patience", get_training_value(cfg, "scheduler_patience", 5)))
    scheduler_factor = float(get_training_value(cfg, "scheduler_factor", 0.5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=scheduler_patience, factor=scheduler_factor)
    epochs = int(get_training_value(cfg, "epochs", 1))
    patience = int(get_training_value(cfg, "live_encoder_patience", get_training_value(cfg, "patience", 8)))
    warmup_epochs = max(0, int(get_training_value(cfg, "warmup_epochs", 0)))
    grad_accum_steps = int(get_training_value(cfg, "grad_accum_steps", max(1, math.ceil(base_batch_size / max(1, train_batch_size)))))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    augmenter = AudioAugmentation(
        time_mask_max=int(get_training_value(cfg, "live_encoder_time_mask_max", 24)),
        freq_mask_max=int(get_training_value(cfg, "live_encoder_freq_mask_max", 12)),
        noise_std=float(get_training_value(cfg, "live_encoder_noise_std", 0.01)),
    )

    canonical_experiment_stem = build_experiment_name(model.variant, training_mode, model.shared_config)
    experiment_stem = decorate_experiment_stem(canonical_experiment_stem, runtime_meta)
    experiment_name = build_run_name(experiment_stem, seed)
    ckpt_dir = PROJECT_ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_filename = f"{experiment_name}.pth"
    history_filename = f"{experiment_name}_history.npz"
    summary_filename = f"{experiment_name}_summary.json"
    curve_filename = f"{experiment_name}_curves.png"
    cm_filename = f"{experiment_name}_confusion_matrix.png"

    best_monitor_value = float("-inf")
    best_secondary_monitor_value = float("-inf")
    best_val_loss_at_best = float("inf")
    best_epoch = 0
    best_state = None
    best_selected_metrics = None
    wait = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_uar": [],
        "val_subset_mean_uar": [],
    }

    print(json.dumps({
        "experiment_name": experiment_name,
        "runtime_profile": runtime_meta,
        "optimizer_lrs": format_group_lrs(optimizer),
        "criterion_name": criterion_name,
        "criterion_config": criterion_config,
    }, ensure_ascii=False, indent=2))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_train_loader, epoch_sampling = build_epoch_train_loader(epoch)
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            apply_warmup(optimizer, base_lrs, epoch, warmup_epochs)
        elif warmup_epochs > 0 and epoch == warmup_epochs + 1:
            apply_warmup(optimizer, base_lrs, warmup_epochs, warmup_epochs)

        tr_loss, tr_acc = run_epoch(
            model,
            epoch_train_loader,
            criterion,
            optimizer,
            device=device,
            augmenter=augmenter,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
        )
        val_metrics = evaluate_epoch(model, val_loader, criterion, device=device)
        per_subset_val_metrics = evaluate_subset_loaders(model, val_loaders_by_subset, criterion, device=device)
        val_metrics["subset_mean_uar"] = compute_subset_mean_metric(per_subset_val_metrics, metric_key="uar", subset_order=main_subsets)

        monitor_value = float(val_metrics["subset_mean_uar"])
        secondary_monitor_value = float(val_metrics["uar"])
        if epoch > warmup_epochs:
            scheduler.step(monitor_value)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_uar"].append(val_metrics["uar"])
        history["val_subset_mean_uar"].append(val_metrics["subset_mean_uar"])

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
            f"Macro-F1: {val_metrics['macro_f1']:.4f} UAR: {val_metrics['uar']:.4f} "
            f"Subset-UAR: {val_metrics['subset_mean_uar']:.4f} | "
            f"Monitor: val_subset_mean_uar={monitor_value:.4f}, val_uar={secondary_monitor_value:.4f} | "
            f"Effective Train: {epoch_sampling['effective_train_samples']} | "
            f"Subsets: {format_subset_counts(epoch_sampling['effective_subset_counts'], main_subsets)} | "
            f"Over-sampled: {format_subset_counts(epoch_sampling['oversampled_subset_counts'], main_subsets)} | "
            f"LR: {format_group_lrs(optimizer)} | {elapsed:.1f}s"
        )

        is_better = (monitor_value > best_monitor_value) or (
            math.isclose(monitor_value, best_monitor_value, rel_tol=0.0, abs_tol=1e-6)
            and secondary_monitor_value > best_secondary_monitor_value
        ) or (
            math.isclose(monitor_value, best_monitor_value, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(secondary_monitor_value, best_secondary_monitor_value, rel_tol=0.0, abs_tol=1e-6)
            and val_metrics["loss"] < best_val_loss_at_best
        )
        if is_better:
            best_monitor_value = monitor_value
            best_secondary_monitor_value = secondary_monitor_value
            best_val_loss_at_best = float(val_metrics["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_selected_metrics = copy.deepcopy(val_metrics)
            wait = 0
            print("  -> 最佳模型已按 val_subset_mean_uar / val_uar / val_loss 保存")
        else:
            wait += 1
            if wait >= patience:
                print(f"\n早停: 连续 {patience} 轮在 val_subset_mean_uar 上无改善")
                break

    if best_state is None or best_selected_metrics is None:
        raise RuntimeError("训练结束后未保存任何最佳模型，请检查训练流程。")

    model.load_state_dict(best_state)
    checkpoint = create_shared_checkpoint(
        model,
        cfg,
        extra={
            "training_mode": training_mode,
            "data_meta": data_meta,
            "data_policy_audit": data_policy_audit,
            "runtime_profile": runtime_meta,
            "runtime_eval_sampling": {
                "val": val_runtime_eval_sampling,
                "test": test_runtime_eval_sampling,
            },
            "protocol_version": int(protocol_version),
            "seed": int(seed),
            "seed_sweep": list(seed_sweep),
            "included_subsets": list(main_subsets),
            "auxiliary_subsets": list(auxiliary_subsets),
            "dataset_audit": dataset_audit["subset_audit"],
            "dataset_normalization_warnings": dataset_audit["normalization_warnings"],
            "auxiliary_eval_plan": auxiliary_eval_plan,
            "split_meta": split_meta,
            "best_metric": "val_subset_mean_uar",
            "best_epoch": int(best_epoch),
            "criterion_name": criterion_name,
            "criterion_config": criterion_config,
            "class_weights": class_weights_dict,
            "virtual_train_class_counts": virtual_train_class_counts_dict,
            "subset_sampling_mode": subset_sampling_mode,
            "subset_sampling_with_replacement": bool(subset_sampling_with_replacement),
            "subset_class_balance_power": float(subset_class_balance_power),
            "subset_epoch_targets": train_sampling_meta["subset_targets"],
            "selected_val_acc": float(best_selected_metrics["acc"]),
            "selected_val_macro_f1": float(best_selected_metrics["macro_f1"]),
            "selected_val_uar": float(best_selected_metrics["uar"]),
            "selected_val_subset_mean_uar": float(best_selected_metrics["subset_mean_uar"]),
            "selected_val_loss": float(best_val_loss_at_best),
            "best_val_loss": float(np.min(history["val_loss"])),
            "best_val_acc": float(np.max(history["val_acc"])),
            "best_val_macro_f1": float(np.max(history["val_macro_f1"])),
            "best_val_uar": float(np.max(history["val_uar"])),
            "best_val_subset_mean_uar": float(np.max(history["val_subset_mean_uar"])),
            "subset_epoch_caps": subset_caps,
            "raw_subset_train_samples": train_sampling_meta["raw_subset_counts"],
            "effective_subset_samples_per_epoch": train_sampling_meta["effective_subset_counts"],
            "oversampled_subset_samples_per_epoch": train_sampling_meta["oversampled_subset_counts"],
            "raw_train_samples": int(len(train_set)),
            "effective_train_samples_per_epoch": int(train_sampling_meta["effective_train_samples_per_epoch"]),
            "augmentation_config": {
                "live_encoder": {
                    "time_mask_max": int(augmenter.time_mask_max),
                    "freq_mask_max": int(augmenter.freq_mask_max),
                    "noise_std": float(augmenter.noise_std),
                }
            },
            "grad_accum_steps": int(grad_accum_steps),
            "train_batch_size": int(train_batch_size),
            "eval_batch_size": int(eval_batch_size),
            "amp_enabled": bool(device.type == "cuda"),
            "optimizer_lrs": format_group_lrs(optimizer),
            "experiment_name": experiment_name,
            "experiment_stem": experiment_stem,
            "canonical_experiment_stem": canonical_experiment_stem,
        },
    )

    experiment_checkpoint_path = ckpt_dir / checkpoint_filename
    experiment_history_path = ckpt_dir / history_filename
    torch.save(checkpoint, experiment_checkpoint_path)
    np.savez(experiment_history_path, **history)
    save_curves(history, ckpt_dir / curve_filename, experiment_name)

    model.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_inputs = move_batch_to_device(batch, device, augmenter=None, is_training=False)
            logits = model(mel=batch_inputs["mel"], attention_mask=batch_inputs["attention_mask"])
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch_inputs["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels_list.extend(labels)

    report_text = classification_report(
        all_labels_list,
        all_preds,
        labels=list(range(len(EMOTION_LABELS))),
        target_names=EMOTION_LABELS,
        zero_division=0,
    )
    print(report_text)
    save_confusion_matrix(all_labels_list, all_preds, ckpt_dir / cm_filename, experiment_name)

    per_subset_val_metrics = evaluate_subset_loaders(model, val_loaders_by_subset, criterion, device=device)
    per_subset_test_metrics = evaluate_subset_loaders(model, test_loaders_by_subset, criterion, device=device)
    val_subset_mean_uar = compute_subset_mean_metric(per_subset_val_metrics, metric_key="uar", subset_order=main_subsets)
    test_subset_mean_uar = compute_subset_mean_metric(per_subset_test_metrics, metric_key="uar", subset_order=main_subsets)
    per_class_recall_values = recall_score(
        all_labels_list,
        all_preds,
        labels=list(range(len(EMOTION_LABELS))),
        average=None,
        zero_division=0,
    )
    per_class_recall = {label: float(score) for label, score in zip(EMOTION_LABELS, per_class_recall_values)}

    summary = {
        "protocol_version": int(protocol_version),
        "num_runs": 1,
        "experiment_name": experiment_name,
        "experiment_stem": experiment_stem,
        "canonical_experiment_stem": canonical_experiment_stem,
        "variant": model.variant,
        "training_mode": training_mode,
        "seed": int(seed),
        "seed_sweep": list(seed_sweep),
        "included_subsets": list(main_subsets),
        "auxiliary_subsets": list(auxiliary_subsets),
        "split_meta": split_meta,
        "dataset_audit": dataset_audit["subset_audit"],
        "dataset_normalization_warnings": dataset_audit["normalization_warnings"],
        "data_policy_audit": data_policy_audit,
        "auxiliary_eval_plan": auxiliary_eval_plan,
        "runtime_profile": runtime_meta,
        "runtime_eval_sampling": {
            "val": val_runtime_eval_sampling,
            "test": test_runtime_eval_sampling,
        },
        "norm": model.shared_config.get("norm"),
        "pooling": model.shared_config.get("pooling"),
        "freeze_strategy": model.shared_config.get("freeze_strategy"),
        "criterion_name": criterion_name,
        "criterion_config": criterion_config,
        "class_weights": class_weights_dict,
        "virtual_train_class_counts": virtual_train_class_counts_dict,
        "subset_sampling_mode": subset_sampling_mode,
        "subset_sampling_with_replacement": bool(subset_sampling_with_replacement),
        "subset_class_balance_power": float(subset_class_balance_power),
        "subset_epoch_targets": train_sampling_meta["subset_targets"],
        "epochs_ran": int(len(history["val_loss"])),
        "best_metric": "val_subset_mean_uar",
        "best_epoch": int(best_epoch),
        "selected_val_acc": float(best_selected_metrics["acc"]),
        "selected_val_macro_f1": float(best_selected_metrics["macro_f1"]),
        "selected_val_uar": float(best_selected_metrics["uar"]),
        "selected_val_subset_mean_uar": float(best_selected_metrics["subset_mean_uar"]),
        "selected_val_loss": float(best_val_loss_at_best),
        "best_val_loss": float(np.min(history["val_loss"])),
        "best_val_acc": float(np.max(history["val_acc"])),
        "best_val_macro_f1": float(np.max(history["val_macro_f1"])),
        "best_val_uar": float(np.max(history["val_uar"])),
        "best_val_subset_mean_uar": float(np.max(history["val_subset_mean_uar"])),
        "val_subset_mean_uar": float(val_subset_mean_uar),
        "test_acc": float(accuracy_score(all_labels_list, all_preds)),
        "macro_f1": float(f1_score(all_labels_list, all_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_labels_list, all_preds, average="weighted", zero_division=0)),
        "uar": float(recall_score(all_labels_list, all_preds, average="macro", zero_division=0)),
        "test_subset_mean_uar": float(test_subset_mean_uar),
        "per_subset_val_metrics": per_subset_val_metrics,
        "per_subset_test_metrics": per_subset_test_metrics,
        "per_class_recall": per_class_recall,
        "aggregate_mean_std": None,
        "subset_epoch_caps": subset_caps,
        "raw_subset_train_samples": train_sampling_meta["raw_subset_counts"],
        "effective_subset_samples_per_epoch": train_sampling_meta["effective_subset_counts"],
        "oversampled_subset_samples_per_epoch": train_sampling_meta["oversampled_subset_counts"],
        "raw_train_samples": int(len(train_set)),
        "effective_train_samples_per_epoch": int(train_sampling_meta["effective_train_samples_per_epoch"]),
        "checkpoint_filename": checkpoint_filename,
        "history_filename": history_filename,
        "summary_filename": summary_filename,
        "confusion_matrix_filename": cm_filename,
        "curve_filename": curve_filename,
    }

    summary_path = ckpt_dir / summary_filename
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    aggregate_summary = aggregate_experiment_summaries(experiment_stem, seed_sweep, ckpt_dir, protocol_version)
    if aggregate_summary is not None:
        print(json.dumps(aggregate_summary, ensure_ascii=False, indent=2))

    print(f"实验 checkpoint 已保存至: {experiment_checkpoint_path}")
    print(f"实验 history 已保存至: {experiment_history_path}")
    print(f"实验摘要已保存至: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
