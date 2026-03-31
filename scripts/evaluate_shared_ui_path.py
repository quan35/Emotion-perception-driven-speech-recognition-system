#!/usr/bin/env python3
"""按 UI 推理路径重评估共享模型 checkpoint。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.audio_utils import LABEL2ID, EMOTION_LABELS, load_audio, load_config
from utils.data_policy import filter_supported_samples
from utils.split_utils import infer_subset_from_path, speaker_group_split
from preprocessing.audio_preprocess import AudioPreprocessor
from preprocessing.whisper_feature_cache import (
    WHISPER_SAMPLE_RATE,
    build_sample_list,
    build_whisper_ser_batch_from_raw_audio,
)
from models.whisper_emotion import (
    WhisperEmotionHead,
    build_shared_model_from_config,
    is_legacy_shared_checkpoint,
    load_shared_checkpoint_state,
)
import whisper


ESD_RAW_LABELS = {
    "angry": "Angry",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 UI 路径重评估共享模型 checkpoint。")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="共享模型 checkpoint 路径；默认使用 config.yaml 中的 paths.best_shared_model。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="配置文件路径，默认 configs/config.yaml。",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="评估划分，默认 test。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批大小，默认读取 training.live_encoder_eval_batch_size。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="仅评估前 N 个样本，用于快速 smoke test。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="推理设备，默认 auto。",
    )
    parser.add_argument(
        "--disable-denoise",
        action="store_true",
        help="关闭 UI 对齐路径中的降噪步骤。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="可选：将评估结果写入 JSON 文件。",
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




def resolve_seeded_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = sorted(path.parent.glob(f"{path.stem}_seed*.pth"))
    if not candidates:
        return path

    preferred = [candidate for candidate in candidates if candidate.stem.endswith("_seed42")]
    return preferred[0] if preferred else candidates[0]


def resolve_checkpoint_path(cfg: dict, checkpoint_arg: Optional[Path], config_path: Path) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = checkpoint_arg
    else:
        checkpoint_path = Path(cfg["paths"]["best_shared_model"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config_path.parent.parent / checkpoint_path).resolve()
    return resolve_seeded_checkpoint_path(checkpoint_path)


def load_checkpoint_summary(checkpoint_path: Path) -> Optional[dict]:
    summary_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_summary.json")
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_shared_model(checkpoint_path: Path, cfg: dict, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    whisper_size = ckpt.get("whisper_size") or cfg.get("model", {}).get("whisper_size")

    if is_legacy_shared_checkpoint(ckpt):
        whisper_model = whisper.load_model(whisper_size, device=str(device))
        model = WhisperEmotionHead(
            whisper_model,
            num_classes=int(ckpt.get("num_classes", cfg["emotion"]["num_classes"])),
            variant="legacy_mlp",
            freeze_strategy="freeze_all",
            pooling="mean",
            whisper_size=whisper_size,
        ).to(device)
        model.classifier.load_state_dict(ckpt["classifier_state"])
        model.eval()
        return model, ckpt

    whisper_model = whisper.load_model(whisper_size, device=str(device))
    model = build_shared_model_from_config(
        whisper_model,
        cfg,
        whisper_size=whisper_size,
        **ckpt.get("shared_model_config", {}),
    ).to(device)
    load_shared_checkpoint_state(model, ckpt)
    model.eval()
    return model, ckpt


def resolve_raw_audio_path(processed_path: str, raw_root: Path) -> Optional[Path]:
    processed = Path(processed_path)
    subset = infer_subset_from_path(processed_path)
    label = processed.parent.name
    stem = processed.stem

    if subset == "ravdess":
        speaker = stem.split("-")[-1]
        return raw_root / "ravdess" / f"Actor_{speaker}" / processed.name

    if subset == "casia":
        speaker, utterance = stem.split("_", 1)
        return raw_root / "casia" / speaker / label / f"{utterance}.wav"

    if subset == "emodb":
        return raw_root / "emodb" / f"{stem.removeprefix('emodb_')}.wav"

    if subset == "esd":
        tokens = stem.removeprefix("esd_").split("_")
        speaker = tokens[0]
        utterance = tokens[-1]
        raw_label = ESD_RAW_LABELS.get(label)
        if raw_label is None:
            return None
        return raw_root / "esd" / speaker / raw_label / f"{speaker}_{utterance}.wav"

    if subset == "iemocap":
        raw_stem = stem.removeprefix("iemocap_")
        conversation = raw_stem.rsplit("_", 1)[0]
        match = re.match(r"Ses(\d{2})", raw_stem)
        if match is None:
            return None
        session_id = int(match.group(1))
        return raw_root / "iemocap" / f"Session{session_id}" / "sentences" / "wav" / conversation / f"{raw_stem}.wav"

    return None


def build_eval_samples(cfg: dict, summary: Optional[dict], split_name: str, limit: Optional[int]) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    training_cfg = cfg.get("training", {})
    included_subsets = list((summary or {}).get("included_subsets") or training_cfg.get("main_subsets", []))
    processed_dir = cfg["paths"]["processed_data"]
    raw_samples = build_sample_list(processed_dir, LABEL2ID, subsets=included_subsets)
    samples, _ = filter_supported_samples(raw_samples, cfg)
    paths = [path for path, _ in samples]

    _, val_idx, test_idx, split_meta = speaker_group_split(
        paths,
        train_ratio=float(training_cfg["train_ratio"]),
        val_ratio=float(training_cfg["val_ratio"]),
        test_ratio=float(training_cfg["test_ratio"]),
        seed=int((summary or {}).get("seed", training_cfg.get("seed", 42))),
        include_subsets=included_subsets,
    )

    selected_indices = val_idx if split_name == "val" else test_idx
    selected = [(samples[idx][0], int(samples[idx][1])) for idx in selected_indices]
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected, split_meta


def batched(items: Sequence[Tuple[str, int]], batch_size: int) -> Iterable[Sequence[Tuple[str, int]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "acc": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "uar": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_checkpoint(
    model,
    samples: Sequence[Tuple[str, int]],
    cfg: dict,
    device: torch.device,
    apply_denoise: bool,
    batch_size: int,
) -> Dict[str, Any]:
    preprocessor = AudioPreprocessor(cfg)
    raw_root = Path(cfg["paths"]["raw_data"])
    if not raw_root.is_absolute():
        raw_root = (PROJECT_ROOT / raw_root).resolve()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_subsets: List[str] = []
    raw_resolved = 0
    processed_fallback = 0

    for batch in batched(list(samples), batch_size):
        audios: List[np.ndarray] = []
        srs: List[int] = []
        labels: List[int] = []
        subsets: List[str] = []
        denoise_flags: List[bool] = []

        for processed_path, label in batch:
            raw_path = resolve_raw_audio_path(processed_path, raw_root)
            use_raw = raw_path is not None and raw_path.is_file()
            source_path = raw_path if use_raw else Path(processed_path)
            audio, sr = load_audio(str(source_path), sr=WHISPER_SAMPLE_RATE)
            audios.append(audio.astype(np.float32, copy=False))
            srs.append(int(sr))
            labels.append(int(label))
            subsets.append(infer_subset_from_path(processed_path))
            denoise_flags.append(bool(apply_denoise and use_raw))
            if use_raw:
                raw_resolved += 1
            else:
                processed_fallback += 1

        mel, attention_mask = build_whisper_ser_batch_from_raw_audio(
            audios=audios,
            srs=srs,
            device=device,
            preprocessor=preprocessor,
            apply_denoise=False,
            apply_denoise_flags=denoise_flags,
        )

        with torch.no_grad():
            logits = model(mel=mel, attention_mask=attention_mask)
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()

        all_labels.extend(labels)
        all_preds.extend(int(pred) for pred in preds)
        all_subsets.extend(subsets)

    overall = compute_metrics(all_labels, all_preds)
    per_subset_metrics: Dict[str, Dict[str, float]] = {}
    for subset in sorted(set(all_subsets)):
        indices = [idx for idx, value in enumerate(all_subsets) if value == subset]
        subset_labels = [all_labels[idx] for idx in indices]
        subset_preds = [all_preds[idx] for idx in indices]
        per_subset_metrics[subset] = {
            "samples": int(len(indices)),
            **compute_metrics(subset_labels, subset_preds),
        }

    subset_mean_uar = float(np.mean([metrics["uar"] for metrics in per_subset_metrics.values()])) if per_subset_metrics else float("nan")
    class_recall = recall_score(
        all_labels,
        all_preds,
        labels=list(range(len(EMOTION_LABELS))),
        average=None,
        zero_division=0,
    )

    return {
        "num_samples": int(len(all_labels)),
        "raw_source_resolved": int(raw_resolved),
        "processed_fallback_count": int(processed_fallback),
        "acc": float(overall["acc"]),
        "macro_f1": float(overall["macro_f1"]),
        "uar": float(overall["uar"]),
        "subset_mean_uar": float(subset_mean_uar),
        "per_subset_metrics": per_subset_metrics,
        "per_class_recall": {
            label: float(class_recall[idx])
            for idx, label in enumerate(EMOTION_LABELS)
        },
    }


def build_summary_reference(summary: Optional[dict], split_name: str) -> Optional[Dict[str, float]]:
    if summary is None:
        return None
    if split_name == "test":
        return {
            "test_acc": float(summary.get("test_acc", float("nan"))),
            "macro_f1": float(summary.get("macro_f1", float("nan"))),
            "uar": float(summary.get("uar", float("nan"))),
            "test_subset_mean_uar": float(summary.get("test_subset_mean_uar", float("nan"))),
        }
    return {
        "selected_val_acc": float(summary.get("selected_val_acc", float("nan"))),
        "selected_val_macro_f1": float(summary.get("selected_val_macro_f1", float("nan"))),
        "selected_val_uar": float(summary.get("selected_val_uar", float("nan"))),
        "selected_val_subset_mean_uar": float(summary.get("selected_val_subset_mean_uar", float("nan"))),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(str(args.config))
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.config)
    if not checkpoint_path.is_file():
        raise SystemExit(f"未找到 checkpoint: {checkpoint_path}")

    summary = load_checkpoint_summary(checkpoint_path)
    model, ckpt = load_shared_model(checkpoint_path, cfg, device)

    batch_size = int(args.batch_size or cfg.get("training", {}).get("live_encoder_eval_batch_size", 16))
    samples, split_meta = build_eval_samples(cfg, summary, args.split, args.limit)
    results = evaluate_checkpoint(
        model=model,
        samples=samples,
        cfg=cfg,
        device=device,
        apply_denoise=not args.disable_denoise,
        batch_size=batch_size,
    )

    if args.split == "test":
        results["test_acc"] = results["acc"]
        results["test_subset_mean_uar"] = results["subset_mean_uar"]
    else:
        results["selected_val_acc"] = results["acc"]
        results["selected_val_macro_f1"] = results["macro_f1"]
        results["selected_val_uar"] = results["uar"]
        results["selected_val_subset_mean_uar"] = results["subset_mean_uar"]

    reference = build_summary_reference(summary, args.split)
    deltas = None
    if reference is not None:
        deltas = {}
        for key, ref_value in reference.items():
            if np.isnan(ref_value):
                continue
            result_key = key
            if key == "test_subset_mean_uar":
                result_key = "test_subset_mean_uar"
            if key == "selected_val_subset_mean_uar":
                result_key = "selected_val_subset_mean_uar"
            current_value = float(results.get(result_key, float("nan")))
            if np.isnan(current_value):
                continue
            deltas[key] = float(current_value - ref_value)

    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "summary_path": None if summary is None else str(checkpoint_path.with_name(f"{checkpoint_path.stem}_summary.json").resolve()),
        "split": args.split,
        "device": str(device),
        "batch_size": int(batch_size),
        "denoise_enabled": bool(not args.disable_denoise),
        "split_meta": split_meta,
        "model_variant": ckpt.get("model_variant", "legacy_mlp" if is_legacy_shared_checkpoint(ckpt) else "transformer_head"),
        **results,
        "summary_reference": reference,
        "summary_delta": deltas,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = (PROJECT_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写入: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
