from __future__ import annotations

import glob
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.split_utils import infer_subset_from_path


SUPPORTED_POLICY_SUBSETS = ("ravdess", "casia", "tess", "esd", "emodb", "iemocap")


def _normalize_source_label(value: object) -> str:
    return str(value).strip().lower()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _lookup_mapping(mapping: Dict[str, str], raw_label: str) -> Optional[str]:
    candidates = (
        str(raw_label).strip(),
        str(raw_label).strip().lower(),
        str(raw_label).strip().upper(),
        str(raw_label).strip().capitalize(),
    )
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def get_data_policy(cfg: dict) -> Dict[str, Any]:
    return dict(cfg.get("data_policy", {}))


def get_dataset_drop_source_labels(cfg: dict, subset: str) -> Tuple[str, ...]:
    subset_policy = get_data_policy(cfg).get("datasets", {}).get(str(subset).strip().lower(), {})
    labels = subset_policy.get("drop_source_labels", [])
    return tuple(_normalize_source_label(label) for label in labels)


def raw_label_allowed(cfg: dict, subset: str, raw_label: Optional[str]) -> bool:
    if raw_label is None:
        return True
    normalized_subset = str(subset).strip().lower()
    return _normalize_source_label(raw_label) not in set(get_dataset_drop_source_labels(cfg, normalized_subset))


def resolve_dataset_target_label(cfg: dict, subset: str, raw_label: str) -> Optional[str]:
    normalized_subset = str(subset).strip().lower()
    if not raw_label_allowed(cfg, normalized_subset, raw_label):
        return None
    mapping = cfg.get("datasets", {}).get(normalized_subset, {}).get("emotions", {})
    return _lookup_mapping(mapping, raw_label)


def _resolve_raw_root(cfg: Optional[dict]) -> Optional[Path]:
    if cfg is None:
        return None
    raw_root = cfg.get("paths", {}).get("raw_data")
    if not raw_root:
        return None
    raw_root = Path(raw_root)
    if not raw_root.is_absolute():
        raw_root = (_project_root() / raw_root).resolve()
    return raw_root


@lru_cache(maxsize=8)
def _iemocap_label_map(raw_root: str) -> Dict[str, str]:
    root = Path(raw_root)
    label_map: Dict[str, str] = {}
    if not root.is_dir():
        return label_map

    for session in range(1, 6):
        for session_dir in (root / f"Session{session}", root / f"session{session}"):
            if not session_dir.is_dir():
                continue
            emo_eval_dir = session_dir / "dialog" / "EmoEvaluation"
            if not emo_eval_dir.is_dir():
                continue
            for txt_path in sorted(glob.glob(str(emo_eval_dir / "*.txt"))):
                with open(txt_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.startswith("["):
                            continue
                        parts = line.strip().split("	")
                        if len(parts) >= 3:
                            label_map[parts[1].strip()] = parts[2].strip()
            break
    return label_map


def _parse_emodb_source_label(stem: str) -> Optional[str]:
    codes = "WLEAFTN"
    matches = [ch for ch in stem.upper() if ch in codes]
    if matches:
        return matches[-1]
    return None


def infer_source_label_from_processed_path(path: str, cfg: Optional[dict] = None) -> Optional[str]:
    subset = infer_subset_from_path(path)
    stem = Path(path).stem

    if subset == "ravdess":
        parts = stem.split("-")
        return parts[2] if len(parts) >= 3 else None

    if subset == "casia":
        return Path(path).parent.name

    if subset == "tess":
        token = stem.removeprefix("tess_").rsplit("_", 1)[-1]
        return token or None

    if subset == "esd":
        return Path(path).parent.name

    if subset == "emodb":
        return _parse_emodb_source_label(stem.removeprefix("emodb_"))

    if subset == "iemocap":
        raw_root = _resolve_raw_root(cfg)
        if raw_root is None:
            return None
        utterance_id = stem.removeprefix("iemocap_")
        return _iemocap_label_map(str(raw_root / "iemocap")).get(utterance_id)

    return None


def build_data_policy_audit(samples: Sequence[Tuple[str, int]], cfg: dict) -> Dict[str, Any]:
    per_subset: Dict[str, Dict[str, Any]] = {}
    total_kept = 0
    total_dropped = 0

    for path, _ in samples:
        subset = infer_subset_from_path(path)
        raw_label = infer_source_label_from_processed_path(path, cfg)
        keep = raw_label_allowed(cfg, subset, raw_label)

        subset_entry = per_subset.setdefault(
            subset,
            {
                "input_samples": 0,
                "kept_samples": 0,
                "dropped_samples": 0,
                "input_source_labels": defaultdict(int),
                "kept_source_labels": defaultdict(int),
                "dropped_source_labels": defaultdict(int),
                "unknown_source_labels": 0,
            },
        )
        subset_entry["input_samples"] += 1
        normalized_raw = raw_label if raw_label is not None else "<unknown>"
        subset_entry["input_source_labels"][normalized_raw] += 1
        if raw_label is None:
            subset_entry["unknown_source_labels"] += 1

        if keep:
            subset_entry["kept_samples"] += 1
            subset_entry["kept_source_labels"][normalized_raw] += 1
            total_kept += 1
        else:
            subset_entry["dropped_samples"] += 1
            subset_entry["dropped_source_labels"][normalized_raw] += 1
            total_dropped += 1

    normalized_per_subset: Dict[str, Dict[str, Any]] = {}
    for subset, values in per_subset.items():
        normalized_per_subset[subset] = {
            "input_samples": int(values["input_samples"]),
            "kept_samples": int(values["kept_samples"]),
            "dropped_samples": int(values["dropped_samples"]),
            "unknown_source_labels": int(values["unknown_source_labels"]),
            "input_source_labels": dict(sorted(values["input_source_labels"].items())),
            "kept_source_labels": dict(sorted(values["kept_source_labels"].items())),
            "dropped_source_labels": dict(sorted(values["dropped_source_labels"].items())),
            "drop_source_labels_config": list(get_dataset_drop_source_labels(cfg, subset)),
        }

    return {
        "enabled": bool(get_data_policy(cfg).get("enabled", True)),
        "profile": str(get_data_policy(cfg).get("profile", "staged_clean")),
        "total_input_samples": int(len(samples)),
        "total_kept_samples": int(total_kept),
        "total_dropped_samples": int(total_dropped),
        "per_subset": normalized_per_subset,
    }


def filter_supported_samples(
    samples: Sequence[Tuple[str, int]],
    cfg: dict,
) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    audit = build_data_policy_audit(samples, cfg)
    if not bool(get_data_policy(cfg).get("enabled", True)):
        return list(samples), audit

    filtered = [
        (path, label)
        for path, label in samples
        if raw_label_allowed(cfg, infer_subset_from_path(path), infer_source_label_from_processed_path(path, cfg))
    ]
    return filtered, audit
