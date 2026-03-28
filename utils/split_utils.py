import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

import numpy as np


SUPPORTED_SUBSETS = ("ravdess", "casia", "tess", "esd", "emodb", "iemocap")
TESS_SPEAKER_ALIASES = {
    "OA": "OAF",
}


def infer_subset_from_path(path: str) -> str:
    parts = [part.lower() for part in Path(path).parts]
    for part in parts:
        if part in SUPPORTED_SUBSETS:
            return part
    raise ValueError(f"无法从路径识别数据子集: {path}")


def _strip_prefix(stem: str, prefix: str) -> str:
    prefix_token = f"{prefix.lower()}_"
    if stem.lower().startswith(prefix_token):
        return stem[len(prefix_token):]
    return stem


def _normalize_subset_filter(subsets: Iterable[str] | None) -> Tuple[str, ...] | None:
    if subsets is None:
        return None

    ordered: List[str] = []
    seen = set()
    for subset in subsets:
        normalized = str(subset).strip().lower()
        if normalized not in SUPPORTED_SUBSETS:
            raise ValueError(f"未支持的数据子集: {subset}")
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)

    if not ordered:
        raise ValueError("subsets 不能为空")
    return tuple(ordered)


def _infer_raw_speaker_id_from_path(path: str, subset: str | None = None) -> str:
    subset = subset or infer_subset_from_path(path)
    stem = Path(path).stem

    if subset == "ravdess":
        parts = stem.split("-")
        if len(parts) >= 7:
            return parts[6]
        raise ValueError(f"RAVDESS 文件名无法解析说话人: {path}")

    if subset == "casia":
        parts = stem.split("_", 1)
        if len(parts) >= 2 and parts[0]:
            return parts[0]
        raise ValueError(f"CASIA 文件名无法解析说话人: {path}")

    if subset == "tess":
        stem = _strip_prefix(stem, "tess")
        parts = stem.split("_", 1)
        if len(parts) >= 2 and parts[0]:
            return parts[0]
        raise ValueError(f"TESS 文件名无法解析说话人: {path}")

    if subset == "esd":
        stem = _strip_prefix(stem, "esd")
        parts = stem.split("_", 1)
        if len(parts) >= 2 and parts[0]:
            return parts[0]
        raise ValueError(f"ESD 文件名无法解析说话人: {path}")

    if subset == "emodb":
        stem = _strip_prefix(stem, "emodb")
        match = re.match(r"([0-9]{2})", stem)
        if match:
            return match.group(1)
        if len(stem) >= 2:
            return stem[:2]
        raise ValueError(f"EMODB 文件名无法解析说话人: {path}")

    if subset == "iemocap":
        stem = _strip_prefix(stem, "iemocap")
        parts = stem.split("_", 1)
        if len(parts) >= 2 and parts[0]:
            return parts[0]
        raise ValueError(f"IEMOCAP 文件名无法解析说话人: {path}")

    raise ValueError(f"未支持的数据子集: {subset}")


def inspect_sample_path(path: str) -> Dict[str, Any]:
    subset = infer_subset_from_path(path)
    raw_speaker_id = _infer_raw_speaker_id_from_path(path, subset=subset)
    speaker_id = raw_speaker_id
    normalization_warning = None

    if subset == "tess":
        raw_token = raw_speaker_id.upper()
        speaker_id = TESS_SPEAKER_ALIASES.get(raw_token, raw_token)
        if speaker_id != raw_token:
            normalization_warning = (
                f"TESS speaker alias normalized: {raw_token} -> {speaker_id} ({path})"
            )

    return {
        "path": path,
        "subset": subset,
        "raw_speaker_id": raw_speaker_id,
        "speaker_id": speaker_id,
        "normalization_warning": normalization_warning,
    }


def infer_speaker_id_from_path(path: str, subset: str | None = None) -> str:
    metadata = inspect_sample_path(path)
    if subset is not None and metadata["subset"] != str(subset).strip().lower():
        raise ValueError(
            f"路径子集与显式 subset 不一致: path_subset={metadata['subset']} subset={subset}"
        )
    return str(metadata["speaker_id"])


def build_group_ids_from_paths(paths: Sequence[str]) -> List[str]:
    group_ids: List[str] = []
    for path in paths:
        metadata = inspect_sample_path(path)
        group_ids.append(f"{metadata['subset']}:{metadata['speaker_id']}")
    return group_ids


def _subset_seed(seed: int, subset: str) -> int:
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(subset))
    return int(seed) * 1009 + offset


def _split_group_counts(
    num_groups: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    needs_val = float(val_ratio) > 0.0
    needs_test = float(test_ratio) > 0.0
    required_splits = 1 + int(needs_val) + int(needs_test)

    if num_groups < required_splits:
        raise ValueError(
            f"唯一说话人组数量不足以完成当前划分: groups={num_groups}, "
            f"required_splits={required_splits}"
        )

    if not needs_val and not needs_test:
        return num_groups, 0, 0

    if needs_val and needs_test:
        if num_groups == 3:
            return 1, 1, 1
        if num_groups == 4:
            return 2, 1, 1
        if num_groups >= 5:
            val_count = max(1, int(np.floor(num_groups * float(val_ratio))))
            test_count = max(1, int(np.floor(num_groups * float(test_ratio))))
            train_count = num_groups - val_count - test_count
            if train_count < 1:
                raise ValueError(
                    f"train 说话人组数量不足: groups={num_groups}, "
                    f"allocated train/val/test={train_count}/{val_count}/{test_count}"
                )
            return train_count, val_count, test_count

    if needs_val:
        val_count = max(1, int(np.floor(num_groups * float(val_ratio))))
        train_count = num_groups - val_count
        if train_count < 1:
            raise ValueError(
                f"train 说话人组数量不足: groups={num_groups}, "
                f"allocated train/val={train_count}/{val_count}"
            )
        return train_count, val_count, 0

    test_count = max(1, int(np.floor(num_groups * float(test_ratio))))
    train_count = num_groups - test_count
    if train_count < 1:
        raise ValueError(
            f"train 说话人组数量不足: groups={num_groups}, "
            f"allocated train/test={train_count}/{test_count}"
        )
    return train_count, 0, test_count


def _build_subset_group_map(
    paths: Sequence[str],
) -> Tuple[
    DefaultDict[str, Dict[str, List[int]]],
    np.ndarray,
    np.ndarray,
    Dict[str, List[str]],
    List[str],
]:
    subset_groups: DefaultDict[str, Dict[str, List[int]]] = defaultdict(dict)
    subset_names: List[str] = []
    group_ids: List[str] = []
    subset_speakers: DefaultDict[str, List[str]] = defaultdict(list)
    normalization_warnings: List[str] = []

    for idx, path in enumerate(paths):
        metadata = inspect_sample_path(path)
        subset = str(metadata["subset"])
        speaker_id = str(metadata["speaker_id"])
        group_id = f"{subset}:{speaker_id}"
        subset_names.append(subset)
        group_ids.append(group_id)
        subset_groups[subset].setdefault(group_id, []).append(idx)
        subset_speakers[subset].append(speaker_id)
        warning = metadata.get("normalization_warning")
        if warning and warning not in normalization_warnings:
            normalization_warnings.append(str(warning))

    return (
        subset_groups,
        np.asarray(subset_names, dtype=object),
        np.asarray(group_ids, dtype=object),
        {
            subset: sorted(set(speakers))
            for subset, speakers in subset_speakers.items()
        },
        normalization_warnings,
    )


def _count_values(values: np.ndarray) -> Dict[str, int]:
    if values.size == 0:
        return {}
    unique_values, counts = np.unique(values, return_counts=True)
    return {
        str(key): int(count)
        for key, count in sorted(
            zip(unique_values.tolist(), counts.tolist()), key=lambda item: item[0]
        )
    }


def audit_subset_groups(
    paths: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    subsets: Sequence[str] | None = None,
) -> Dict[str, Any]:
    subset_filter = _normalize_subset_filter(subsets)
    subset_groups, _, _, subset_speakers, normalization_warnings = _build_subset_group_map(paths)
    requested_subsets = subset_filter or tuple(
        subset for subset in SUPPORTED_SUBSETS if subset in subset_groups
    )
    required_splits = 1 + int(float(val_ratio) > 0.0) + int(float(test_ratio) > 0.0)

    subset_audit: Dict[str, Dict[str, Any]] = {}
    for subset in requested_subsets:
        group_map = subset_groups.get(subset, {})
        ordered_group_ids = sorted(group_map.keys())
        subset_audit[subset] = {
            "num_groups": int(len(ordered_group_ids)),
            "num_samples": int(sum(len(group_map[group_id]) for group_id in ordered_group_ids)),
            "speaker_ids": list(subset_speakers.get(subset, [])),
            "can_three_way_split": bool(len(ordered_group_ids) >= required_splits),
            "required_splits": int(required_splits),
        }

    return {
        "required_splits": int(required_splits),
        "requested_subsets": list(requested_subsets),
        "subset_audit": subset_audit,
        "normalization_warnings": normalization_warnings,
    }


def validate_subset_groups(
    paths: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    subsets: Sequence[str],
) -> Dict[str, Any]:
    audit = audit_subset_groups(
        paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        subsets=subsets,
    )
    invalid = {
        subset: data
        for subset, data in audit["subset_audit"].items()
        if int(data["num_groups"]) < int(audit["required_splits"])
    }
    if invalid:
        details = ", ".join(
            f"{subset}: groups={data['num_groups']} required={data['required_splits']}"
            for subset, data in invalid.items()
        )
        raise ValueError(
            "以下数据子集无法满足当前 speaker-group split 要求，请改为辅助评估或修复数据："
            f"{details}"
        )
    return audit


def build_group_holdout_folds(paths: Sequence[str], subset: str) -> Dict[str, Any]:
    normalized_subset = str(subset).strip().lower()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"未支持的数据子集: {subset}")

    subset_groups, _, _, subset_speakers, normalization_warnings = _build_subset_group_map(paths)
    group_map = subset_groups.get(normalized_subset)
    if not group_map:
        raise ValueError(f"数据中不存在子集: {normalized_subset}")

    ordered_group_ids = sorted(group_map.keys())
    if len(ordered_group_ids) < 2:
        raise ValueError(
            f"{normalized_subset} 唯一说话人组数量不足以构建 holdout folds: groups={len(ordered_group_ids)}"
        )

    folds: List[Dict[str, Any]] = []
    for fold_idx, test_group_id in enumerate(ordered_group_ids, start=1):
        train_group_ids = [group_id for group_id in ordered_group_ids if group_id != test_group_id]
        train_indices = sorted(
            idx for group_id in train_group_ids for idx in group_map[group_id]
        )
        test_indices = sorted(group_map[test_group_id])
        folds.append(
            {
                "fold_index": int(fold_idx),
                "train_group_ids": [group_id.split(":", 1)[1] for group_id in train_group_ids],
                "test_group_ids": [test_group_id.split(":", 1)[1]],
                "train_samples": int(len(train_indices)),
                "test_samples": int(len(test_indices)),
                "train_indices": train_indices,
                "test_indices": test_indices,
            }
        )

    return {
        "subset": normalized_subset,
        "speaker_ids": list(subset_speakers.get(normalized_subset, [])),
        "num_groups": int(len(ordered_group_ids)),
        "num_samples": int(sum(len(group_map[group_id]) for group_id in ordered_group_ids)),
        "folds": folds,
        "normalization_warnings": normalization_warnings,
    }


def speaker_group_split(
    paths: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    include_subsets: Sequence[str] | None = None,
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    if not paths:
        raise ValueError("paths 为空，无法执行 speaker-group split")

    ratio_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"train/val/test 比例之和必须为 1.0，当前为 {ratio_sum:.6f}"
        )

    subset_filter = _normalize_subset_filter(include_subsets)
    subset_groups, subset_names, group_ids, subset_speakers, normalization_warnings = _build_subset_group_map(paths)
    included_subsets = subset_filter or tuple(
        subset for subset in SUPPORTED_SUBSETS if subset in subset_groups
    )
    if not included_subsets:
        raise ValueError("当前配置未选中任何可切分的数据子集")

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    subset_group_counts: Dict[str, int] = {}
    subset_sample_counts: Dict[str, int] = {}
    train_subset_group_counts: Dict[str, int] = {}
    val_subset_group_counts: Dict[str, int] = {}
    test_subset_group_counts: Dict[str, int] = {}
    train_subset_sample_counts: Dict[str, int] = {}
    val_subset_sample_counts: Dict[str, int] = {}
    test_subset_sample_counts: Dict[str, int] = {}

    for subset in included_subsets:
        group_map = subset_groups.get(subset)
        if not group_map:
            continue

        ordered_group_ids = sorted(group_map.keys())
        num_groups = len(ordered_group_ids)
        subset_group_counts[subset] = int(num_groups)
        subset_sample_counts[subset] = int(
            sum(len(group_map[group_id]) for group_id in ordered_group_ids)
        )

        train_group_count, val_group_count, test_group_count = _split_group_counts(
            num_groups,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        rng = np.random.default_rng(_subset_seed(seed, subset))
        shuffled_group_ids = rng.permutation(
            np.asarray(ordered_group_ids, dtype=object)
        ).tolist()

        train_group_ids = shuffled_group_ids[:train_group_count]
        val_group_ids = shuffled_group_ids[
            train_group_count:train_group_count + val_group_count
        ]
        test_group_ids = shuffled_group_ids[train_group_count + val_group_count:]

        train_subset_group_counts[subset] = int(len(train_group_ids))
        val_subset_group_counts[subset] = int(len(val_group_ids))
        test_subset_group_counts[subset] = int(len(test_group_ids))

        train_subset_indices = sorted(
            idx for group_id in train_group_ids for idx in group_map[group_id]
        )
        val_subset_indices = sorted(
            idx for group_id in val_group_ids for idx in group_map[group_id]
        )
        test_subset_indices = sorted(
            idx for group_id in test_group_ids for idx in group_map[group_id]
        )

        train_subset_sample_counts[subset] = int(len(train_subset_indices))
        val_subset_sample_counts[subset] = int(len(val_subset_indices))
        test_subset_sample_counts[subset] = int(len(test_subset_indices))

        train_indices.extend(train_subset_indices)
        val_indices.extend(val_subset_indices)
        test_indices.extend(test_subset_indices)

    train_idx = np.asarray(sorted(train_indices), dtype=np.int64)
    val_idx = np.asarray(sorted(val_indices), dtype=np.int64)
    test_idx = np.asarray(sorted(test_indices), dtype=np.int64)
    if (len(train_idx) + len(val_idx) + len(test_idx)) > 0:
        filtered_sample_indices = np.concatenate([train_idx, val_idx, test_idx]).astype(np.int64, copy=False)
        unique_groups = np.unique(group_ids[filtered_sample_indices])
    else:
        filtered_sample_indices = np.asarray([], dtype=np.int64)
        unique_groups = np.asarray([], dtype=object)
    excluded_subsets = [
        subset for subset in SUPPORTED_SUBSETS if subset not in included_subsets
    ]

    split_meta: Dict[str, Any] = {
        "included_subsets": list(included_subsets),
        "excluded_subsets": excluded_subsets,
        "total_samples": int(len(filtered_sample_indices)),
        "total_groups": int(len(unique_groups)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "train_groups": int(len(np.unique(group_ids[train_idx]))),
        "val_groups": int(len(np.unique(group_ids[val_idx]))),
        "test_groups": int(len(np.unique(group_ids[test_idx]))),
        "subset_group_counts": subset_group_counts,
        "subset_sample_counts": subset_sample_counts,
        "train_subset_group_counts": train_subset_group_counts,
        "val_subset_group_counts": val_subset_group_counts,
        "test_subset_group_counts": test_subset_group_counts,
        "train_subset_sample_counts": train_subset_sample_counts,
        "val_subset_sample_counts": val_subset_sample_counts,
        "test_subset_sample_counts": test_subset_sample_counts,
        "train_subset_distribution": _count_values(subset_names[train_idx]),
        "val_subset_distribution": _count_values(subset_names[val_idx]),
        "test_subset_distribution": _count_values(subset_names[test_idx]),
        "subset_speakers": {
            subset: list(subset_speakers.get(subset, []))
            for subset in included_subsets
            if subset in subset_speakers
        },
        "normalization_warnings": normalization_warnings,
    }
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), split_meta
