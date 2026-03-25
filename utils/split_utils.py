import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


SUPPORTED_SUBSETS = ("ravdess", "casia", "tess", "esd", "emodb", "iemocap")


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


def infer_speaker_id_from_path(path: str, subset: str | None = None) -> str:
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


def build_group_ids_from_paths(paths: Sequence[str]) -> List[str]:
    group_ids: List[str] = []
    for path in paths:
        subset = infer_subset_from_path(path)
        speaker_id = infer_speaker_id_from_path(path, subset=subset)
        group_ids.append(f"{subset}:{speaker_id}")
    return group_ids


def speaker_group_split(
    paths: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int], Dict[str, int]]:
    if not paths:
        raise ValueError("paths 为空，无法执行 speaker-group split")

    ratio_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"train/val/test 比例之和必须为 1.0，当前为 {ratio_sum:.6f}"
        )

    indices = np.arange(len(paths))
    group_ids = np.asarray(build_group_ids_from_paths(paths), dtype=object)
    unique_groups = np.unique(group_ids)

    if len(unique_groups) < 3 and val_ratio > 0 and test_ratio > 0:
        raise ValueError(
            f"唯一说话人组数量不足以划分 train/val/test: {len(unique_groups)}"
        )

    if test_ratio > 0:
        outer_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=float(test_ratio),
            random_state=int(seed),
        )
        train_val_rel, test_rel = next(
            outer_splitter.split(indices, groups=group_ids)
        )
        train_val_idx = indices[train_val_rel]
        test_idx = indices[test_rel]
    else:
        train_val_idx = indices
        test_idx = np.array([], dtype=np.int64)

    if val_ratio > 0:
        remaining_ratio = float(train_ratio) + float(val_ratio)
        inner_test_ratio = float(val_ratio) / remaining_ratio
        inner_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=inner_test_ratio,
            random_state=int(seed) + 1,
        )
        train_rel, val_rel = next(
            inner_splitter.split(train_val_idx, groups=group_ids[train_val_idx])
        )
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=np.int64)

    split_meta: Dict[str, int] = {
        "total_samples": int(len(indices)),
        "total_groups": int(len(unique_groups)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "train_groups": int(len(np.unique(group_ids[train_idx]))),
        "val_groups": int(len(np.unique(group_ids[val_idx]))),
        "test_groups": int(len(np.unique(group_ids[test_idx]))),
    }
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), split_meta
