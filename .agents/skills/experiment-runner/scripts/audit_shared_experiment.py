#!/usr/bin/env python3
"""核验共享模型实验产物，并输出可复用的摘要。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml


REQUIRED_SUMMARY_KEYS = (
    "experiment_name",
    "training_mode",
    "norm",
    "freeze_strategy",
    "best_metric",
    "best_epoch",
    "epochs_ran",
    "selected_val_subset_mean_uar",
    "selected_val_uar",
    "selected_val_loss",
    "best_val_subset_mean_uar",
    "best_val_uar",
    "best_val_loss",
    "test_subset_mean_uar",
    "uar",
    "macro_f1",
    "test_acc",
)

REQUIRED_HISTORY_KEYS = (
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "val_uar",
    "val_subset_mean_uar",
)

FLOAT_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="核验共享模型实验 summary/history/checkpoint 的一致性。",
    )
    parser.add_argument(
        "--run-name",
        help="实验运行名，例如 shared_transformer_head_live_encoder_attention_derf_unfreeze_last_2_seed42。",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="直接指定 summary.json 路径。",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="checkpoint 根目录，默认 checkpoints。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="配置文件路径，默认 configs/config.yaml。",
    )
    parser.add_argument(
        "--check-config-default",
        action="store_true",
        help="同时检查 config.yaml 中 paths.best_shared_model 是否指向当前实验 checkpoint。",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def close_enough(left: float, right: float, tol: float = FLOAT_TOL) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.summary is None and not args.run_name:
        fail("必须提供 --run-name 或 --summary。")

    summary_path = args.summary
    if summary_path is None:
        summary_path = args.checkpoints_dir / f"{args.run_name}_summary.json"

    if not summary_path.is_file():
        fail(f"未找到 summary 文件: {summary_path}")

    return summary_path.resolve(), args.checkpoints_dir.resolve()


def derive_artifact_path(summary: dict, summary_path: Path, key: str, fallback_name: str) -> Path:
    value = summary.get(key)
    if value:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (summary_path.parent / candidate.name).resolve()
    return (summary_path.parent / fallback_name).resolve()


def history_dict(npz_path: Path) -> dict[str, np.ndarray]:
    obj = np.load(npz_path, allow_pickle=True)
    return {key: obj[key] for key in obj.files}


def choose_best_epoch(history: dict[str, np.ndarray]) -> int:
    best_idx = 0
    for idx in range(1, len(history["val_loss"])):
        curr_subset = float(history["val_subset_mean_uar"][idx])
        best_subset = float(history["val_subset_mean_uar"][best_idx])
        if curr_subset > best_subset + FLOAT_TOL:
            best_idx = idx
            continue

        if close_enough(curr_subset, best_subset):
            curr_uar = float(history["val_uar"][idx])
            best_uar = float(history["val_uar"][best_idx])
            if curr_uar > best_uar + FLOAT_TOL:
                best_idx = idx
                continue

            if close_enough(curr_uar, best_uar):
                curr_loss = float(history["val_loss"][idx])
                best_loss = float(history["val_loss"][best_idx])
                if curr_loss < best_loss - FLOAT_TOL:
                    best_idx = idx

    return best_idx + 1


def check_required_keys(container: dict, required: tuple[str, ...], label: str, errors: list[str]) -> None:
    missing = [key for key in required if key not in container]
    if missing:
        errors.append(f"{label} 缺少字段: {', '.join(missing)}")


def main() -> int:
    args = parse_args()
    summary_path, checkpoints_dir = resolve_paths(args)
    summary = load_json(summary_path)

    run_name = summary.get("experiment_name") or args.run_name or summary_path.stem.removesuffix("_summary")
    checkpoint_path = derive_artifact_path(summary, summary_path, "checkpoint_filename", f"{run_name}.pth")
    history_path = derive_artifact_path(summary, summary_path, "history_filename", f"{run_name}_history.npz")
    curves_path = derive_artifact_path(summary, summary_path, "curve_filename", f"{run_name}_curves.png")
    confusion_path = derive_artifact_path(summary, summary_path, "confusion_matrix_filename", f"{run_name}_confusion_matrix.png")

    errors: list[str] = []
    infos: list[str] = []

    check_required_keys(summary, REQUIRED_SUMMARY_KEYS, "summary", errors)

    for label, path in (
        ("checkpoint", checkpoint_path),
        ("history", history_path),
        ("curves", curves_path),
        ("confusion_matrix", confusion_path),
    ):
        if not path.is_file():
            errors.append(f"{label} 文件不存在: {path}")

    if not history_path.is_file():
        for message in errors:
            print(f"[ERROR] {message}")
        return 1

    history = history_dict(history_path)
    check_required_keys(history, REQUIRED_HISTORY_KEYS, "history", errors)

    lengths = {key: len(history[key]) for key in REQUIRED_HISTORY_KEYS if key in history}
    if lengths:
        unique_lengths = sorted(set(lengths.values()))
        if len(unique_lengths) != 1:
            errors.append(f"history 长度不一致: {lengths}")
        else:
            epochs_ran = int(unique_lengths[0])
            infos.append(f"history epochs={epochs_ran}")
            if "epochs_ran" in summary and int(summary["epochs_ran"]) != epochs_ran:
                errors.append(
                    f"summary.epochs_ran={summary['epochs_ran']} 与 history 长度 {epochs_ran} 不一致"
                )

    if not errors:
        derived_best_epoch = choose_best_epoch(history)
        summary_best_epoch = int(summary["best_epoch"])
        if derived_best_epoch != summary_best_epoch:
            errors.append(
                f"best_epoch 不一致: summary={summary_best_epoch}, derived={derived_best_epoch}"
            )

        best_idx = summary_best_epoch - 1
        if best_idx < 0 or best_idx >= len(history["val_loss"]):
            errors.append(f"best_epoch 越界: {summary_best_epoch}")
        else:
            comparisons = (
                ("selected_val_subset_mean_uar", history["val_subset_mean_uar"][best_idx]),
                ("selected_val_uar", history["val_uar"][best_idx]),
                ("selected_val_loss", history["val_loss"][best_idx]),
                ("selected_val_macro_f1", history["val_macro_f1"][best_idx]),
                ("selected_val_acc", history["val_acc"][best_idx]),
                ("best_val_subset_mean_uar", np.max(history["val_subset_mean_uar"])),
                ("best_val_uar", np.max(history["val_uar"])),
                ("best_val_loss", np.min(history["val_loss"])),
            )
            for key, expected in comparisons:
                if key in summary and not close_enough(summary[key], expected):
                    errors.append(
                        f"{key} 不一致: summary={summary[key]}, derived={float(expected)}"
                    )

    if args.check_config_default:
        if not args.config.is_file():
            errors.append(f"未找到配置文件: {args.config}")
        else:
            cfg = load_config(args.config)
            configured = Path(cfg["paths"]["best_shared_model"])
            if not configured.is_absolute():
                configured = (args.config.parent.parent / configured).resolve()
            if configured != checkpoint_path.resolve():
                errors.append(
                    "config.yaml 的 paths.best_shared_model 未指向当前 checkpoint: "
                    f"{configured} != {checkpoint_path.resolve()}"
                )

    for message in infos:
        print(f"[INFO] {message}")

    for message in errors:
        print(f"[ERROR] {message}")

    if errors:
        return 1

    print("[OK] 实验产物核验通过")
    print(f"实验名: {summary['experiment_name']}")
    print(
        "配置: "
        f"training_mode={summary['training_mode']}, "
        f"norm={summary['norm']}, "
        f"freeze_strategy={summary['freeze_strategy']}"
    )
    print(f"监控规则: {summary['best_metric']}")
    print(f"best_epoch: {summary['best_epoch']}")
    print(f"selected_val_subset_mean_uar: {summary['selected_val_subset_mean_uar']:.6f}")
    print(f"selected_val_uar: {summary['selected_val_uar']:.6f}")
    print(f"test_subset_mean_uar: {summary['test_subset_mean_uar']:.6f}")
    print(f"test_uar: {summary['uar']:.6f}")
    print(f"macro_f1: {summary['macro_f1']:.6f}")
    print(f"test_acc: {summary['test_acc']:.6f}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {summary_path}")
    print(f"history: {history_path}")
    print(f"curves: {curves_path}")
    print(f"confusion_matrix: {confusion_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
