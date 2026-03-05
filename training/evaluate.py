"""
评估脚本：生成混淆矩阵、分类报告、准确率/F1 等指标。
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import load_config, EMOTION_LABELS, EMOTION_NAMES_ZH
from preprocessing.feature_extract import EmotionDataset
from models.emotion_cnn_bilstm import EmotionRecognizer


def load_emotion_model(cfg, device):
    model = EmotionRecognizer(
        num_classes=cfg["emotion"]["num_classes"],
        n_mels=cfg["audio"]["n_mels"],
        cnn_channels=tuple(cfg["model"]["cnn_channels"]),
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
    )
    ckpt_path = cfg["paths"]["best_emotion_model"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_test_loader(cfg):
    features_dir = cfg["paths"]["features"]
    all_datasets = []
    for subset in ("ravdess", "casia"):
        d = os.path.join(features_dir, subset)
        if os.path.isdir(os.path.join(d, "mel")):
            all_datasets.append(EmotionDataset(d, feature_type="mel"))
    if not all_datasets:
        raise FileNotFoundError("未找到特征文件")

    full = torch.utils.data.ConcatDataset(all_datasets)
    total = len(full)
    train_n = int(total * cfg["training"]["train_ratio"])
    val_n = int(total * cfg["training"]["val_ratio"])
    test_n = total - train_n - val_n

    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    _, _, test_set = random_split(full, [train_n, val_n, test_n], generator=generator)
    return DataLoader(test_set, batch_size=cfg["training"]["batch_size"], shuffle=False)


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_preds = []
    all_labels = []
    all_probs = []

    for feats, labels in loader:
        feats = feats.to(device)
        logits = model(feats)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_training_history(history_path, save_dir):
    data = np.load(history_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(data["train_loss"], label="Train")
    ax1.plot(data["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(data["train_acc"], label="Train")
    ax2.plot(data["val_acc"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def evaluate_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_emotion_model(cfg, device)
    test_loader = get_test_loader(cfg)

    y_pred, y_true, y_probs = collect_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*50}")
    print(f"测试集评估结果")
    print(f"{'='*50}")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"加权 F1-Score:     {f1:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    plot_confusion_matrix(
        y_true, y_pred, EMOTION_LABELS,
        os.path.join(ckpt_dir, "confusion_matrix.png"),
    )

    history_path = os.path.join(ckpt_dir, "emotion_history.npz")
    if os.path.isfile(history_path):
        plot_training_history(history_path, ckpt_dir)

    return {"accuracy": acc, "f1_weighted": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估情感识别模型")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_model(cfg)
