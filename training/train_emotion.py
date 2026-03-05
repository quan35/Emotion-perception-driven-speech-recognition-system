"""
CNN+BiLSTM+Attention 情感模型训练脚本。
可在本地或 Colab/AutoDL 上运行。
"""

import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import load_config
from preprocessing.feature_extract import EmotionDataset, AudioAugmentation
from models.emotion_cnn_bilstm import EmotionRecognizer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(cfg):
    features_dir = cfg["paths"]["features"]
    batch_size = cfg["training"]["batch_size"]

    all_feature_dirs = []
    for subset in ("ravdess", "casia"):
        d = os.path.join(features_dir, subset)
        if os.path.isdir(os.path.join(d, "mel")):
            all_feature_dirs.append(d)

    if not all_feature_dirs:
        raise FileNotFoundError(
            f"未找到特征文件，请先运行 preprocessing/feature_extract.py。"
            f"检查目录: {features_dir}"
        )

    augmentation = AudioAugmentation()
    datasets_train = []
    datasets_plain = []
    for d in all_feature_dirs:
        datasets_train.append(EmotionDataset(d, feature_type="mel", transform=augmentation))
        datasets_plain.append(EmotionDataset(d, feature_type="mel", transform=None))

    full_train = torch.utils.data.ConcatDataset(datasets_train)
    full_plain = torch.utils.data.ConcatDataset(datasets_plain)

    total = len(full_plain)
    train_n = int(total * cfg["training"]["train_ratio"])
    val_n = int(total * cfg["training"]["val_ratio"])
    test_n = total - train_n - val_n

    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_idx, val_idx, test_idx = random_split(
        range(total), [train_n, val_n, test_n], generator=generator,
    )

    train_set = torch.utils.data.Subset(full_train, train_idx.indices)
    val_set = torch.utils.data.Subset(full_plain, val_idx.indices)
    test_set = torch.utils.data.Subset(full_plain, test_idx.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"数据集划分: 训练={train_n}, 验证={val_n}, 测试={test_n}")
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += feats.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)

        total_loss += loss.item() * feats.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += feats.size(0)

    return total_loss / total, correct / total


def train(cfg):
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    model = EmotionRecognizer(
        num_classes=cfg["emotion"]["num_classes"],
        n_mels=cfg["audio"]["n_mels"],
        cnn_channels=tuple(cfg["model"]["cnn_channels"]),
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        lstm_dropout=cfg["model"]["dropout"],
        attn_dim=cfg["model"]["attention_dim"],
        cls_hidden=cfg["model"]["classifier_hidden"],
        cls_dropout=cfg["model"]["classifier_dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n开始训练 (共 {epochs} 轮)...\n")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_state, cfg["paths"]["best_emotion_model"])
            print(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: 验证损失连续 {patience} 轮未改善")
                break

    # 加载最佳模型评估测试集
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n测试集结果: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")

    # 保存训练历史
    np.savez(os.path.join(ckpt_dir, "emotion_history.npz"), **history)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 CNN+BiLSTM+Attention 情感模型")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
