"""
Whisper 共享编码器情感分类训练脚本。
冻结 Whisper Encoder，只训练分类头。
"""

import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import whisper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import load_config, load_audio, pad_or_trim, LABEL2ID
from models.whisper_emotion import WhisperEmotionHead
import glob


class WhisperMelDataset(Dataset):
    """
    直接加载音频，使用 Whisper 的 log_mel_spectrogram 生成输入。
    """

    def __init__(self, processed_dir):
        self.samples = []
        for label_name in sorted(os.listdir(processed_dir)):
            label_dir = os.path.join(processed_dir, label_name)
            if not os.path.isdir(label_dir) or label_name not in LABEL2ID:
                continue
            label_id = LABEL2ID[label_name]
            for wav_path in sorted(glob.glob(os.path.join(label_dir, "*.wav"))):
                self.samples.append((wav_path, label_id))
        print(f"WhisperMelDataset: 加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel, label


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg):
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 收集所有预处理后的数据集
    processed_dir = cfg["paths"]["processed_data"]
    all_datasets = []
    for subset in ("ravdess", "casia"):
        d = os.path.join(processed_dir, subset)
        if os.path.isdir(d):
            all_datasets.append(WhisperMelDataset(d))

    if not all_datasets:
        raise FileNotFoundError(f"未找到预处理数据，请先运行 preprocessing/audio_preprocess.py")

    full_dataset = torch.utils.data.ConcatDataset(all_datasets)
    total = len(full_dataset)
    train_n = int(total * cfg["training"]["train_ratio"])
    val_n = int(total * cfg["training"]["val_ratio"])
    test_n = total - train_n - val_n

    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_set, val_set, test_set = random_split(
        full_dataset, [train_n, val_n, test_n], generator=generator,
    )

    batch_size = min(cfg["training"]["batch_size"], 16)  # Whisper Mel 较大，减小 batch
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"数据集划分: 训练={train_n}, 验证={val_n}, 测试={test_n}")

    # 加载 Whisper 模型并构建情感分类头
    whisper_size = cfg["model"]["whisper_size"]
    print(f"加载 Whisper {whisper_size} 模型...")
    whisper_model = whisper.load_model(whisper_size, device=device)
    model = WhisperEmotionHead(
        whisper_model, num_classes=cfg["emotion"]["num_classes"], freeze_encoder=True,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: 总计={total_params:,}, 可训练={trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
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

        # 训练
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * mels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += mels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for mels, labels in val_loader:
                mels, labels = mels.to(device), labels.to(device)
                logits = model(mels)
                loss = criterion(logits, labels)
                val_loss += loss.item() * mels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += mels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
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
            best_model_state = copy.deepcopy(model.classifier.state_dict())
            patience_counter = 0
            torch.save({
                "classifier_state": best_model_state,
                "whisper_size": whisper_size,
                "num_classes": cfg["emotion"]["num_classes"],
            }, cfg["paths"]["best_shared_model"])
            print(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: 验证损失连续 {patience} 轮未改善")
                break

    # 测试集评估
    model.classifier.load_state_dict(best_model_state)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for mels, labels in test_loader:
            mels, labels = mels.to(device), labels.to(device)
            preds = model(mels).argmax(1)
            test_correct += (preds == labels).sum().item()
            test_total += mels.size(0)

    print(f"\n测试集准确率: {test_correct / test_total:.4f}")
    np.savez(os.path.join(ckpt_dir, "shared_history.npz"), **history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 Whisper 共享编码器情感分类模型")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
