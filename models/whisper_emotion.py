"""
毕设主线模型：Whisper Encoder + Transformer Emotion Head（theory.md §二）
=====================================================================

论文定位：基于预训练 Transformer 的迁移学习 SER 模型，是毕设的核心研究对象。

设计思路（theory.md §二.1）：
    与传统模型从零训练不同，主线模型采用 **迁移学习** 策略：
    - Whisper Encoder 在 68 万小时多样化音频上预训练，已学会提取丰富的通用声学特征
    - 在其输出序列之上，增加轻量 Transformer Emotion Head，专门建模情感时序依赖
    - 通过 Attention Pooling 聚焦情感关键帧，避免平均池化的信息稀释

    音频 → Whisper log-mel(80) → [Whisper Encoder] → Transformer Emotion Head
         → Attention Pooling → 分类头 → 情感类别

消融实验（theory.md §四）：
    A. 归一化策略：LayerNorm / DyT / Derf
    B. 微调策略：  freeze_all / unfreeze_last_2

总参数量: ~101M（其中可训练 ~14.6M，freeze_all 模式）
"""

from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


WHISPER_DIMS = {
    # Whisper 各规模的隐藏维度（theory.md §二.2.2）
    # 本项目默认使用 small（768 维）
    "tiny": 384,
    "base": 512,
    "small": 768,
    "medium": 1024,
    "large": 1280,
}

DEFAULT_SHARED_MODEL_CONFIG = {
    # 默认参数（theory.md §二.3.4）
    "variant": "transformer_head",       # 毕设主线
    "checkpoint_format": 2,
    "training_mode": "live_encoder",     # 直接输入音频，端到端前向
    "pooling": "attention",              # Attention Pooling（theory.md §二.4）
    "norm": "layernorm",                 # 归一化策略（消融实验 A）
    "freeze_strategy": "freeze_all",     # 冻结策略（消融实验 B）
    "head_layers": 2,                    # Transformer Head 层数
    "head_hidden_dim": None,             # 默认与 Whisper enc_dim 一致（768）
    "num_heads": 8,                      # 注意力头数（每头 96 维）
    "ff_mult": 4,                        # FFN 扩展倍数（中间维度 3072）
    "dropout": 0.1,
    "classifier_hidden": 256,            # 分类头：768 → 256 → 6
    "attention_pool_hidden": 256,        # Attention Pooling 隐藏维度
    "legacy_hidden_dims": [256, 64],     # legacy_mlp 兼容路径
    "cache_feature_dtype": "float16",
}


def get_shared_model_config(cfg: Optional[dict] = None, **overrides) -> Dict[str, Any]:
    shared_cfg = dict(DEFAULT_SHARED_MODEL_CONFIG)
    if cfg is not None:
        shared_cfg.update(cfg.get("shared_model", {}))
        model_cfg = cfg.get("model", {})
        if "whisper_size" in model_cfg:
            shared_cfg["whisper_size"] = model_cfg["whisper_size"]
    shared_cfg.update({k: v for k, v in overrides.items() if v is not None})
    return shared_cfg


def is_legacy_shared_checkpoint(ckpt: Dict[str, Any]) -> bool:
    return "classifier_state" in ckpt and "state_dict" not in ckpt


def _infer_encoder_dim(encoder, whisper_size: Optional[str] = None) -> int:
    ln_post = getattr(encoder, "ln_post", None)
    normalized_shape = getattr(ln_post, "normalized_shape", None)
    if isinstance(normalized_shape, Sequence) and normalized_shape:
        return int(normalized_shape[0])
    if isinstance(normalized_shape, int):
        return int(normalized_shape)
    if whisper_size in WHISPER_DIMS:
        return WHISPER_DIMS[whisper_size]
    raise ValueError("无法推断 Whisper encoder hidden dim")


def _normalize_strategy(name: str) -> str:
    return str(name).strip().lower()


def _normalize_variant(name: str) -> str:
    name = str(name).strip().lower()
    if name in {"legacy", "legacy_mlp", "mlp"}:
        return "legacy_mlp"
    if name in {"transformer", "transformer_head"}:
        return "transformer_head"
    raise ValueError(f"不支持的 shared model variant: {name}")


def _normalize_freeze_strategy(name: str) -> str:
    name = str(name).strip().lower()
    if name in {"freeze", "freeze_all", "all_frozen"}:
        return "freeze_all"
    if name in {"unfreeze_last_2", "last2", "last_2"}:
        return "unfreeze_last_2"
    if name in {"unfreeze_all", "train_all"}:
        return "unfreeze_all"
    raise ValueError(f"不支持的 freeze strategy: {name}")


def _to_hidden_dims(hidden_dims: Iterable[int]) -> Sequence[int]:
    dims = [int(dim) for dim in hidden_dims if int(dim) > 0]
    if not dims:
        raise ValueError("hidden_dims 不能为空")
    return dims


class DynamicTanh(nn.Module):
    """DyT：无归一化 Transformer 方案（theory.md §二.3.3，消融实验 A2）。

    来自论文 "Transformers without Normalization"，
    用可学习的逐元素非线性变换替代 LayerNorm：

        DyT(x) = gamma * tanh(alpha * x) + beta

    其中 alpha、gamma、beta 均为可学习参数（维度 = hidden_dim）。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(hidden_dim))
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class DynamicErf(nn.Module):
    """Derf：更强的无归一化 Transformer 方案（theory.md §二.3.3，消融实验 A3）。

    来自论文 "Stronger Normalization-Free Transformers"，
    在 DyT 基础上引入额外 shift 参数：

        Derf(x) = gamma * erf(alpha * x + shift) + beta

    其中 alpha、gamma、beta、shift 均为可学习参数。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(hidden_dim))
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.shift = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.erf(self.alpha * x + self.shift) + self.beta


def build_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    """构建归一化模块（theory.md §二.3.3，消融实验 A）。

    三种可替换的归一化策略，在 Transformer Block 的 Pre-Norm 位置使用：
      - layernorm: 标准 LayerNorm（A1，默认）
      - dyt:       DynamicTanh，无归一化 Transformer（A2）
      - derf:      DynamicErf，更强的无归一化方案（A3）
    """
    norm_type = _normalize_strategy(norm_type)
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "dyt":
        return DynamicTanh(hidden_dim)
    if norm_type == "derf":
        return DynamicErf(hidden_dim)
    raise ValueError(f"不支持的 norm 类型: {norm_type}")


class MeanPooling(nn.Module):
    """简单平均池化（theory.md §四，可选补充实验 C2）。

    将序列 (B, T, D) 沿时间维取平均得到 (B, D)。
    支持 attention_mask 以排除填充帧。

    缺点：情感信息往往集中在少数关键帧，平均池化会稀释情感峰值。
    用于与 AttentionPooling 的对比实验。
    """
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        if x.ndim == 2:
            pooled = x
            weights = None
        elif attention_mask is None:
            pooled = x.mean(dim=1)
            weights = None
        else:
            mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (x * mask).sum(dim=1) / denom
            weights = mask.squeeze(-1) / denom.squeeze(-1)
        if return_weights:
            return pooled, weights
        return pooled


class AttentionPooling(nn.Module):
    """可学习的注意力池化（theory.md §二.4，可选补充实验 C1）。

    通过可学习的评分网络自适应聚焦情感关键帧：
        x → Linear(D, pool_hidden) → Tanh → Linear(pool_hidden, 1) → Softmax → 加权求和

    相比 MeanPooling 的优势：
      - 情感信息往往集中在少数关键帧，平均池化会稀释情感峰值
      - AttentionPooling 自动学习哪些帧对情感判断最重要
    """
    def __init__(self, hidden_dim: int, pool_hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, pool_hidden_dim),
            nn.Tanh(),
            nn.Linear(pool_hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        if x.ndim == 2:
            pooled = x
            weights = None
            return (pooled, weights) if return_weights else pooled

        scores = self.score(x).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        if attention_mask is not None:
            valid = attention_mask.to(dtype=weights.dtype)
            weights = weights * valid
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        if return_weights:
            return pooled, weights
        return pooled


class MLPClassifier(nn.Module):
    """分类头（theory.md §二.5）。

    默认结构：Linear(768 → 256) → GELU → Dropout(0.1) → Linear(256 → 6)
    保持轻量，将主要学习能力留给 Transformer Emotion Head。
    """
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int, dropout: float):
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmotionTransformerBlock(nn.Module):
    """Transformer Emotion Head 的单层 Block（theory.md §二.3.2）。

    采用 Pre-Norm 结构，将归一化放在注意力/FFN 之前，训练更稳定：

        x → Norm₁ → Multi-Head Self-Attention → Dropout → 残差相加
          → Norm₂ → FFN(Linear → GELU → Dropout → Linear) → Dropout → 残差相加

    Pre-Norm 结构是归一化策略消融实验（A）的切入点：
    Norm₁ 和 Norm₂ 可替换为 LayerNorm / DyT / Derf。
    """
    def __init__(self, hidden_dim: int, num_heads: int, ff_mult: int, dropout: float, norm_type: str):
        super().__init__()
        self.norm1 = build_norm(norm_type, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = build_norm(norm_type, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        attn_input = self.norm1(x)
        attn_out, attn_weights = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout(attn_out)

        ff_input = self.norm2(x)
        ff_out = self.ffn(ff_input)
        x = x + self.dropout(ff_out)
        return x, attn_weights


class WhisperEmotionHead(nn.Module):
    """统一的 Whisper SER 模型入口（theory.md §二）。

    variant="legacy_mlp"（兼容基线）：
        Whisper Encoder → Mean Pool → MLP
        无 Transformer Head，无 Attention Pooling，仅用于向后兼容。

    variant="transformer_head"（毕设主线 ★）：
        Whisper Encoder → [线性投影] → N 层 Transformer Block(Pre-Norm)
                        → 输出归一化 → Attention Pooling → MLP 分类器

        默认维度流（theory.md §二.6）：
          输入 Whisper mel:         (B, 80, 3000)
          Encoder 输出:             (B, 1500, 768)
          Transformer Head 输出:    (B, 1500, 768)
          Attention Pooling 输出:   (B, 768)
          分类器:                   768 → 256 → 6

    冻结策略（theory.md §二.2.3，消融实验 B）：
      - freeze_all:      冻结全部 Encoder 参数，推理时 torch.no_grad()
      - unfreeze_last_2:  解冻最后 2 层 Transformer Block + ln_post
    """

    def __init__(
        self,
        whisper_model,
        num_classes: int = 6,
        freeze_encoder: Optional[bool] = True,
        variant: str = "legacy_mlp",
        freeze_strategy: Optional[str] = None,
        pooling: str = "attention",
        norm: str = "layernorm",
        head_layers: int = 2,
        head_hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        classifier_hidden: int = 256,
        attention_pool_hidden: int = 256,
        legacy_hidden_dims: Sequence[int] = (256, 64),
        whisper_size: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = whisper_model.encoder
        self.num_classes = int(num_classes)
        self.variant = _normalize_variant(variant)
        self.pooling_type = str(pooling).strip().lower()
        self.norm_type = _normalize_strategy(norm)
        self.whisper_size = whisper_size

        if freeze_strategy is None:
            freeze_strategy = "freeze_all" if freeze_encoder is not False else "unfreeze_all"
        self.freeze_strategy = _normalize_freeze_strategy(freeze_strategy)

        self.enc_dim = _infer_encoder_dim(self.encoder, whisper_size=whisper_size)
        self.head_hidden_dim = int(head_hidden_dim or self.enc_dim)
        self.num_heads = int(num_heads)
        self.head_layers = int(head_layers)
        self.ff_mult = int(ff_mult)
        self.dropout_p = float(dropout)

        if self.variant == "transformer_head" and self.head_hidden_dim % self.num_heads != 0:
            raise ValueError("head_hidden_dim 必须能被 num_heads 整除")

        self._apply_freeze_strategy()

        # ---- legacy_mlp 兼容基线（无 Transformer Head）----
        if self.variant == "legacy_mlp":
            self.input_proj = nn.Identity()
            self.transformer_blocks = nn.ModuleList()  # 空，无 Transformer 层
            self.output_norm = nn.Identity()
            self.pool = MeanPooling()                  # 简单平均池化
            self.classifier = MLPClassifier(
                input_dim=self.enc_dim,
                hidden_dims=_to_hidden_dims(legacy_hidden_dims),
                num_classes=self.num_classes,
                dropout=max(self.dropout_p, 0.2),
            )
        else:
            # ---- 毕设主线：Transformer Emotion Head（theory.md §二.3）----
            # 线性投影：当 head_hidden_dim != enc_dim 时统一维度
            self.input_proj = (
                nn.Identity() if self.head_hidden_dim == self.enc_dim
                else nn.Linear(self.enc_dim, self.head_hidden_dim)
            )
            # N 层 Transformer Block（Pre-Norm，theory.md §二.3.2）
            self.transformer_blocks = nn.ModuleList([
                EmotionTransformerBlock(
                    hidden_dim=self.head_hidden_dim,
                    num_heads=self.num_heads,
                    ff_mult=self.ff_mult,
                    dropout=self.dropout_p,
                    norm_type=self.norm_type,
                )
                for _ in range(self.head_layers)
            ])
            # 输出归一化（归一化策略由 norm_type 控制，消融实验 A）
            self.output_norm = build_norm(self.norm_type, self.head_hidden_dim)
            # Attention Pooling：聚焦情感关键帧（theory.md §二.4）
            self.pool = self._build_pool(self.pooling_type, self.head_hidden_dim, int(attention_pool_hidden))
            # 分类头（theory.md §二.5）
            self.classifier = MLPClassifier(
                input_dim=self.head_hidden_dim,
                hidden_dims=[int(classifier_hidden)],
                num_classes=self.num_classes,
                dropout=self.dropout_p,
            )

        self.shared_config = {
            "variant": self.variant,
            "freeze_strategy": self.freeze_strategy,
            "pooling": self.pooling_type,
            "norm": self.norm_type,
            "head_layers": self.head_layers,
            "head_hidden_dim": self.head_hidden_dim,
            "num_heads": self.num_heads,
            "ff_mult": self.ff_mult,
            "dropout": self.dropout_p,
            "classifier_hidden": int(classifier_hidden),
            "attention_pool_hidden": int(attention_pool_hidden),
            "legacy_hidden_dims": list(legacy_hidden_dims),
        }

    def _build_pool(self, pooling: str, hidden_dim: int, attention_pool_hidden: int) -> nn.Module:
        if pooling == "attention":
            return AttentionPooling(hidden_dim, attention_pool_hidden)
        if pooling == "mean":
            return MeanPooling()
        raise ValueError(f"不支持的 pooling 类型: {pooling}")

    def _apply_freeze_strategy(self):
        """应用 Whisper Encoder 冻结策略（theory.md §二.2.3，消融实验 B）。

        freeze_all:      冻结全部参数，防止灾难性遗忘（B1）
        unfreeze_last_2:  解冻最后 2 层 Block + ln_post，允许适配情感任务（B2）
        unfreeze_all:    解冻全部参数（非论文主线，仅供实验）
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.freeze_strategy == "freeze_all":
            return

        if self.freeze_strategy == "unfreeze_last_2":
            blocks = getattr(self.encoder, "blocks", None)
            if not blocks:
                raise ValueError("当前 Whisper encoder 不支持按 block 部分解冻")
            for block in blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
            ln_post = getattr(self.encoder, "ln_post", None)
            if ln_post is not None:
                for param in ln_post.parameters():
                    param.requires_grad = True
            return

        if self.freeze_strategy == "unfreeze_all":
            for param in self.encoder.parameters():
                param.requires_grad = True
            return

        raise ValueError(f"不支持的 freeze strategy: {self.freeze_strategy}")

    def _to_key_padding_mask(self, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        return ~attention_mask.bool()

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Whisper Encoder 前向传播（theory.md §二.2）。

        输入:  mel (B, 80, 3000) — Whisper 格式的 log-mel 频谱图
        输出:  (B, 1500, D) — 编码器输出的序列特征

        freeze_all 模式下使用 torch.no_grad() 避免计算梯度，节省显存。
        """
        if self.freeze_strategy == "freeze_all":
            with torch.no_grad():
                return self.encoder(mel)
        return self.encoder(mel)

    def forward_features(
        self,
        sequence_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        """Transformer Emotion Head 前向传播（theory.md §二.3）。

        对 Whisper Encoder 输出的序列特征进行情感任务适配：
          transformer_head: 投影 → N 层 Transformer Block → 输出归一化 → 池化 → 分类
          legacy_mlp:       直接池化 → MLP 分类

        Args:
            sequence_features: (B, T, D) 或 (B, D)（pooled 模式）
            attention_mask:    (B, T) 布尔掩码，True 表示有效帧
            return_details:    是否返回中间结果（注意力权重等，用于可视化）
        """
        if self.variant == "legacy_mlp":
            if sequence_features.ndim == 2:
                pooled = sequence_features
                pool_weights = None
            else:
                if return_details:
                    pooled, pool_weights = self.pool(sequence_features, attention_mask, return_weights=True)
                else:
                    pooled = self.pool(sequence_features, attention_mask, return_weights=False)
                    pool_weights = None
            logits = self.classifier(pooled)
            if return_details:
                return {
                    "logits": logits,
                    "pooled": pooled,
                    "pool_weights": pool_weights,
                    "sequence": sequence_features,
                    "attentions": [],
                }
            return logits

        if sequence_features.ndim != 3:
            raise ValueError("transformer_head 需要三维 sequence features: (B, T, D)")

        x = self.input_proj(sequence_features)
        key_padding_mask = self._to_key_padding_mask(attention_mask)
        attentions = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, key_padding_mask=key_padding_mask, return_attention=return_details)
            if return_details:
                attentions.append(attn_weights)

        x = self.output_norm(x)
        if return_details:
            pooled, pool_weights = self.pool(x, attention_mask, return_weights=True)
        else:
            pooled = self.pool(x, attention_mask, return_weights=False)
            pool_weights = None
        logits = self.classifier(pooled)

        if return_details:
            return {
                "logits": logits,
                "pooled": pooled,
                "pool_weights": pool_weights,
                "sequence": x,
                "attentions": attentions,
            }
        return logits

    def forward(
        self,
        mel: Optional[torch.Tensor] = None,
        sequence_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        if (mel is None) == (sequence_features is None):
            raise ValueError("mel 和 sequence_features 必须二选一")

        if sequence_features is None:
            sequence_features = self.encode(mel)

        outputs = self.forward_features(
            sequence_features=sequence_features,
            attention_mask=attention_mask,
            return_details=return_details,
        )
        if return_details:
            return outputs
        return outputs

    def predict_proba(
        self,
        mel: Optional[torch.Tensor] = None,
        sequence_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(
            mel=mel,
            sequence_features=sequence_features,
            attention_mask=attention_mask,
        )
        return F.softmax(logits, dim=-1)


def build_shared_model_from_config(
    whisper_model,
    cfg: dict,
    num_classes: Optional[int] = None,
    **overrides,
) -> WhisperEmotionHead:
    shared_cfg = get_shared_model_config(cfg, **overrides)
    return WhisperEmotionHead(
        whisper_model=whisper_model,
        num_classes=num_classes or cfg["emotion"]["num_classes"],
        variant=shared_cfg["variant"],
        freeze_strategy=shared_cfg["freeze_strategy"],
        pooling=shared_cfg["pooling"],
        norm=shared_cfg["norm"],
        head_layers=shared_cfg["head_layers"],
        head_hidden_dim=shared_cfg.get("head_hidden_dim"),
        num_heads=shared_cfg["num_heads"],
        ff_mult=shared_cfg["ff_mult"],
        dropout=shared_cfg["dropout"],
        classifier_hidden=shared_cfg["classifier_hidden"],
        attention_pool_hidden=shared_cfg["attention_pool_hidden"],
        legacy_hidden_dims=shared_cfg["legacy_hidden_dims"],
        whisper_size=shared_cfg.get("whisper_size") or cfg.get("model", {}).get("whisper_size"),
    )


def create_shared_checkpoint(
    model: WhisperEmotionHead,
    cfg: dict,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    shared_cfg = get_shared_model_config(cfg)
    stored_shared_cfg = {k: v for k, v in shared_cfg.items() if k != "whisper_size"}
    stored_shared_cfg.update(model.shared_config)
    checkpoint = {
        "format_version": int(shared_cfg.get("checkpoint_format", 2)),
        "model_variant": model.variant,
        "shared_model_config": stored_shared_cfg,
        "whisper_size": cfg.get("model", {}).get("whisper_size"),
        "num_classes": int(cfg["emotion"]["num_classes"]),
        "label_order": list(cfg["emotion"].get("labels", [])),
        "state_dict": model.state_dict(),
    }
    if extra:
        checkpoint.update(extra)
    return checkpoint
