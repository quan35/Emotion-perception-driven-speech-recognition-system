# 情感识别模型理论详解

本文档阐述项目中两条模型路线的设计原理与架构细节。

1. **传统基线模型**：从零训练的 `CNN+BiLSTM+Attention` 模型，用于对比实验。
2. **毕设主线模型**：基于 `Whisper Encoder + Transformer Emotion Head` 的迁移学习模型，支持归一化策略（LayerNorm / DyT / Derf）和冻结策略（freeze_all / unfreeze_last_2）的消融实验。

---

# 一、传统基线模型：EmotionRecognizer

## 1. 设计思路

语音情感识别（Speech Emotion Recognition, SER）的核心挑战在于：情感信息同时蕴含在语音的**局部声学特征**（如音高突变、能量爆发）和**全局时序模式**（如语速变化趋势、语调走向）中。

本模型采用 **CNN + BiLSTM + Attention** 三阶段级联架构，每个模块承担明确职责：

```
Mel 频谱图 → 残差SE-CNN(局部特征) → BiLSTM(时序建模) → 多头注意力(关键帧聚合) → 分类头 → 情感类别
```

| 阶段 | 模块 | 职责 |
| ---- | ---- | ---- |
| 1 | **残差 SE-CNN** | 从频谱图中提取局部声学模式 |
| 2 | **BiLSTM** | 沿时间轴双向建模上下文依赖关系 |
| 3 | **多头注意力** | 自适应聚焦并加权聚合情感最强烈的时序特征 |
| 4 | **分类头** | 将聚合后的特征映射到 6 种情感类别 |

## 2. 输入表示：Mel 频谱图

原始音频波形先转换为 **Mel 频谱图**——一种二维时频表示：

- **横轴**：时间帧（每帧约 32ms）
- **纵轴**：Mel 频率通道（`n_mels=128`）
- **值**：该时间-频率位置的对数能量

在默认配置（`n_mels=128`、`hop_length=512`、`n_fft=2048`）下，输入张量形状为 `(B, 1, 128, T)`。

Mel 刻度模拟人耳对频率的非线性感知，低频分辨率高、高频分辨率低，与人类对语音情感的感知特性一致。

## 3. 第一阶段：残差 SE-CNN 局部特征提取

由 3 个级联的残差 CNN 块组成，每个块融合了残差连接和 SE 通道注意力：

- **残差连接**：允许梯度通过快捷通道直接回传，缓解深度网络中的梯度消失问题。
- **SE 通道注意力**：通过全局平均池化 + 两层全连接网络，动态学习每个特征通道的重要性，放大关键特征（如共振峰、音高相关特征），抑制无关特征。
- **Spatial Dropout**：在特征图层面随机丢弃整个通道，强迫网络学习更多样化的特征。

```
输入 (B, 1, 128, T)
  → 残差SE-CNN 第1层 → (B, 32, 64, T/2)
  → 残差SE-CNN 第2层 → (B, 64, 32, T/4)
  → 残差SE-CNN 第3层 → (B, 128, 16, T/8)
  → 展平为序列      → (B, T/8, 2048)
```

## 4. 第二阶段：BiLSTM 时序建模

CNN 输出被重塑为 `(B, T', 2048)` 的序列，送入双层双向 LSTM。

- 输出维度：`2 × lstm_hidden = 128`
- 情感在语音中是上下文相关的，BiLSTM 通过前向和后向两个方向，使每个时间步都能融合过去和未来的信息。

## 5. 第三阶段：多头注意力加权聚合

BiLSTM 输出了特征序列，但分类器需要固定长度向量。本模型使用自定义的**多头自注意力时序聚合**机制（4 头，每头 32 维）：

1. **自注意力变换**：通过 Q/K/V 矩阵计算缩放点积注意力，生成上下文感知的序列
2. **时间维加权聚合**：从注意力分数中派生时序重要性权重，对上下文序列加权求和，得到 `(B, 128)` 的句级表示

多头机制让模型在独立子空间中学习不同的关注模式（如音高变化、能量爆发等），避免单头注意力的表达瓶颈。

## 6. 分类头

```
Linear(128 → 64) → ReLU → Dropout(0.5) → Linear(64 → 6)
```

## 7. 默认维度总结

| 位置 | 维度 |
| ---- | ---- |
| 输入 Mel | `(B, 1, 128, T)` |
| CNN 通道 | `(32, 64, 128)` |
| CNN 输出展平 | `(B, T', 2048)` |
| BiLSTM 输出 | `(B, T', 128)` |
| 注意力头数 | 4 |
| 句级表示 | `(B, 128)` |
| 分类器 | `128 → 64 → 6` |
| 总参数量 | ~1.4M（全部参与训练） |

## 8. 训练策略

- **高斯噪声注入**：训练时在输入频谱图上叠加微小噪声（std=0.01），提升鲁棒性
- **权重初始化**：Conv 用 Kaiming、LSTM 用 Orthogonal、Linear 用 Xavier
- **AdamW 优化器**：解耦权重衰减
- **LayerNorm**：在 BiLSTM 和 Attention 之间稳定输入分布
- **Label Smoothing / Focal Loss**：可选，按数据分布选择

---

# 二、毕设主线模型：Whisper + Transformer Emotion Head

## 1. 设计思路

与传统模型从零训练不同，主线模型采用**迁移学习**策略：

- **Whisper Encoder** 在 68 万小时多样化音频上预训练，已学会提取丰富的通用声学特征
- 在其输出序列之上，增加一个轻量 **Transformer Emotion Head**，专门建模情感时序依赖
- 通过 **Attention Pooling** 聚焦情感关键帧，避免平均池化的信息稀释

```
音频 → Whisper log-mel → [Whisper Encoder] → Transformer Emotion Head → Attention Pooling → 分类头 → 情感类别
```

| 模块 | 作用 | 研究重点 |
| ---- | ---- | ---- |
| **Whisper Encoder** | 提供强通用语音表征 | 冻结/部分解冻策略 |
| **Transformer Emotion Head** | 情感任务的序列关系重建 | **核心模块**，归一化策略消融 |
| **Attention Pooling** | 聚焦情感关键帧 | 可与 Mean Pooling 对比 |
| **分类头** | 输出 6 类情感 logits | 保持轻量 |

## 2. Whisper Encoder

### 2.1 输入

Whisper 使用自己的频谱参数：`n_mels=80`，音频统一填充/裁剪为 30 秒，对应 1500 个编码帧。输入形状为 `(B, 80, 3000)`。

### 2.2 架构

Whisper Encoder 是一个 **Transformer** 网络，由多层 Transformer Block 堆叠而成。输出特征序列形状为 `(B, 1500, D)`，其中 `D` 取决于模型规模：

| Whisper 规模 | 隐藏维度 D |
| ---- | ---- |
| tiny | 384 |
| base | 512 |
| **small**（默认） | **768** |
| medium | 1024 |
| large | 1280 |

### 2.3 冻结策略

| 策略 | 行为 | 适用场景 |
| ---- | ---- | ---- |
| `freeze_all` | 冻结全部 Encoder 参数，推理时使用 `torch.no_grad()` | 默认，防止灾难性遗忘 |
| `unfreeze_last_2` | 解冻最后 2 层 Transformer Block + `ln_post` | 允许 Encoder 适配情感任务 |

冻结策略是消融实验 B 的核心变量。

## 3. Transformer Emotion Head

这是毕设的**核心创新模块**。在 Whisper Encoder 输出的序列特征之上，增加轻量 Transformer 层进行情感任务适配。

### 3.1 结构

```
Whisper 序列特征 (B, 1500, 768)
  → 线性投影（可选，统一维度）
  → N 层 Transformer Encoder Block（Pre-Norm）
  → 输出归一化
  → Attention Pooling → (B, 768)
  → MLP 分类器 → (B, 6)
```

### 3.2 Transformer Block（Pre-Norm 结构）

每个 Block 的内部结构：

```
x → Norm₁ → Multi-Head Self-Attention → Dropout → 残差相加
  → Norm₂ → FFN(Linear → GELU → Dropout → Linear) → Dropout → 残差相加
```

Pre-Norm 结构将归一化放在注意力/FFN 之前，训练更稳定，这也是归一化策略消融实验的切入点。

### 3.3 归一化策略（消融实验 A）

三种可替换的归一化模块：

**LayerNorm（标准方案）**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

**DynamicTanh / DyT（无归一化 Transformer）**

来自论文 *Transformers without Normalization*，用可学习的逐元素非线性变换替代 LayerNorm：

$$\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$$

其中 $\alpha$、$\gamma$、$\beta$ 均为可学习参数。

**DynamicErf / Derf（更强的无归一化方案）**

来自论文 *Stronger Normalization-Free Transformers*，引入额外的偏移参数：

$$\text{Derf}(x) = \gamma \cdot \text{erf}(\alpha x + s) + \beta$$

其中 $\alpha$、$\gamma$、$\beta$、$s$ 均为可学习参数。

### 3.4 默认参数

| 参数 | 默认值 |
| ---- | ---- |
| Transformer 层数 | 2 |
| 隐藏维度 | 768（与 Whisper small 一致） |
| 注意力头数 | 8（每头 96 维） |
| FFN 扩展倍数 | 4（即 FFN 中间维度 3072） |
| Dropout | 0.1 |
| 归一化 | LayerNorm |

## 4. Attention Pooling

Transformer Head 输出仍是序列 `(B, T, D)`，需要聚合为句级表示。

**Attention Pooling** 通过可学习的评分网络自适应聚焦情感关键帧：

```
x → Linear(D, 256) → Tanh → Linear(256, 1) → Softmax → 加权求和
```

相比 Mean Pooling 的优势：
- 情感信息往往集中在少数关键帧，平均池化会稀释情感峰值
- Attention Pooling 自动学习哪些帧对情感判断最重要

## 5. 分类头

```
Linear(768 → 256) → GELU → Dropout(0.1) → Linear(256 → 6)
```

## 6. 默认维度总结

| 位置 | 维度 |
| ---- | ---- |
| 输入 Whisper mel | `(B, 80, 3000)` |
| Encoder 输出 | `(B, 1500, 768)` |
| Transformer Head 输出 | `(B, 1500, 768)` |
| Attention Pooling 输出 | `(B, 768)` |
| 分类器 | `768 → 256 → 6` |
| 总参数量 | ~101M（其中可训练 ~14.6M） |

## 7. 训练配置

- **AMP 混合精度**：`live_encoder` 模式下自动启用 float16，减少显存占用
- **梯度累积**：默认 32 步累积，等效 batch size 256
- **AdamW 优化器**：学习率 5e-4，权重衰减 1e-5
- **ReduceLROnPlateau**：验证集 loss 连续 5 轮无改善时学习率减半
- **早停**：验证集 loss 连续 15 轮无改善时停止训练

---

# 三、两模型对比

| 特性 | **CNN+BiLSTM+Attention** | **Whisper+Transformer** |
| ---- | ---- | ---- |
| **核心架构** | 残差SE-CNN + BiLSTM + 多头注意力 | Whisper Encoder + Transformer Head + Attention Pooling |
| **训练范式** | 从零训练 | 迁移学习 |
| **知识来源** | 完全依赖当前情感数据集 | Whisper 68万小时预训练 + 情感数据微调 |
| **总参数量** | ~1.4M（全部训练） | ~101M（可训练 ~14.6M） |
| **特征特异性** | 完全针对情感分类 | 通用声学特征 + 情感任务适配 |
| **跨域泛化** | 受限于训练数据分布 | 天然具备跨域鲁棒性 |
| **训练成本** | 中等，需完整训练整个网络 | 较高（live_encoder 模式），但收敛更快 |
| **论文定位** | 传统架构对照组 | **毕设主线** |

---

# 四、消融实验设计

## 主模型

**Whisper + Transformer Emotion Head + Attention Pooling**

## 消融实验 A：归一化策略

| 编号 | 归一化 | 说明 |
| ---- | ---- | ---- |
| A1 | LayerNorm | 标准 Transformer 归一化 |
| A2 | DyT | 无归一化 Transformer（可学习 tanh） |
| A3 | Derf | 更强的无归一化方案（可学习 erf） |

研究问题：在 Whisper + Transformer Emotion Head 的 SER 任务中，无归一化设计能否提升泛化能力与训练稳定性？

## 消融实验 B：微调策略

| 编号 | 冻结策略 | 说明 |
| ---- | ---- | ---- |
| B1 | freeze_all | 冻结全部 Encoder，只训练 Head |
| B2 | unfreeze_last_2 | 解冻最后 2 层 Encoder Block |

研究问题：在小规模 SER 数据集上，哪种微调策略能在泛化能力、训练成本和过拟合风险之间取得更好平衡？

## 可选补充实验：聚合策略

| 编号 | 聚合方式 | 说明 |
| ---- | ---- | ---- |
| C1 | Attention Pooling | 可学习的关键帧聚焦 |
| C2 | Mean Pooling | 简单平均 |

---

# 五、论文叙事逻辑

本课题形成清晰的递进关系：

> 传统深度学习 SER 基线（CNN+BiLSTM+Attention）→ 预训练 Transformer 迁移学习主线（Whisper+Transformer Emotion Head）→ 归一化策略与微调策略的消融研究

具体叙事：

1. 先实现传统深度学习 SER 模型，发现其跨域泛化受限
2. 转向预训练 Transformer，利用 Whisper 的通用声学表征
3. 在 Whisper Encoder 之上设计 Transformer Emotion Head，进行情感任务适配
4. 研究 DyT / Derf 等无归一化 Transformer 在 SER 任务中的适用性
5. 研究不同冻结策略对小样本 SER 的影响
