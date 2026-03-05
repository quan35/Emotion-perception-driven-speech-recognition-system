# 情感感知驱动的说话人语音识别系统

基于 Whisper + CNN-BiLSTM-Attention 的语音识别与情感分析系统。

## 功能

- 中英文语音实时转文字（基于 OpenAI Whisper）
- 语音情感识别（高兴、愤怒、悲伤、中性、恐惧、惊讶）
- 两种情感识别模型可在界面上实时切换
- 共享特征编码器方案（复用 Whisper Encoder 进行情感分类）
- Gradio 可视化交互界面（雷达图、波形图、情感着色文本）

## 系统架构

系统分为**训练阶段**和**使用阶段**：

```
训练阶段                              使用阶段（UI 界面）
┌──────────────────────┐        ┌──────────────────────────┐
│数据集（RAVDESS/CASIA）│        │ 用户录音 / 上传音频文件    │
│        ↓             │        │          ↓               │
│  预处理 + 特征提取    │        │  Whisper 语音转文字 (ASR) │
│        ↓             │ ─────→ │          ↓               │
│  训练情感模型         │        │  加载训练好的模型权重      │
│        ↓             │        │  进行情感识别 (SER)       │
│  保存 best_*.pth     │        │          ↓               │
└──────────────────────┘        │  可视化展示结果           │
                                └──────────────────────────┘
```

先训练出模型，再启动界面。界面上对新音频的分析全部依赖训练好的模型权重。

## 两种情感识别模型

|                   | CNN+BiLSTM+Attention                                                          | Whisper 共享编码器                                                            |
| ----------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **训练 notebook** | `notebooks/03_train_emotion.ipynb`                                            | `notebooks/04_train_shared.ipynb`                                             |
| **模型文件**      | `checkpoints/best_emotion.pth`                                                | `checkpoints/best_shared.pth`                                                 |
| **原理**          | 从 Mel 频谱图中用 CNN 提取局部特征，BiLSTM 建模时序，Attention 加权聚合关键帧 | 复用 Whisper 编码器（预训练于 68 万小时语音）提取特征，冻结编码器只训练分类头 |
| **特点**          | 自主设计的完整网络，训练所有参数                                              | 借助预训练大模型的迁移学习，训练参数少、收敛快                                |
| **毕设对应**      | 设计情感分类网络                                                              | 探索共享特征提取层                                                            |

界面左栏提供模型切换选项，可以对同一段音频分别用两种模型分析，便于在论文中对比效果。

## 置信度说明

界面上显示的"置信度"是模型输出经过 softmax 后的最大概率值。例如模型对 6 种情感的概率输出为 `[0.05, 0.02, 0.03, 0.80, 0.01, 0.09]`，则预测结果为"中性"，置信度 80%。

提高置信度和准确率的方法：

- **增加训练数据**：加入 CASIA 等更多数据集
- **数据增强**：启用 SpecAugment（时间/频率遮挡）
- **调整超参数**：在 `configs/config.yaml` 中修改学习率、batch size 等
- **使用共享编码器模型**：Whisper 的预训练特征质量通常优于从头训练的 CNN

## 数据集

支持 RAVDESS 和 CASIA 两个数据集，均为 **.wav** 格式音频文件。

**RAVDESS** — 放入 `data/raw/ravdess/`，脚本会自动按文件名中的情感编码整理：

```
data/raw/ravdess/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
└── Actor_24/
```

**CASIA** — 放入 `data/raw/casia/`，脚本会自动处理多层目录结构：

```
data/raw/casia/
├── liuchanhg/          # 说话人目录
│   ├── angry/
│   │   ├── 201.wav
│   │   └── ...
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   ├── fear/
│   └── surprise/
└── zhaoyan/            # 其他说话人
    └── ...
```

预处理时会自动将 `{人名}/{情感}/{文件}.wav` 扁平化为 `{情感}/{人名}_{文件}.wav`，无需手动整理。

## 运行环境

```bash
PyTorch  2.3.0
Python  3.12(ubuntu22.04)
CUDA  12.1
```

## 完整使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 预处理（降噪、静音切除、重采样、统一时长）
python preprocessing/audio_preprocess.py

# 3. 提取特征（Mel 频谱图 + MFCC）
python preprocessing/feature_extract.py

# 4. 训练模型（在 Jupyter 中逐格运行）
#    - notebooks/03_train_emotion.ipynb   → 训练 CNN+BiLSTM+Attention
#    - notebooks/04_train_shared.ipynb    → 训练 Whisper 共享编码器

# 5. 启动界面（加载训练好的模型，对新音频做实时分析）
python ui/app.py
```

## 项目结构

```
identifier/
├── configs/           # 配置文件（config.yaml 统一管理所有参数）
├── models/            # 模型定义（CNN+BiLSTM+Attention、Whisper 共享编码器）
├── preprocessing/     # 数据预处理（降噪/VAD）与特征提取（Mel/MFCC）
├── inference/         # 推理流水线（整合 ASR + SER，支持模型切换）
├── ui/                # Gradio 交互界面
├── notebooks/         # Jupyter 训练笔记本（推荐在 AutoDL 上运行）
├── utils/             # 工具函数（音频处理、可视化）
├── checkpoints/       # 训练产出的模型权重文件
├── data/
│   ├── raw/           # 原始数据集
│   ├── processed/     # 预处理后的音频
│   └── features/      # 提取的 Mel/MFCC 特征
└── requirements.txt   # 依赖文件
```
