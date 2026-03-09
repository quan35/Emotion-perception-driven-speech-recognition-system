# 情感感知驱动的语音识别系统

基于 Whisper + CNN-BiLSTM-Attention 的语音识别（ASR）与语音情感识别（SER）系统。

## 功能

- 中英文语音转文字（基于 OpenAI Whisper）
- 语音情感识别（高兴、愤怒、悲伤、中性、恐惧、惊讶）
- 两种情感识别模型可在界面上实时切换
- 共享特征编码器方案（复用 Whisper Encoder 进行情感分类）
- Gradio 可视化交互界面（雷达图、波形图、情感着色文本）

## 系统架构

系统分为**训练阶段**和**使用阶段**：

- **训练阶段**
  - **数据集**：RAVDESS / CASIA / TESS / ESD
  - **预处理**：`preprocessing/audio_preprocess.py`（降噪、静音切除、重采样、定长）
  - **特征提取**：`preprocessing/feature_extract.py`（Mel(128) + MFCC，保存为 `.npy`）
  - **训练**：notebooks 中训练两种 SER 模型
  - **产出**：`checkpoints/best_emotion.pth`、`checkpoints/best_shared.pth`

- **使用阶段**
  - **输入**：录音/上传音频（`ui/app.py`）
  - **ASR**：Whisper 转写（`inference/pipeline.py`）
  - **SER**：两种情感模型（可切换）（`inference/pipeline.py`）
  - **展示**：结果文本 + 可视化图表（`utils/visualization.py`）

先训练出模型，再启动界面。界面上对新音频的分析全部依赖训练好的模型权重。

## 两种情感识别模型

|                   | CNN+BiLSTM+Attention                                                          | Whisper 共享编码器                                                            |
| ----------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **训练 notebook** | `notebooks/03_train_emotion.ipynb`                                            | `notebooks/04_train_shared.ipynb`                                             |
| **模型文件**      | `checkpoints/best_emotion.pth`                                                | `checkpoints/best_shared.pth`                                                 |
| **原理**          | 从 Mel 频谱图中用 CNN 提取局部特征，BiLSTM 建模时序，Attention 加权聚合关键帧 | 复用 Whisper 编码器（预训练于 68 万小时语音）提取特征，冻结编码器只训练分类头 |
| **特点**          | 自主设计的完整网络，训练所有参数                                              | 借助预训练大模型的迁移学习，训练参数少、收敛快                                |

说明：

- `CNN+BiLSTM+Attention` 路径使用 `librosa` 提取 **Mel(128)** 特征进行情感识别。
- `Whisper 共享编码器` 路径使用 Whisper 内置的 **log-mel(80)** 特征，并复用 Whisper Encoder 做迁移学习。

## 置信度说明

界面上显示的"置信度"是模型输出经过 softmax 后的最大概率值。例如模型对 6 种情感的概率输出为 `[0.05, 0.02, 0.03, 0.80, 0.01, 0.09]`，则预测置信度 80%。

## 数据集

代码中支持多个公开数据集（均为 **.wav** 音频），并在 `configs/config.yaml` 中维护情感目录/编码映射：

- RAVDESS https://zenodo.org/record/1188976
- CASIA https://www.modelscope.cn/datasets/Westwest/CASIA
- TESS https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- ESD https://github.com/HLTSingapore/Emotional-Speech-Data

* 数据集位于项目文件夹之外，通过软链接接入

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

**TESS/ESD** — 若放入 `data/raw/tess/`、`data/raw/esd/`，同样会在预处理脚本中自动整理并输出到 `data/processed/`。

- `preprocessing/audio_preprocess.py` - 音频预处理脚本 会在 `data/raw` 目录中生成整理后的数据集 `*_organized/`。

## 运行环境

```bash
PyTorch  2.3.0
Python  3.12(ubuntu22.04)
CUDA  12.1
```

依赖版本以 `requirements.txt` 为准。

## 完整使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 预处理（数据集整理、降噪、基于能量阈值的静音切除、统一采样率与时长）
python preprocessing/audio_preprocess.py

# 3. 提取特征（Mel 频谱图 + MFCC）
python preprocessing/feature_extract.py

# 4. 训练模型（在 Jupyter 中逐格运行）
#    - notebooks/03_train_emotion.ipynb   → 训练 CNN+BiLSTM+Attention
#    - notebooks/04_train_shared.ipynb    → 训练 Whisper 共享编码器

# 5. 启动界面（加载训练好的模型，对新音频做实时分析）
python ui/app.py
```

运行提示：

- `预处理` 会在 `data/raw/` 下生成 `*_organized` 目录，并将预处理后的音频输出到 `data/processed/`。
- `提取特征` 会将特征输出到 `data/features/{dataset}/mel|mfcc/`。
- 若 `checkpoints/best_emotion.pth` 或 `checkpoints/best_shared.pth` 不存在，界面仍可启动，但情感模型会使用随机权重，结果不具备参考意义。

## 常见问题与排障

### 1) `libgomp: Invalid value for environment variable OMP_NUM_THREADS`

**现象**：运行预处理脚本或 notebook 时出现上述提示。

**原因**：OpenMP 运行库读取到非法的 `OMP_NUM_THREADS`（例如空字符串、非整数等），属于环境问题。

**处理**：

- 属于警告，可直接无视。

### 2) `04_train_shared.ipynb` 的“准备数据集”阶段内存/资源占用高

**现象**：首次提取 Whisper encoder 特征时内存占用大、GPU 利用率呈尖峰。

**原因**：

- 逐文件 I/O + 解码 + 特征计算存在 CPU/I/O 瓶颈。
- 若将所有特征先堆在 Python list 再 `torch.stack`，会导致内存峰值很高。

**解决**：

- 使用 `preprocessing/whisper_feature_cache.py` 将 pooled 特征落盘缓存（memmap 写入），避免重复提取。
- 用 `DataLoader(num_workers>0)` 并行读取音频，并在 GPU 端批量运行 encoder。

**缓存位置**：

- `data/features_shared/whisper_<size>_pooled_features.npy`
- `data/features_shared/whisper_<size>_pooled_labels.npy`

**如何强制重新提取**：删除上述缓存文件，或在 notebook 中设置 `overwrite=True`。

### 3) GPU 利用率高但显存占用不高（提特征阶段）

**解释**：前向推理 + `no_grad` + 混合精度会显著降低显存需求；同时工作负载可能是 compute-bound，因此出现“GPU 很忙但显存占用不高”。这属于正常现象。

### 4) 训练阶段 GPU 利用率低但训练很快（共享编码器方案）

**解释**：共享编码器方案在训练阶段只训练轻量 MLP 分类头；encoder 重计算已在“准备数据集/缓存特征”完成，因此训练阶段对 GPU 需求很小，速度快是预期行为。

## 项目结构

```
 identifier/
 ├── configs/           # 配置文件（config.yaml 统一管理所有参数）
 ├── models/            # 模型定义（CNN+BiLSTM+Attention、Whisper 共享编码器）
 ├── preprocessing/     # 数据预处理（降噪/静音切除/定长）与特征提取（Mel/MFCC）
 ├── inference/         # 推理流水线（整合 Whisper ASR + SER，支持模型切换）
 ├── ui/                # Gradio 交互界面
 ├── notebooks/         # Jupyter 训练笔记本（推荐在 AutoDL 上运行）
 ├── utils/             # 工具函数（音频处理、可视化）
 ├── checkpoints/       # 训练产出的模型权重文件
 ├── data/              # 通过软链接接入，该文件不在本项目仓库中
 │   ├── raw/           # 原始数据集
 │   ├── processed/     # 预处理后的音频
 │   └── features/      # 提取的 Mel/MFCC 特征
 └── requirements.txt   # 依赖文件
```
