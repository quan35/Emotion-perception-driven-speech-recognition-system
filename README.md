# 情感感知驱动的说话人语音识别系统

基于 Whisper + CNN-BiLSTM-Attention 的语音识别与情感分析系统。

## 功能

- 中英文语音实时转文字（基于 OpenAI Whisper）
- 语音情感识别（高兴、愤怒、悲伤、中性、恐惧、惊讶）
- 共享特征编码器方案（复用 Whisper Encoder 进行情感分类）
- Gradio 可视化交互界面（雷达图、波形图、情感着色文本）

## 安装

```bash
pip install -r requirements.txt
```

## 数据集

```
data/raw/
├── ravdess/        # 下载后解压放这里
│   ├── Actor_01/
│   │   ├── 03-01-01-01-01-01-01.wav
│   │   └── ...
│   └── Actor_24/
└── casia/          # CASIA 按情感目录组织
    ├── happy/
    ├── angry/
    ├── sad/
    ├── neutral/
    ├── fear/
    └── surprise/
```

## 预处理

```bash
python preprocessing/audio_preprocess.py
```

## 提取特征

```bash
python preprocessing/feature_extract.py
```

## 训练模型

notebooks/\*.ipynb

## 使用

```bash
python ui/app.py
```

## 项目结构

```
identifier/
├── configs/           # 配置文件
├── models/            # 模型定义
├── preprocessing/     # 数据预处理与特征提取
├── inference/         # 推理流水线
├── ui/                # Gradio 界面
├── notebooks/         # 训练笔记本
└── utils/             # 工具函数
```
