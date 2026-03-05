"""
可视化工具：雷达图、波形图、Mel频谱图、训练曲线等。
用于 Gradio 界面和评估报告。
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

EMOTION_LABELS = ["happy", "angry", "sad", "neutral", "fear", "surprise"]
EMOTION_NAMES_ZH = {
    "happy": "高兴", "angry": "愤怒", "sad": "悲伤",
    "neutral": "中性", "fear": "恐惧", "surprise": "惊讶",
}
EMOTION_COLORS = {
    "happy": "#FF8C00", "angry": "#DC143C", "sad": "#4169E1",
    "neutral": "#808080", "fear": "#8B008B", "surprise": "#228B22",
}


def create_radar_chart(emotion_probs):
    """
    创建情感概率雷达图 (Plotly)。
    emotion_probs: dict {label: probability} 或 list/array 长度6。
    """
    if isinstance(emotion_probs, dict):
        labels = list(emotion_probs.keys())
        values = list(emotion_probs.values())
    else:
        labels = EMOTION_LABELS
        values = list(emotion_probs)

    display_labels = [f"{EMOTION_NAMES_ZH.get(l, l)}\n{l}" for l in labels]

    # 闭合雷达图
    display_labels_closed = display_labels + [display_labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=display_labels_closed,
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.3)",
        line=dict(color="rgb(99, 110, 250)", width=2),
        name="情感概率",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        title=dict(text="情感分布雷达图", x=0.5, font=dict(size=16)),
        margin=dict(l=60, r=60, t=60, b=60),
        height=400,
        width=450,
    )
    return fig


def create_waveform_plot(audio, sr=16000):
    """创建波形图 (Matplotlib)，返回 Figure 对象。"""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    t = np.arange(len(audio)) / sr
    ax.plot(t, audio, color="#4169E1", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim([0, t[-1] if len(t) > 0 else 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_mel_spectrogram_plot(audio, sr=16000, n_mels=128, hop_length=512):
    """创建 Mel 频谱图 (Matplotlib)，返回 Figure 对象。"""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    return fig


def create_emotion_text_html(text, emotion, emotion_zh, color, confidence):
    """生成带情感颜色标注的 HTML 文本。"""
    html = f"""
    <div style="padding: 16px; border-radius: 8px; border: 2px solid {color};
                background: linear-gradient(135deg, {color}15, {color}05);">
        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
            识别结果 | 情感: <span style="color: {color}; font-weight: bold;">
            {emotion_zh} ({emotion})</span>
            | 置信度: <strong>{confidence:.1%}</strong>
        </div>
        <div style="font-size: 20px; color: {color}; font-weight: 500;
                    line-height: 1.6; letter-spacing: 0.5px;">
            {text if text else '<em style="color:#999;">（未检测到语音内容）</em>'}
        </div>
    </div>
    """
    return html


def create_emotion_history_chart(history):
    """
    创建情感历史折线图。
    history: list of dicts, 每个 dict 包含 emotion_probs。
    """
    if not history:
        fig = go.Figure()
        fig.update_layout(title="情感变化趋势（暂无数据）", height=300)
        return fig

    n = len(history)
    x = list(range(1, n + 1))

    fig = go.Figure()
    for label in EMOTION_LABELS:
        y = [h.get(label, 0) for h in history]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=EMOTION_NAMES_ZH[label],
            line=dict(color=EMOTION_COLORS[label], width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title=dict(text="情感变化趋势", x=0.5),
        xaxis_title="分析次序",
        yaxis_title="概率",
        yaxis=dict(range=[0, 1]),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
