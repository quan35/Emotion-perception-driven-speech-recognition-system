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

from utils.audio_utils import EMOTION_LABELS, EMOTION_NAMES_ZH, EMOTION_COLORS


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


# ---------- 三模型对比可视化 ----------

MODEL_DISPLAY_COLORS = {
    "CNN+BiLSTM+Attention": "#4169E1",
    "Whisper+Transformer": "#DC143C",
}


def create_model_comparison_bar(compare_results):
    """
    分组柱状图：各模型对各情感的预测概率。
    compare_results: dict {model_name: result_dict}，result_dict 包含 emotion_probs。
    """
    fig = go.Figure()

    for model_name, result in compare_results.items():
        probs = result["emotion_probs"]
        display_labels = [f"{EMOTION_NAMES_ZH.get(l, l)}\n{l}" for l in EMOTION_LABELS]
        values = [probs.get(l, 0) for l in EMOTION_LABELS]
        color = MODEL_DISPLAY_COLORS.get(model_name, "#888888")

        fig.add_trace(go.Bar(
            name=model_name,
            x=display_labels,
            y=values,
            marker_color=color,
            opacity=0.85,
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="模型情感预测对比", x=0.5, font=dict(size=16)),
        xaxis_title="情感类别",
        yaxis_title="预测概率",
        yaxis=dict(range=[0, 1]),
        height=420,
        margin=dict(l=50, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def create_comparison_result_html(compare_results):
    """
    HTML 并排卡片：展示各模型的预测结果。
    compare_results: dict {model_name: result_dict}。
    """
    if not compare_results:
        return "<p style='color:#999;padding:20px;'>暂无对比结果</p>"

    cards = []
    for model_name, result in compare_results.items():
        color = result.get("emotion_color", "#888")
        emotion_zh = result.get("emotion_zh", "")
        emotion = result.get("emotion", "")
        confidence = result.get("confidence", 0)
        model_color = MODEL_DISPLAY_COLORS.get(model_name, "#888")

        card = f"""
        <div style="flex:1; min-width:200px; padding:12px; border-radius:8px;
                    border:2px solid {color};
                    background:linear-gradient(135deg, {color}15, {color}05);">
            <div style="font-size:0.8em; color:{model_color}; font-weight:bold;
                        margin-bottom:6px; padding:2px 6px; background:{model_color}15;
                        border-radius:4px; display:inline-block;">
                {model_name}
            </div>
            <div style="font-size:1.3em; color:{color}; font-weight:bold; margin:6px 0;">
                {emotion_zh} ({emotion})
            </div>
            <div style="font-size:0.9em; color:#666;">
                置信度: <strong>{confidence:.1%}</strong>
            </div>
        </div>
        """
        cards.append(card)

    # 转录文本（取第一个模型的）
    first_result = next(iter(compare_results.values()))
    text = first_result.get("text", "")
    text_html = f"""
    <div style="margin-top:12px; padding:10px; border-radius:6px;
                background:#f8f9fa; border:1px solid #dee2e6;">
        <div style="font-size:0.8em; color:#666; margin-bottom:4px;">转录文本:</div>
        <div style="font-size:1.1em; color:#333; line-height:1.5;">
            {text if text else '<em style="color:#999;">（未检测到语音内容）</em>'}
        </div>
    </div>
    """

    html = f"""
    <div style="display:flex; gap:10px; flex-wrap:wrap;">
        {"".join(cards)}
    </div>
    {text_html}
    """
    return html
