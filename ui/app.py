"""
情感感知驱动的说话人语音识别系统 -- Gradio 界面。
支持实时录音和文件上传，展示转录文字（带情感颜色）、雷达图、波形图和频谱图。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from utils.audio_utils import load_config, load_audio, EMOTION_LABELS, EMOTION_NAMES_ZH, EMOTION_COLORS
from utils.visualization import (
    create_radar_chart, create_waveform_plot, create_mel_spectrogram_plot,
    create_emotion_text_html, create_emotion_history_chart,
)
from inference.pipeline import EmotionAwareSpeechPipeline

cfg = load_config()

pipeline = None
emotion_history = []


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = EmotionAwareSpeechPipeline(config=cfg)
    return pipeline


def process_audio(audio_input):
    """
    处理音频输入（来自录音或文件上传）。
    Gradio audio 组件返回 (sample_rate, numpy_array) 或文件路径。
    """
    global emotion_history

    if audio_input is None:
        return (
            "<p style='color:#999;padding:20px;'>请录音或上传音频文件</p>",
            create_radar_chart([0] * 6),
            None,
            None,
            create_emotion_history_chart([]),
        )

    pipe = get_pipeline()

    if isinstance(audio_input, tuple):
        sr, audio_array = audio_input
        audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / 32768.0
        result = pipe.process((audio_array, sr))
        display_audio = audio_array
        if sr != 16000:
            import librosa
            display_audio = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        display_sr = 16000
    else:
        result = pipe.process(audio_input)
        display_audio, display_sr = load_audio(audio_input, sr=16000)

    text_html = create_emotion_text_html(
        text=result["text"],
        emotion=result["emotion"],
        emotion_zh=result["emotion_zh"],
        color=result["emotion_color"],
        confidence=result["confidence"],
    )

    radar_fig = create_radar_chart(result["emotion_probs"])
    wave_fig = create_waveform_plot(display_audio, sr=display_sr)
    mel_fig = create_mel_spectrogram_plot(display_audio, sr=display_sr)

    emotion_history.append(result["emotion_probs"])
    if len(emotion_history) > 20:
        emotion_history = emotion_history[-20:]
    history_fig = create_emotion_history_chart(emotion_history)

    return text_html, radar_fig, wave_fig, mel_fig, history_fig


def clear_history():
    global emotion_history
    emotion_history = []
    return (
        "<p style='color:#999;padding:20px;'>已清空，请录音或上传音频文件</p>",
        create_radar_chart([0] * 6),
        None,
        None,
        create_emotion_history_chart([]),
    )


THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    font=gr.themes.GoogleFont("Noto Sans SC"),
)

CSS = """
.main-title {
    text-align: center;
    margin-bottom: 4px;
}
.main-title h1 {
    font-size: 1.8em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 0.95em;
    margin-bottom: 16px;
}
.legend-box {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    padding: 8px;
    margin-bottom: 8px;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.85em;
}
.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
}
"""


def build_legend_html():
    items = []
    for label in EMOTION_LABELS:
        zh = EMOTION_NAMES_ZH[label]
        color = EMOTION_COLORS[label]
        items.append(
            f'<span class="legend-item">'
            f'<span class="legend-dot" style="background:{color};"></span>'
            f'{zh}({label})'
            f'</span>'
        )
    return '<div class="legend-box">' + "".join(items) + "</div>"


def create_app():
    with gr.Blocks(theme=THEME, css=CSS, title="情感语音识别系统") as app:
        gr.HTML('<div class="main-title"><h1>情感感知驱动的说话人语音识别系统</h1></div>')
        gr.HTML('<div class="subtitle">基于 Whisper + CNN-BiLSTM-Attention 的语音识别与情感分析</div>')
        gr.HTML(build_legend_html())

        with gr.Row(equal_height=False):
            # ---- 左栏: 输入 ----
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 音频输入")
                audio_mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="录音",
                )
                audio_file = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="上传音频文件",
                )
                with gr.Row():
                    btn_mic = gr.Button("分析录音", variant="primary", size="sm")
                    btn_file = gr.Button("分析文件", variant="primary", size="sm")
                btn_clear = gr.Button("清空历史", variant="secondary", size="sm")

            # ---- 中栏: 结果 ----
            with gr.Column(scale=2, min_width=400):
                gr.Markdown("### 识别结果")
                text_output = gr.HTML(
                    value="<p style='color:#999;padding:20px;'>等待输入...</p>",
                    label="转录与情感",
                )
                radar_output = gr.Plot(label="情感分布雷达图")
                history_output = gr.Plot(label="情感变化趋势")

            # ---- 右栏: 可视化 ----
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 音频可视化")
                wave_output = gr.Plot(label="波形图")
                mel_output = gr.Plot(label="Mel 频谱图")

        outputs = [text_output, radar_output, wave_output, mel_output, history_output]

        btn_mic.click(fn=process_audio, inputs=[audio_mic], outputs=outputs)
        btn_file.click(fn=process_audio, inputs=[audio_file], outputs=outputs)
        btn_clear.click(fn=clear_history, inputs=[], outputs=outputs)

        gr.Markdown(
            "---\n"
            "**毕业设计** | 情感类别: 高兴·愤怒·悲伤·中性·恐惧·惊讶 | "
            "ASR: OpenAI Whisper | SER: CNN+BiLSTM+Attention"
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
        inbrowser=True,
    )
