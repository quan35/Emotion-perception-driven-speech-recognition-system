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
    create_model_comparison_bar, create_comparison_result_html,
)
from inference.pipeline import EmotionAwareSpeechPipeline

cfg = load_config()
legacy_enabled = bool(cfg.get("legacy", {}).get("enabled", False))

pipeline = None
emotion_history = []

MODEL_CHOICES = [EmotionAwareSpeechPipeline.MODEL_SHARED]
if legacy_enabled:
    MODEL_CHOICES.append(EmotionAwareSpeechPipeline.MODEL_CNN)


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = EmotionAwareSpeechPipeline(config=cfg)
    return pipeline


def process_audio(audio_input, model_choice):
    """
    处理音频输入（来自录音或文件上传）。
    model_choice: 选择使用的情感识别模型。
    """
    global emotion_history

    empty_result = (
        "<p style='color:#999;padding:20px;'>请录音或上传音频文件</p>",
        create_radar_chart([0] * 6),
        None,
        None,
        create_emotion_history_chart([]),
    )

    if audio_input is None:
        return empty_result

    pipe = get_pipeline()
    pipe.set_model(model_choice)

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

    model_tag = f'<span style="display:inline-block;background:#eef;color:#446;padding:2px 8px;border-radius:4px;font-size:0.8em;margin-bottom:6px;">模型: {result["model_used"]}</span><br/>'

    text_html = model_tag + create_emotion_text_html(
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


def process_compare_audio(audio_input):
    """用两个模型对比分析同一段音频。"""
    empty = (
        "<p style='color:#999;padding:20px;'>请先录音或上传音频文件</p>",
        None,
    )
    if not legacy_enabled:
        return ("<p style='color:#999;padding:20px;'>当前默认仅启用 Whisper 主线，未开启早期探索模型对比。</p>", None)
    if audio_input is None:
        return empty

    pipe = get_pipeline()

    if isinstance(audio_input, tuple):
        sr, audio_array = audio_input
        audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / 32768.0
        compare_results = pipe.process_compare((audio_array, sr))
    else:
        compare_results = pipe.process_compare(audio_input)

    compare_html = create_comparison_result_html(compare_results)
    compare_bar = create_model_comparison_bar(compare_results)

    return compare_html, compare_bar


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
    with gr.Blocks(theme=THEME, css=CSS, title="情感感知驱动的说话人语音识别系统") as app:
        gr.HTML('<div class="main-title"><h1>情感感知驱动的说话人语音识别系统</h1></div>')
        if legacy_enabled:
            gr.HTML('<div class="subtitle">主线：Whisper + Transformer Emotion Head；探索：CNN+BiLSTM+Attention</div>')
        else:
            gr.HTML('<div class="subtitle">默认运行：Whisper + Transformer Emotion Head</div>')
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
                gr.Markdown("### 情感识别模型")
                selector_info = "Whisper+Transformer Emotion Head: 正式主线"
                if legacy_enabled:
                    selector_info += " | CNN+BiLSTM+Attention: 早期探索"
                model_selector = gr.Radio(
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES[0],
                    label="选择模型",
                    info=selector_info,
                )
                with gr.Row():
                    btn_mic = gr.Button("分析录音", variant="primary", size="sm")
                    btn_file = gr.Button("分析文件", variant="primary", size="sm")
                if legacy_enabled:
                    with gr.Row():
                        btn_compare_mic = gr.Button("对比录音", variant="secondary", size="sm")
                        btn_compare_file = gr.Button("对比文件", variant="secondary", size="sm")
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

        compare_html_output = None
        compare_bar_output = None
        compare_outputs = []
        if legacy_enabled:
            with gr.Accordion("两模型对比分析", open=False):
                compare_html_output = gr.HTML(
                    value="<p style='color:#999;padding:20px;'>点击「对比录音」或「对比文件」查看两模型对比结果</p>",
                    label="对比结果",
                )
                compare_bar_output = gr.Plot(label="模型情感预测对比柱状图")
            compare_outputs = [compare_html_output, compare_bar_output]

        outputs = [text_output, radar_output, wave_output, mel_output, history_output]

        btn_mic.click(fn=process_audio, inputs=[audio_mic, model_selector], outputs=outputs)
        btn_file.click(fn=process_audio, inputs=[audio_file, model_selector], outputs=outputs)
        if legacy_enabled:
            btn_compare_mic.click(fn=process_compare_audio, inputs=[audio_mic], outputs=compare_outputs)
            btn_compare_file.click(fn=process_compare_audio, inputs=[audio_file], outputs=compare_outputs)
        btn_clear.click(fn=clear_history, inputs=[], outputs=outputs)

        gr.Markdown(
            "---\n" + (
                "**毕业设计** | 情感类别: 高兴·愤怒·悲伤·中性·恐惧·惊讶 | "
                "ASR: OpenAI Whisper | SER: Whisper+Transformer Emotion Head（主线）"
                if not legacy_enabled
                else "**毕业设计** | 情感类别: 高兴·愤怒·悲伤·中性·恐惧·惊讶 | ASR: OpenAI Whisper | SER: CNN+BiLSTM+Attention（早期探索） · Whisper+Transformer Emotion Head（主线）"
            )
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
