"""
音频预处理模块：降噪、VAD静音切除、重采样、统一时长。
将原始音频处理后输出到 data/processed/{emotion_label}/ 目录。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import struct
from typing import Optional, Dict
import numpy as np
import librosa
import noisereduce as nr

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import load_config, load_audio, save_audio, pad_or_trim
from utils.data_policy import resolve_dataset_target_label


class AudioPreprocessor:
    def __init__(self, config=None):
        self.cfg = config or load_config()
        self.sr = self.cfg["audio"]["sample_rate"]
        self.max_dur = self.cfg["audio"]["max_duration"]
        self.min_dur = self.cfg["audio"]["min_duration"]
        self.target_samples = int(self.sr * self.max_dur)

    def denoise(self, audio):
        return nr.reduce_noise(y=audio, sr=self.sr, prop_decrease=0.8)

    def remove_silence(self, audio, top_db=25):
        """基于能量阈值的静音切除。"""
        intervals = librosa.effects.split(audio, top_db=top_db)
        if len(intervals) == 0:
            return audio
        trimmed = np.concatenate([audio[start:end] for start, end in intervals])
        return trimmed

    def normalize(self, audio):
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    def process_single(self, audio_path):
        """处理单个音频文件，返回预处理后的 numpy 数组。"""
        audio, _ = load_audio(audio_path, sr=self.sr)

        audio = self.denoise(audio)
        audio = self.remove_silence(audio)

        min_samples = int(self.sr * self.min_dur)
        if len(audio) < min_samples:
            return None

        audio = self.normalize(audio)
        audio = pad_or_trim(audio, self.target_samples)
        return audio

    def process_dataset(self, input_dir, output_dir):
        """
        批量预处理整个数据集。
        input_dir 下应按 {emotion_label}/{filename}.wav 组织，
        或为 RAVDESS 格式的扁平目录。
        """
        os.makedirs(output_dir, exist_ok=True)
        processed = 0
        skipped = 0

        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            for filepath in glob.glob(os.path.join(input_dir, "**", ext), recursive=True):
                rel = os.path.relpath(filepath, input_dir)
                out_path = os.path.join(output_dir, os.path.splitext(rel)[0] + ".wav")

                audio = self.process_single(filepath)
                if audio is None:
                    skipped += 1
                    continue

                save_audio(audio, out_path, sr=self.sr)
                processed += 1

        print(f"预处理完成: {processed} 个文件已处理, {skipped} 个跳过")
        return processed, skipped


def parse_ravdess_filename(filename):
    """
    RAVDESS 文件名格式: 03-01-05-01-02-01-12.wav
    第3位是情感编码。
    """
    parts = os.path.splitext(os.path.basename(filename))[0].split("-")
    if len(parts) >= 3:
        return parts[2]
    return None


def organize_ravdess(raw_dir, output_dir, emotion_map, config=None):
    """
    将 RAVDESS 扁平目录按情感标签重新组织到子文件夹。
    emotion_map: {"01": "neutral", "03": "happy", ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for filepath in glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True):
        emotion_code = parse_ravdess_filename(filepath)
        if emotion_code:
            label = resolve_dataset_target_label(config, "ravdess", emotion_code) if config is not None else emotion_map.get(emotion_code)
            if label is None:
                continue
            dest_dir = os.path.join(output_dir, label)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(filepath))

            audio, sr = load_audio(filepath)
            save_audio(audio, dest, sr=sr)
            count += 1
    print(f"RAVDESS 整理完成: {count} 个文件")
    return count


def organize_casia(raw_dir, output_dir, emotion_map, config=None):
    """
    将 CASIA 目录按情感标签重新组织。
    CASIA 结构: casia/{person}/{emotion}/{id}.wav
    输出结构:   output_dir/{emotion}/{person}_{id}.wav
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for filepath in glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True):
        parts = os.path.normpath(filepath).split(os.sep)
        # 找到 emotion 目录名（wav 文件的直接父目录）
        emotion_dir_name = parts[-2] if len(parts) >= 2 else None
        person_name = parts[-3] if len(parts) >= 3 else "unknown"

        if emotion_dir_name is None:
            continue

        # 匹配情感标签（支持中英文目录名）
        if config is not None:
            label = resolve_dataset_target_label(config, "casia", emotion_dir_name)
        else:
            label = emotion_map.get(emotion_dir_name)
            if label is None:
                label = emotion_map.get(emotion_dir_name.lower())
        if label is None:
            continue

        dest_dir = os.path.join(output_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        basename = os.path.basename(filepath)
        dest = os.path.join(dest_dir, f"{person_name}_{basename}")

        audio, sr = load_audio(filepath)
        save_audio(audio, dest, sr=sr)
        count += 1
    print(f"CASIA 整理完成: {count} 个文件")
    return count


def organize_tess(raw_dir, output_dir, emotion_map, config=None):
    """
    将 TESS 数据集按情感标签重新组织。
    TESS 结构: {speaker_emotion}/ 下有 {speaker}_{word}_{emotion}.wav
    文件夹名形如 OAF_angry, YAF_happy 等，情感从文件夹名中提取。
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for folder in os.listdir(raw_dir):
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        parts = folder.split("_", 1)
        emotion_key = parts[1].lower() if len(parts) > 1 else folder.lower()

        if config is not None:
            label = resolve_dataset_target_label(config, "tess", emotion_key)
            if label is None:
                label = resolve_dataset_target_label(config, "tess", emotion_key.replace(" ", "_"))
        else:
            label = emotion_map.get(emotion_key)
            if label is None:
                label = emotion_map.get(emotion_key.replace(" ", "_"))
        if label is None:
            continue

        dest_dir = os.path.join(output_dir, label)
        os.makedirs(dest_dir, exist_ok=True)

        for filepath in glob.glob(os.path.join(folder_path, "*.wav")):
            basename = os.path.basename(filepath)
            dest = os.path.join(dest_dir, f"tess_{basename}")
            audio, sr = load_audio(filepath)
            save_audio(audio, dest, sr=sr)
            count += 1
    print(f"TESS 整理完成: {count} 个文件")
    return count


def organize_esd(raw_dir, output_dir, emotion_map, config=None):
    """
    将 ESD 数据集按情感标签重新组织。
    ESD 结构: {speaker_id}/{emotion_name}/{####}.wav
    中文说话人编号 0011-0020，英文说话人编号 0001-0010。
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for filepath in glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True):
        parts = os.path.normpath(filepath).split(os.sep)
        emotion_dir_name = parts[-2] if len(parts) >= 2 else None
        speaker_id = parts[-3] if len(parts) >= 3 else "unknown"

        if emotion_dir_name is None:
            continue

        if config is not None:
            label = resolve_dataset_target_label(config, "esd", emotion_dir_name)
        else:
            label = emotion_map.get(emotion_dir_name)
            if label is None:
                label = emotion_map.get(emotion_dir_name.capitalize())
        if label is None:
            continue

        dest_dir = os.path.join(output_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        basename = os.path.basename(filepath)
        dest = os.path.join(dest_dir, f"esd_{speaker_id}_{basename}")

        audio, sr = load_audio(filepath)
        save_audio(audio, dest, sr=sr)
        count += 1
    print(f"ESD 整理完成: {count} 个文件")
    return count


def parse_emodb_filename(filename: str) -> Optional[str]:
    """Parse EMODB emotion code from filename.

    Typical EMODB filenames look like: 03a01Fa.wav
    The emotion code is a single letter in: W L E A F T N.

    Returns the uppercase emotion code, or None if not found.
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    # Fast path: many EMODB names end with the code letter + a trailing char (e.g., 'a'/'b')
    # We search for any known code letter anywhere in the basename and pick the last match.
    codes = "WLEAFTN"
    matches = [ch for ch in base.upper() if ch in codes]
    if matches:
        return matches[-1]

    return None


def organize_emodb(raw_dir: str, output_dir: str, emotion_map: Dict[str, str], config=None):
    """Organize flat EMODB wav files into label subfolders.

    Input:  data/raw/emodb/*.wav
    Output: data/raw/emodb_organized/<label>/emodb_<original>.wav

    emotion_map comes from cfg['datasets']['emodb']['emotions'].
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    skipped = 0

    for filepath in glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True):
        code = parse_emodb_filename(filepath)
        if not code:
            skipped += 1
            continue

        if config is not None:
            label = resolve_dataset_target_label(config, "emodb", code)
        else:
            label = emotion_map.get(code) or emotion_map.get(code.upper())
        if label is None:
            skipped += 1
            continue

        dest_dir = os.path.join(output_dir, label)
        os.makedirs(dest_dir, exist_ok=True)

        basename = os.path.basename(filepath)
        dest = os.path.join(dest_dir, f"emodb_{basename}")

        audio, sr = load_audio(filepath)
        save_audio(audio, dest, sr=sr)
        count += 1

    print(f"EMODB 整理完成: {count} 个文件, {skipped} 个跳过")
    return count, skipped


def parse_iemocap_labels(emo_eval_dir: str) -> Dict[str, str]:
    """
    解析 IEMOCAP EmoEvaluation 目录下的 .txt 文件。
    返回 {utterance_id: emotion_code} 字典。

    EmoEvaluation 文件格式示例：
    [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
    """
    label_dict = {}
    for txt_file in glob.glob(os.path.join(emo_eval_dir, "*.txt")):
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('['):
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    utterance_id = parts[1].strip()
                    emotion = parts[2].strip()
                    label_dict[utterance_id] = emotion
    return label_dict


def organize_iemocap(raw_dir: str, output_dir: str, emotion_map: Dict[str, str], config=None):
    """
    将 IEMOCAP 数据集按情感标签重新组织。
    IEMOCAP 结构: session{1-5}/sentences/wav/*.wav + dialog/EmoEvaluation/*.txt
    输出结构: output_dir/{emotion}/{utterance_id}.wav
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    skipped = 0

    for session in range(1, 6):
        session_dir = os.path.join(raw_dir, f"Session{session}")
        if not os.path.isdir(session_dir):
            # 尝试小写
            session_dir = os.path.join(raw_dir, f"session{session}")
            if not os.path.isdir(session_dir):
                continue

        # 解析情感标签
        emo_eval_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
        if not os.path.isdir(emo_eval_dir):
            continue
        label_dict = parse_iemocap_labels(emo_eval_dir)

        # 处理音频文件
        wav_dir = os.path.join(session_dir, "sentences", "wav")
        if not os.path.isdir(wav_dir):
            continue

        for speaker_dir in os.listdir(wav_dir):
            speaker_path = os.path.join(wav_dir, speaker_dir)
            if not os.path.isdir(speaker_path):
                continue

            for filepath in glob.glob(os.path.join(speaker_path, "*.wav")):
                basename = os.path.splitext(os.path.basename(filepath))[0]

                # 查找情感标签
                emotion_code = label_dict.get(basename)
                if emotion_code is None:
                    skipped += 1
                    continue

                # 映射到 6 类标签
                if config is not None:
                    label = resolve_dataset_target_label(config, "iemocap", emotion_code)
                else:
                    label = emotion_map.get(emotion_code)
                if label is None:
                    skipped += 1
                    continue

                dest_dir = os.path.join(output_dir, label)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, f"iemocap_{basename}.wav")

                audio, sr = load_audio(filepath)
                save_audio(audio, dest, sr=sr)
                count += 1

    print(f"IEMOCAP 整理完成: {count} 个文件, {skipped} 个跳过")
    return count, skipped


if __name__ == "__main__":
    cfg = load_config()
    preprocessor = AudioPreprocessor(cfg)

    raw = cfg["paths"]["raw_data"]
    processed = cfg["paths"]["processed_data"]

    ravdess_raw = os.path.join(raw, "ravdess")
    if os.path.isdir(ravdess_raw):
        ravdess_organized = os.path.join(raw, "ravdess_organized")
        organize_ravdess(
            ravdess_raw,
            ravdess_organized,
            cfg["datasets"]["ravdess"]["emotions"],
            config=cfg,
        )
        preprocessor.process_dataset(ravdess_organized, os.path.join(processed, "ravdess"))

    casia_raw = os.path.join(raw, "casia")
    if os.path.isdir(casia_raw):
        casia_organized = os.path.join(raw, "casia_organized")
        organize_casia(
            casia_raw,
            casia_organized,
            cfg["datasets"]["casia"]["emotions"],
            config=cfg,
        )
        preprocessor.process_dataset(casia_organized, os.path.join(processed, "casia"))

    tess_raw = os.path.join(raw, "tess")
    if os.path.isdir(tess_raw):
        tess_organized = os.path.join(raw, "tess_organized")
        organize_tess(
            tess_raw,
            tess_organized,
            cfg["datasets"]["tess"]["emotions"],
            config=cfg,
        )
        preprocessor.process_dataset(tess_organized, os.path.join(processed, "tess"))

    esd_raw = os.path.join(raw, "esd")
    if os.path.isdir(esd_raw):
        esd_organized = os.path.join(raw, "esd_organized")
        organize_esd(
            esd_raw,
            esd_organized,
            cfg["datasets"]["esd"]["emotions"],
            config=cfg,
        )
        preprocessor.process_dataset(esd_organized, os.path.join(processed, "esd"))

    emodb_raw = os.path.join(raw, "emodb")
    if os.path.isdir(emodb_raw):
        emodb_organized = os.path.join(raw, "emodb_organized")
        organize_emodb(
            emodb_raw,
            emodb_organized,
            cfg["datasets"]["emodb"]["emotions"],
            config=cfg,
        )
        preprocessor.process_dataset(emodb_organized, os.path.join(processed, "emodb"))

    iemocap_raw = os.path.join(raw, "iemocap")
    if os.path.isdir(iemocap_raw):
        iemocap_organized = os.path.join(raw, "iemocap_organized")
        organize_iemocap(
            iemocap_raw,
            iemocap_organized,
            cfg["datasets"]["iemocap"]["emotions"],
            config=cfg,
        )
        preprocessor.process_dataset(iemocap_organized, os.path.join(processed, "iemocap"))

    print("全部预处理完成。")
