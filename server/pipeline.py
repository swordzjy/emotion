"""两种分析管线：run_paraformer_analysis / run_sensevoice_analysis"""

import re
import logging

import torch
import numpy as np

from .config import (
    SAMPLE_RATE,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SILENCE_MS,
)
from .models import ModelManager

logger = logging.getLogger(__name__)

# ---- 映射表 ----
EMOTION_LABELS_4 = ["中性", "生气", "开心", "悲伤"]

SENSEVOICE_EMOTION_MAP = {
    "HAPPY": "开心",
    "SAD": "悲伤",
    "ANGRY": "生气",
    "NEUTRAL": "中性",
    "SURPRISE": "惊讶",
    "FEARFUL": "恐惧",
    "DISGUSTED": "厌恶",
}

SENSEVOICE_LANG_MAP = {
    "zh": "中文",
    "en": "英文",
    "ja": "日语",
    "ko": "韩语",
    "yue": "粤语",
}

SENSEVOICE_EVENT_MAP = {
    "Speech": "语音",
    "Laughter": "笑声",
    "Applause": "掌声",
    "Cry": "哭泣",
    "Music": "音乐",
}


# ---- 通用工具 ----
def apply_vad(audio_tensor: torch.Tensor) -> tuple[torch.Tensor, list]:
    """VAD 过滤，返回 (有效语音 tensor, timestamps)"""
    mm = ModelManager()
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()

    with mm.vad_lock:
        get_speech_timestamps = mm.vad_utils[0]
        collect_chunks = mm.vad_utils[4]

        timestamps = get_speech_timestamps(
            audio_tensor,
            mm.vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        )

    if not timestamps:
        return audio_tensor, []

    speech_audio = collect_chunks(timestamps, audio_tensor)
    duration = sum(t["end"] - t["start"] for t in timestamps) / SAMPLE_RATE
    logger.info(f"VAD: {len(timestamps)} 段语音, 共 {duration:.2f}s")
    return speech_audio, timestamps


def compute_audio_features(audio_tensor: torch.Tensor) -> dict:
    """计算音频特征：响度、时长"""
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    loudness_db = (20 * torch.log10(rms + 1e-8)).item()
    duration_sec = len(audio_tensor) / SAMPLE_RATE
    return {"loudness_db": round(loudness_db, 1), "duration_sec": round(duration_sec, 2)}


def analyze_text_sentiment(text: str) -> dict | None:
    """SnowNLP 文本情感分析"""
    if not text:
        return None
    try:
        from snownlp import SnowNLP
        s = SnowNLP(text)
        score = s.sentiments
        label = "正面" if score > 0.6 else "负面" if score < 0.4 else "中性"
        return {"score": round(score, 4), "label": label}
    except Exception:
        return None


# ---- Paraformer 管线 ----
def run_paraformer_analysis(audio_tensor: torch.Tensor) -> dict:
    """Paraformer(ASR) + SpeechBrain(情感) 分析管线"""
    mm = ModelManager()

    # 1. VAD
    speech_audio, timestamps = apply_vad(audio_tensor)
    if not timestamps:
        return {"type": "result", "mode": "paraformer", "error": "未检测到语音"}

    # 2. Paraformer ASR
    audio_np = speech_audio.numpy() if isinstance(speech_audio, torch.Tensor) else speech_audio
    result = mm.paraformer_model.generate(input=audio_np, batch_size_s=300)

    transcript = ""
    if result and len(result) > 0:
        transcript = result[0].get("text", "")

    # 3. SpeechBrain 情感（若启用了 EMOTION_LAZY_LOAD 则首次分析时在此加载）
    mm.ensure_emotion_loaded()
    audio_input = speech_audio.unsqueeze(0) if speech_audio.dim() == 1 else speech_audio
    prob, score, index, emotion = mm.emotion_classifier.classify_batch(audio_input)

    pred_idx = index[0].item()
    pred_score = score[0].item()

    probabilities = {}
    emotion_keys = ["neu", "ang", "hap", "sad"]
    for i, key in enumerate(emotion_keys):
        probabilities[key] = round(prob[0][i].item(), 4)

    # 4. 音频特征
    audio_features = compute_audio_features(speech_audio)

    # 5. 文本情感
    text_sentiment = analyze_text_sentiment(transcript)

    return {
        "type": "result",
        "mode": "paraformer",
        "transcript": transcript,
        "emotion": {
            "label": emotion[0],
            "label_zh": EMOTION_LABELS_4[pred_idx],
            "confidence": round(pred_score, 4),
            "probabilities": probabilities,
        },
        "audio_features": audio_features,
        "text_sentiment": text_sentiment,
    }


# ---- SenseVoice 管线 ----
def run_sensevoice_analysis(audio_tensor: torch.Tensor) -> dict:
    """SenseVoice 一站式分析管线（ASR + 情感 + 事件）"""
    mm = ModelManager()

    # 1. VAD
    speech_audio, timestamps = apply_vad(audio_tensor)
    if not timestamps:
        return {"type": "result", "mode": "sensevoice", "error": "未检测到语音"}

    # 2. SenseVoice
    audio_np = speech_audio.numpy() if isinstance(speech_audio, torch.Tensor) else speech_audio
    result = mm.sensevoice_model.generate(
        input=audio_np,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
    )

    if not result or len(result) == 0:
        return {"type": "result", "mode": "sensevoice", "error": "识别失败"}

    raw_text = result[0].get("text", "")

    # 3. 解析 <|lang|><|emotion|><|event|>text
    pattern = r"<\|([^|]+)\|>"
    tags = re.findall(pattern, raw_text)
    clean_text = re.sub(pattern, "", raw_text).strip()

    language_code = tags[0] if len(tags) > 0 else "unknown"
    emotion_raw = tags[1] if len(tags) > 1 else "NEUTRAL"
    event_raw = tags[2] if len(tags) > 2 else "Speech"

    emotion = SENSEVOICE_EMOTION_MAP.get(emotion_raw, emotion_raw)
    event = SENSEVOICE_EVENT_MAP.get(event_raw, event_raw)
    language = SENSEVOICE_LANG_MAP.get(language_code, language_code)

    # 4. 音频特征
    audio_features = compute_audio_features(speech_audio)

    # 5. 文本情感
    text_sentiment = analyze_text_sentiment(clean_text)

    return {
        "type": "result",
        "mode": "sensevoice",
        "transcript": clean_text,
        "language": language,
        "language_code": language_code,
        "emotion": emotion,
        "emotion_raw": emotion_raw,
        "event": event,
        "event_raw": event_raw,
        "audio_features": audio_features,
        "text_sentiment": text_sentiment,
    }
