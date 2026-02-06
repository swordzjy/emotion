"""
语音情感识别程序 v4.1 - SenseVoice 版本
使用阿里 SenseVoice 模型，同时支持：
- 语音识别（中英日韩粤 50+ 语言）
- 情感识别（开心/悲伤/生气/中性/惊讶）
- 语音事件检测（笑声、掌声、咳嗽等）

SenseVoice 是一站式方案，无需单独加载情感模型！
"""

import pyaudio
import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# ============== 配置参数 ==============
SAMPLE_RATE = 16000
RECORD_SECONDS = 8
CHUNK = 512
# ============== 配置 ffmpeg 路径 ==============
FFMPEG_PATH = r"E:\Python\ffmpeg\bin"  # ← 确认这个路径

if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

# ============== 1. 加载模型 ==============
print("=" * 50)
print("正在加载 SenseVoice 模型...")
print("（首次运行会下载约 900MB）")
print("=" * 50)
SILERO_VAD_PATH = r"E:\Workspace\claude\emotion\silero-vad"  # ← 本地 Silero VAD 仓库路径
# 1.1 加载 Silero VAD
print("\n[1/2] 加载 Silero VAD...")
# 如果本地目录存在，从本地加载；否则从网络下载
if os.path.exists(SILERO_VAD_PATH):
    print(f"   从本地加载: {SILERO_VAD_PATH}")
    vad_model, utils = torch.hub.load(
        repo_or_dir=SILERO_VAD_PATH,
        model='silero_vad',
        source='local',  # 关键：指定本地源
        onnx=False
    )
else:
    print("   本地模型不存在，从网络下载...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("✓ Silero VAD 加载完成")

# 1.2 加载 SenseVoice
print("\n[2/2] 加载 SenseVoice...")
from funasr import AutoModel

# SenseVoice-Small: 多语言 + 情感 + 事件检测
sensevoice_model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    disable_update=True,
    device="cpu"
)
print("✓ SenseVoice 加载完成")
print("  支持: 语音识别 + 情感识别 + 语音事件检测")

print("\n" + "=" * 50)
print("模型加载完成！")
print("=" * 50)


# ============== 2. 录音函数 ==============
def capture_audio(duration=RECORD_SECONDS, rate=SAMPLE_RATE):
    """录制音频"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print(f"\n🎤 正在录音 {duration} 秒...")
    frames = []
    for i in range(0, int(rate / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
        # 进度显示
        if i % 30 == 0:
            print(f"\r   录音中: {i * CHUNK / rate:.1f}s / {duration}s", end="")
    
    print(f"\r✓ 录音结束: {duration}s                    ")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio = np.hstack(frames).astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


# ============== 3. Silero VAD 处理 ==============
def apply_silero_vad(audio_tensor, sample_rate=SAMPLE_RATE):
    """使用 Silero VAD 提取有效语音段"""
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sample_rate,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30
    )
    
    if not speech_timestamps:
        print("⚠️ 未检测到有效语音！")
        return audio_tensor, []
    
    speech_audio = collect_chunks(speech_timestamps, audio_tensor)
    
    total_speech = sum(ts['end'] - ts['start'] for ts in speech_timestamps) / sample_rate
    print(f"✓ VAD 检测到 {len(speech_timestamps)} 个语音段")
    print(f"  有效语音时长: {total_speech:.2f} 秒")
    
    return speech_audio, speech_timestamps


# ============== 4. SenseVoice 识别 ==============
def recognize_with_sensevoice(audio_tensor, sample_rate=SAMPLE_RATE):
    """
    使用 SenseVoice 进行识别
    返回: 文本、语言、情感、事件
    """
    audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
    
    # SenseVoice 识别
    result = sensevoice_model.generate(
        input=audio_np,
        cache={},
        language="auto",  # 自动检测语言，或 "zh", "en", "ja", "ko", "yue"
        use_itn=True,     # 逆文本归一化（数字、日期等转换）
        batch_size_s=60,
    )
    
    if not result or len(result) == 0:
        return {
            'text': '',
            'language': 'unknown',
            'emotion': 'unknown',
            'event': None,
            'raw': None
        }
    
    raw_text = result[0].get('text', '')
    
    # 解析 SenseVoice 输出格式: <|语言|><|情感|><|事件|>文本内容
    # 例如: <|zh|><|HAPPY|><|Speech|>今天天气真好
    parsed = parse_sensevoice_output(raw_text)
    
    return parsed


def parse_sensevoice_output(text):
    """
    解析 SenseVoice 输出
    格式: <|lang|><|emotion|><|event|>实际文本
    """
    import re
    
    # 情感映射
    emotion_map = {
        'HAPPY': '开心 😊',
        'SAD': '悲伤 😢',
        'ANGRY': '生气 😠',
        'NEUTRAL': '中性 😐',
        'SURPRISE': '惊讶 😲',
        'FEARFUL': '恐惧 😰',
        'DISGUSTED': '厌恶 🤢',
    }
    
    # 事件映射
    event_map = {
        'Speech': '语音',
        'Laughter': '笑声 😄',
        'Applause': '掌声 👏',
        'Cry': '哭泣 😭',
        'Cough': '咳嗽',
        'Sneeze': '喷嚏',
        'Music': '音乐 🎵',
        'BGM': '背景音乐',
    }
    
    # 提取标签
    pattern = r'<\|([^|]+)\|>'
    tags = re.findall(pattern, text)
    
    # 移除标签获取纯文本
    clean_text = re.sub(pattern, '', text).strip()
    
    # 解析
    language = tags[0] if len(tags) > 0 else 'unknown'
    emotion_raw = tags[1] if len(tags) > 1 else 'NEUTRAL'
    event_raw = tags[2] if len(tags) > 2 else 'Speech'
    
    # 语言映射
    lang_map = {
        'zh': '中文',
        'en': '英文',
        'ja': '日语',
        'ko': '韩语',
        'yue': '粤语',
    }
    
    return {
        'text': clean_text,
        'language': lang_map.get(language, language),
        'language_code': language,
        'emotion': emotion_map.get(emotion_raw, emotion_raw),
        'emotion_raw': emotion_raw,
        'event': event_map.get(event_raw, event_raw),
        'event_raw': event_raw,
        'raw': text
    }


# ============== 5. 文本情感分析 ==============
def analyze_text_sentiment(text):
    """分析文本情感（作为补充）"""
    try:
        from snownlp import SnowNLP
        s = SnowNLP(text)
        return {
            'score': s.sentiments,
            'sentiment': "正面" if s.sentiments > 0.6 else "负面" if s.sentiments < 0.4 else "中性"
        }
    except:
        return None


# ============== 6. 音频特征 ==============
def analyze_audio_features(audio_tensor):
    """分析音频特征"""
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    loudness_db = 20 * torch.log10(rms + 1e-8)
    
    return {
        'loudness_db': loudness_db.item(),
        'duration': len(audio_tensor) / SAMPLE_RATE,
    }


# ============== 7. 保存音频 ==============
def save_audio_file(audio_tensor, filename="recording.wav"):
    """保存音频文件"""
    try:
        import soundfile as sf
        audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
        sf.write(filename, audio_np, SAMPLE_RATE)
        print(f"✓ 音频已保存: {filename}")
    except:
        pass


# ============== 8. 主程序 ==============
def main():
    # 录音
    raw_audio = capture_audio()
    
    # VAD 处理
    print("\n" + "=" * 50)
    print("【Silero VAD 语音检测】")
    print("=" * 50)
    speech_audio, timestamps = apply_silero_vad(raw_audio)
    
    if len(timestamps) == 0:
        print("未检测到语音，请重试！")
        return
    
    save_audio_file(speech_audio)
    
    # SenseVoice 识别
    print("\n" + "=" * 50)
    print("【SenseVoice 语音识别 + 情感分析】")
    print("=" * 50)
    
    result = recognize_with_sensevoice(speech_audio)
    print( f"✓ 识别完成！:{result}")
    print(f"\n📝 识别文本: {result['text']}")
    print(f"🌐 识别语言: {result['language']}")
    print(f"🎭 语音情感: {result['emotion']}")
    print(f"🔊 语音事件: {result['event']}")
    
    # 音频特征
    features = analyze_audio_features(speech_audio)
    print(f"\n📊 音量: {features['loudness_db']:.1f} dB | 时长: {features['duration']:.2f}s")
    
    # 文本情感分析（补充）
    if result['text']:
        print("\n" + "-" * 30)
        print("【文本情感分析 (SnowNLP)】")
        text_sent = analyze_text_sentiment(result['text'])
        if text_sent:
            print(f"   得分: {text_sent['score']:.2%} → {text_sent['sentiment']}")
    
    # 最终总结
    print("\n" + "=" * 50)
    print("📊 分析总结")
    print("=" * 50)
    print(f"  📝 内容: {result['text']}")
    print(f"  🌐 语言: {result['language']}")
    print(f"  🎭 情感: {result['emotion']}")
    print(f"  🔊 事件: {result['event']}")
    
    # 综合判断
    print("\n💡 综合判断:")
    emotion_raw = result.get('emotion_raw', 'NEUTRAL')
    loudness = features['loudness_db']
    
    if emotion_raw == 'ANGRY' and loudness > -20:
        print("   → 🔥 大声生气，情绪激动！")
    elif emotion_raw == 'HAPPY' and loudness > -25:
        print("   → 🎉 兴奋开心，情绪高涨！")
    elif emotion_raw == 'SAD' and loudness < -30:
        print("   → 💧 低声悲伤，情绪低落")
    elif emotion_raw == 'SURPRISE':
        print("   → ❗ 语气惊讶，可能有意外发现")
    else:
        print("   → 😌 情绪平稳")


if __name__ == "__main__":
    main()
