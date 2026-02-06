"""
语音情感识别程序 v4.0
使用 Silero VAD + FunASR Paraformer（中文识别效果最佳）

FunASR 是阿里达摩院开源的语音识别工具包，Paraformer 模型在中文上效果远超 Whisper
"""

import pyaudio
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# ============== 配置参数 ==============
SAMPLE_RATE = 16000
RECORD_SECONDS = 8
CHUNK = 512

# FunASR 模型选项:
# - "paraformer-zh"          : 中文，效果最好，推荐
# - "paraformer-zh-streaming": 中文流式
# - "sensevoice-small"       : 多语言 + 情感识别
ASR_MODEL = "paraformer-zh"

# ============== 1. 加载模型 ==============
print("=" * 50)
print("正在加载模型...")
print("=" * 50)

# 1.1 加载 Silero VAD
print("\n[1/3] 加载 Silero VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("✓ Silero VAD 加载完成")

# 1.2 加载 FunASR
print("\n[2/3] 加载 FunASR ASR 模型...")
try:
    from funasr import AutoModel
    
    # Paraformer-zh: 中文效果最好的开源模型
    asr_model = AutoModel(
        model="paraformer-zh",  # 或 "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        vad_model="fsmn-vad",   # FunASR 自带的 VAD（可选，我们已有 Silero）
        punc_model="ct-punc",   # 标点恢复模型
        device="cpu"
    )
    print(f"✓ FunASR Paraformer 加载完成")
    ASR_TYPE = "funasr"
    
except ImportError:
    print("⚠️ FunASR 未安装，尝试使用 faster-whisper...")
    try:
        from faster_whisper import WhisperModel
        asr_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        print("✓ Faster-Whisper large-v3 加载完成")
        ASR_TYPE = "whisper"
    except:
        print("❌ 请安装 FunASR: pip install funasr modelscope")
        exit(1)

# 1.3 加载情感识别模型
print("\n[3/3] 加载情感识别模型...")
import sys
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = None

from speechbrain.inference.interfaces import foreign_class

emotion_classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
    run_opts={"device": "cpu"}
)
print("✓ 情感识别模型加载完成")

print("\n" + "=" * 50)
print("所有模型加载完成！")
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
    for _ in range(0, int(rate / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    
    print("✓ 录音结束")
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


# ============== 4. ASR 转录 ==============
def transcribe_audio(audio_tensor, sample_rate=SAMPLE_RATE):
    """语音转文字"""
    audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
    
    if ASR_TYPE == "funasr":
        # FunASR Paraformer
        result = asr_model.generate(
            input=audio_np,
            batch_size_s=300,  # 批处理大小（秒）
        )
        
        # 提取文本
        if result and len(result) > 0:
            text = result[0].get('text', '')
            return {
                'text': text,
                'language': 'zh',
                'model': 'FunASR Paraformer'
            }
        return {'text': '', 'language': 'zh', 'model': 'FunASR Paraformer'}
    
    else:
        # Whisper fallback
        segments, info = asr_model.transcribe(audio_np, language="zh")
        text = " ".join([seg.text for seg in segments])
        return {
            'text': text.strip(),
            'language': info.language,
            'model': 'Whisper large-v3'
        }


# ============== 5. 情感分析 ==============
def analyze_emotion(audio_tensor):
    """分析语音情感"""
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    prob, score, index, emotion = emotion_classifier.classify_batch(audio_tensor)
    
    return {
        'probs': prob[0],
        'score': score[0].item(),
        'index': index[0].item(),
        'label': emotion[0]
    }


def analyze_audio_features(audio_tensor):
    """分析音频特征"""
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    loudness_db = 20 * torch.log10(rms + 1e-8)
    
    return {
        'rms': rms.item(),
        'loudness_db': loudness_db.item(),
        'duration': len(audio_tensor) / SAMPLE_RATE,
    }


# ============== 6. 文本情感分析 ==============
def analyze_text_sentiment(text):
    """分析文本情感"""
    results = {}
    
    try:
        from snownlp import SnowNLP
        s = SnowNLP(text)
        results['snownlp'] = {
            'score': s.sentiments,
            'sentiment': "正面" if s.sentiments > 0.6 else "负面" if s.sentiments < 0.4 else "中性"
        }
    except ImportError:
        pass
    
    return results


# ============== 7. 保存音频 ==============
def save_audio_file(audio_tensor, filename="recording.wav", sample_rate=SAMPLE_RATE):
    """保存音频文件"""
    audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
    try:
        import soundfile as sf
        sf.write(filename, audio_np, sample_rate)
        print(f"✓ 音频已保存: {filename}")
    except:
        pass


# ============== 8. 主程序 ==============
def main():
    emotion_labels = ["neu（中性）", "ang（生气）", "hap（开心）", "sad（悲伤）"]
    
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
    
    save_audio_file(speech_audio, "vad_processed.wav")
    
    # ASR 转录
    print("\n" + "=" * 50)
    print(f"【语音转录 - {ASR_TYPE.upper()}】")
    print("=" * 50)
    
    asr_result = transcribe_audio(speech_audio)
    transcript = asr_result['text'].strip()
    
    print(f"使用模型: {asr_result.get('model', ASR_TYPE)}")
    print(f"识别语言: {asr_result.get('language', 'zh')}")
    print(f"转录文本: {transcript}")
    
    # 情感分析
    print("\n" + "=" * 50)
    print("【语音情感分析】")
    print("=" * 50)
    
    emotion_result = analyze_emotion(speech_audio)
    audio_features = analyze_audio_features(speech_audio)
    
    print(f"\n预测情绪: {emotion_labels[emotion_result['index']]} ({emotion_result['label']})")
    print(f"置信度: {emotion_result['score']:.4f}")
    
    print("\n各类别概率：")
    for i, label in enumerate(emotion_labels):
        bar = "█" * int(emotion_result['probs'][i] * 20)
        print(f"  {label:12} : {bar} {emotion_result['probs'][i]:.2%}")
    
    # 音频特征
    print(f"\n音量: {audio_features['loudness_db']:.1f} dB | 时长: {audio_features['duration']:.2f}s")
    
    # 文本情感
    if transcript:
        print("\n" + "=" * 50)
        print("【文本情感分析】")
        print("=" * 50)
        
        text_sentiment = analyze_text_sentiment(transcript)
        if text_sentiment.get('snownlp'):
            s = text_sentiment['snownlp']
            print(f"SnowNLP: {s['score']:.2%} → {s['sentiment']}")
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 分析总结")
    print("=" * 50)
    print(f"  📝 内容: {transcript}")
    print(f"  🎭 语音情绪: {emotion_labels[emotion_result['index']]}")
    print(f"  📈 置信度: {emotion_result['score']:.0%}")


if __name__ == "__main__":
    main()
