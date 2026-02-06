"""
语音情感识别程序 v2.0
使用 Silero VAD（轻量、准确）+ Whisper（转录效果更好）
"""

import pyaudio
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# ============== 配置参数 ==============
SAMPLE_RATE = 16000
RECORD_SECONDS = 8
CHUNK = 512  # Silero VAD 推荐的块大小
WHISPER_MODEL = "base"  # tiny, base, small, medium, large

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

# 1.2 加载 Whisper
print("\n[2/3] 加载 Whisper ASR...")
import whisper
model_path = "./whisper_models/base.pt"

whisper_model = whisper.load_model(model_path)
print(f"✓ Whisper ({model_path}) 加载完成")

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
    
    # 归一化到 [-1, 1]
    audio = np.hstack(frames).astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


# ============== 3. Silero VAD 处理 ==============
def apply_silero_vad(audio_tensor, sample_rate=SAMPLE_RATE):
    """
    使用 Silero VAD 提取有效语音段
    返回: 处理后的音频 tensor, 语音时间戳列表
    """
    # 确保是 1D tensor
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    # 获取语音时间戳
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sample_rate,
        threshold=0.5,           # 语音检测阈值
        min_speech_duration_ms=250,  # 最短语音段 250ms
        min_silence_duration_ms=100,  # 最短静音段 100ms
        speech_pad_ms=30         # 前后填充
    )
    
    if not speech_timestamps:
        print("⚠️ 未检测到有效语音！")
        return audio_tensor, []
    
    # 合并语音段
    speech_audio = collect_chunks(speech_timestamps, audio_tensor)
    
    # 打印 VAD 结果
    total_speech = sum(ts['end'] - ts['start'] for ts in speech_timestamps) / sample_rate
    print(f"✓ VAD 检测到 {len(speech_timestamps)} 个语音段")
    print(f"  有效语音时长: {total_speech:.2f} 秒")
    
    for i, ts in enumerate(speech_timestamps):
        start_sec = ts['start'] / sample_rate
        end_sec = ts['end'] / sample_rate
        print(f"  段 {i+1}: {start_sec:.2f}s - {end_sec:.2f}s")
    
    return speech_audio, speech_timestamps


# ============== 4. Whisper 转录 ==============
def transcribe_with_whisper(audio_tensor, sample_rate=SAMPLE_RATE):
    """
    使用 Whisper 进行语音转录
    自动检测语言
    """
    # Whisper 需要 numpy 数组
    audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
    
    # 确保是 float32
    audio_np = audio_np.astype(np.float32)
    
    # Whisper 需要 16kHz，如果不是则重采样
    if sample_rate != 16000:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
    
    # 转录
    result = whisper_model.transcribe(
        audio_np,
        language=None,  # 自动检测语言
        task="transcribe",
        fp16=False  # CPU 使用 fp32
    )
    
    return result


# ============== 5. 情感分析 ==============
def analyze_emotion(audio_tensor):
    """分析语音情感"""
    # 确保格式正确 [1, time]
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # 推理
    prob, score, index, emotion = emotion_classifier.classify_batch(audio_tensor)
    
    return {
        'probs': prob[0],
        'score': score[0].item(),
        'index': index[0].item(),
        'label': emotion[0]
    }


def analyze_audio_features(audio_tensor):
    """分析音频特征（音量、能量等）"""
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    loudness_db = 20 * torch.log10(rms + 1e-8)
    
    # 计算过零率（语音活跃度指标）
    zero_crossings = torch.sum(torch.abs(torch.diff(torch.sign(audio_tensor)))) / 2
    zcr = zero_crossings / len(audio_tensor)
    
    return {
        'rms': rms.item(),
        'loudness_db': loudness_db.item(),
        'duration': len(audio_tensor) / SAMPLE_RATE,
        'zero_crossing_rate': zcr.item()
    }


# ============== 6. 文本情感分析 ==============
def analyze_text_sentiment(text):
    """分析文本情感"""
    results = {}
    
    # SnowNLP (中文)
    try:
        from snownlp import SnowNLP
        s = SnowNLP(text)
        results['snownlp'] = {
            'score': s.sentiments,
            'sentiment': "正面" if s.sentiments > 0.6 else "负面" if s.sentiments < 0.4 else "中性"
        }
    except ImportError:
        results['snownlp'] = None
    
    # TextBlob (英文)
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        results['textblob'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment': "正面" if blob.sentiment.polarity > 0.1 else "负面" if blob.sentiment.polarity < -0.1 else "中性"
        }
    except ImportError:
        results['textblob'] = None
    
    return results


# ============== 7. 保存音频 ==============
def save_audio_file(audio_tensor, filename="last_recording.wav", sample_rate=SAMPLE_RATE):
    """保存音频文件"""
    audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
    
    try:
        import soundfile as sf
        sf.write(filename, audio_np, sample_rate)
        print(f"✓ 音频已保存: {filename}")
        return True
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")
        return False


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
    
    # 保存处理后的音频
    save_audio_file(speech_audio, "vad_processed.wav")
    save_audio_file(raw_audio, "raw_recording.wav")
    
    # Whisper 转录
    print("\n" + "=" * 50)
    print("【Whisper 语音转录】")
    print("=" * 50)
    whisper_result = transcribe_with_whisper(speech_audio)
    transcript = whisper_result['text'].strip()
    detected_language = whisper_result.get('language', 'unknown')
    
    print(f"检测语言: {detected_language}")
    print(f"转录文本: {transcript}")
    
    # 情感分析（基于声音）
    print("\n" + "=" * 50)
    print("【语音情感分析】")
    print("=" * 50)
    
    emotion_result = analyze_emotion(speech_audio)
    audio_features = analyze_audio_features(speech_audio)
    
    print(f"\n预测情绪: {emotion_labels[emotion_result['index']]} ({emotion_result['label']})")
    print(f"置信度: {emotion_result['score']:.4f}")
    
    print("\n各类别概率：")
    for i, label in enumerate(emotion_labels):
        print(f"  {label:12} : {emotion_result['probs'][i]:.4%}")
    
    # 音频特征
    print("\n" + "-" * 30)
    print("【音频特征】")
    print(f"  音量水平: {audio_features['loudness_db']:.2f} dB")
    print(f"  音频长度: {audio_features['duration']:.2f} 秒")
    print(f"  RMS 能量: {audio_features['rms']:.6f}")
    print(f"  过零率: {audio_features['zero_crossing_rate']:.6f}")
    
    # 综合判断
    emotion_label = emotion_result['label']
    loudness = audio_features['loudness_db']
    
    print("\n【综合分析】")
    if emotion_label == 'ang' and loudness > -20:
        print("→ 大声生气！可能在争吵")
    elif emotion_label == 'hap' and loudness > -25:
        print("→ 兴奋开心，大声表达")
    elif emotion_label == 'sad' and loudness < -35:
        print("→ 低声伤心，很低落")
    else:
        print("→ 情绪正常音量")
    
    max_prob = emotion_result['probs'].max().item()
    if max_prob > 0.8:
        print("→ 情绪判断很明确")
    elif max_prob > 0.6:
        print("→ 情绪比较明显，但有一定不确定性")
    else:
        print("→ 情绪较模糊，可能为中性或混合情绪")
    
    # 文本情感分析
    if transcript:
        print("\n" + "=" * 50)
        print("【文本情感分析】")
        print("=" * 50)
        
        text_sentiment = analyze_text_sentiment(transcript)
        
        if text_sentiment.get('snownlp'):
            print(f"\nSnowNLP (中文):")
            print(f"  情感得分: {text_sentiment['snownlp']['score']:.4f}")
            print(f"  判断结果: {text_sentiment['snownlp']['sentiment']}")
        
        if text_sentiment.get('textblob'):
            print(f"\nTextBlob (英文):")
            print(f"  极性: {text_sentiment['textblob']['polarity']:.4f}")
            print(f"  主观性: {text_sentiment['textblob']['subjectivity']:.4f}")
            print(f"  判断结果: {text_sentiment['textblob']['sentiment']}")
    
    # 最终总结
    print("\n" + "=" * 50)
    print("【分析总结】")
    print("=" * 50)
    print(f"  转录内容: {transcript}")
    print(f"  语音情绪: {emotion_labels[emotion_result['index']]}")
    print(f"  语音置信度: {emotion_result['score']:.2%}")
    if transcript and text_sentiment.get('snownlp'):
        print(f"  文本情感: {text_sentiment['snownlp']['sentiment']} ({text_sentiment['snownlp']['score']:.2%})")
    print("=" * 50)


if __name__ == "__main__":
    main()
