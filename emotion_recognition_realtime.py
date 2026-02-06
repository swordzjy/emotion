"""
语音情感识别 v2.1 - 实时流式版本
特性：
- 实时 VAD 检测，语音结束自动停止
- 无需固定录音时长
- 更好的用户体验
"""

import pyaudio
import numpy as np
import torch
import time
import warnings
warnings.filterwarnings("ignore")

# ============== 配置参数 ==============
SAMPLE_RATE = 16000
CHUNK = 512  # Silero VAD 推荐
MAX_RECORD_SECONDS = 30  # 最大录音时长
SILENCE_THRESHOLD_SEC = 1.5  # 静音多久后停止录音
MIN_SPEECH_SEC = 0.5  # 最短有效语音

WHISPER_MODEL = "base"

# ============== 加载模型 ==============
print("=" * 50)
print("正在加载模型...")

# Silero VAD
print("[1/3] Silero VAD...")
vad_model, utils = torch.hub.load(
    'snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=False
)
(get_speech_timestamps, _, _, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
print("✓ Silero VAD")

# Whisper
print("[2/3] Whisper...")
import whisper
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"✓ Whisper ({WHISPER_MODEL})")

# 情感识别
print("[3/3] 情感识别模型...")
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
print("✓ 情感识别")
print("=" * 50 + "\n")


class RealtimeVADRecorder:
    """实时 VAD 录音器"""
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk = CHUNK
        
    def record_until_silence(self):
        """录音直到检测到语音结束"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("🎤 开始录音（说话后停顿将自动结束）...")
        print("   最长录音: {}秒 | 静音阈值: {}秒".format(
            MAX_RECORD_SECONDS, SILENCE_THRESHOLD_SEC
        ))
        
        audio_chunks = []
        speech_detected = False
        silence_start = None
        start_time = time.time()
        
        vad_iterator.reset_states()
        
        while True:
            # 检查超时
            if time.time() - start_time > MAX_RECORD_SECONDS:
                print("\n⏱️ 达到最大录音时长")
                break
            
            # 读取音频
            data = stream.read(self.chunk, exception_on_overflow=False)
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float)
            
            audio_chunks.append(audio_float)
            
            # VAD 检测
            speech_dict = vad_iterator(audio_tensor)
            
            if speech_dict:
                if 'start' in speech_dict:
                    if not speech_detected:
                        print("   🗣️ 检测到语音开始...")
                    speech_detected = True
                    silence_start = None
                    
                if 'end' in speech_dict:
                    print("   🤫 检测到语音段结束")
                    silence_start = time.time()
            
            # 检查静音时长
            if speech_detected and silence_start:
                silence_duration = time.time() - silence_start
                if silence_duration >= SILENCE_THRESHOLD_SEC:
                    print(f"\n✓ 静音 {SILENCE_THRESHOLD_SEC}s，停止录音")
                    break
            
            # 显示进度
            elapsed = time.time() - start_time
            status = "🗣️" if speech_detected and not silence_start else "⏳"
            print(f"\r   {status} 录音中: {elapsed:.1f}s", end="", flush=True)
        
        print()
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 合并音频
        if not audio_chunks:
            return None
        
        audio = np.hstack(audio_chunks)
        total_duration = len(audio) / self.sample_rate
        print(f"✓ 录音完成: {total_duration:.2f}秒")
        
        if not speech_detected:
            print("⚠️ 未检测到有效语音")
            return None
        
        return torch.from_numpy(audio)


def process_audio(audio_tensor):
    """处理音频并分析"""
    
    # VAD 提取有效语音
    print("\n" + "-" * 40)
    print("【VAD 处理】")
    timestamps = get_speech_timestamps(
        audio_tensor, vad_model, 
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=250
    )
    
    if not timestamps:
        print("❌ VAD 未提取到有效语音")
        return
    
    speech_audio = collect_chunks(timestamps, audio_tensor)
    speech_duration = len(speech_audio) / SAMPLE_RATE
    print(f"✓ 提取有效语音: {speech_duration:.2f}秒 ({len(timestamps)} 段)")
    
    # 保存音频
    try:
        import soundfile as sf
        sf.write("recording.wav", speech_audio.numpy(), SAMPLE_RATE)
        print("✓ 已保存: recording.wav")
    except:
        pass
    
    # Whisper 转录
    print("\n" + "-" * 40)
    print("【Whisper 转录】")
    result = whisper_model.transcribe(speech_audio.numpy(), fp16=False)
    transcript = result['text'].strip()
    language = result.get('language', 'unknown')
    print(f"语言: {language}")
    print(f"转录: {transcript}")
    
    # 情感分析
    print("\n" + "-" * 40)
    print("【语音情感】")
    
    emotion_labels = ["中性", "生气", "开心", "悲伤"]
    
    prob, score, index, emotion = emotion_classifier.classify_batch(
        speech_audio.unsqueeze(0)
    )
    
    pred_idx = index[0].item()
    pred_score = score[0].item()
    
    print(f"预测: {emotion_labels[pred_idx]} ({emotion[0]}) | 置信度: {pred_score:.2%}")
    print("\n概率分布:")
    for i, label in enumerate(emotion_labels):
        bar = "█" * int(prob[0][i] * 20)
        print(f"  {label}: {bar} {prob[0][i]:.1%}")
    
    # 文本情感
    if transcript:
        print("\n" + "-" * 40)
        print("【文本情感】")
        try:
            from snownlp import SnowNLP
            s = SnowNLP(transcript)
            sent = "正面" if s.sentiments > 0.6 else "负面" if s.sentiments < 0.4 else "中性"
            print(f"SnowNLP: {s.sentiments:.2%} → {sent}")
        except:
            pass
    
    # 总结
    print("\n" + "=" * 40)
    print("📊 分析总结")
    print("=" * 40)
    print(f"  内容: {transcript}")
    print(f"  语音情绪: {emotion_labels[pred_idx]} ({pred_score:.0%})")
    
    return {
        'transcript': transcript,
        'emotion': emotion_labels[pred_idx],
        'confidence': pred_score
    }


def main():
    recorder = RealtimeVADRecorder()
    
    while True:
        print("\n" + "=" * 50)
        input("按 Enter 开始录音（Ctrl+C 退出）...")
        
        audio = recorder.record_until_silence()
        
        if audio is not None and len(audio) > SAMPLE_RATE * MIN_SPEECH_SEC:
            result = process_audio(audio)
        else:
            print("⚠️ 音频太短，请重试")
        
        print("\n" + "-" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序退出")
