"""
语音情感识别程序 v5.0 - 命令行版本
支持切换两种模式：
  1. paraformer - Paraformer(ASR) + SpeechBrain(情感) 组合
  2. sensevoice - SenseVoice 一站式（ASR + 情感 + 事件检测）

使用方法：
  python emotion_recognition_cli.py                  # 交互式选择
  python emotion_recognition_cli.py -m paraformer   # 直接使用 Paraformer
  python emotion_recognition_cli.py -m sensevoice   # 直接使用 SenseVoice
  python emotion_recognition_cli.py -m both         # 两个都测试对比
"""

import os
import sys
import argparse

# ============== 配置 ==============
FFMPEG_PATH = r"E:\Python\ffmpeg\bin"
if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

SAMPLE_RATE = 16000
RECORD_SECONDS = 8
CHUNK = 512

# ============== 导入库 ==============
import pyaudio
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")


# ============== 全局变量 ==============
vad_model = None
vad_utils = None
paraformer_model = None
sensevoice_model = None
emotion_classifier = None


# ============== 模型加载函数 ==============
def load_silero_vad():
    """加载 Silero VAD"""
    global vad_model, vad_utils
    
    if vad_model is not None:
        return
    
    print("\n[VAD] 加载 Silero VAD...")
    
    # 本地路径
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero-vad")
    
    if os.path.exists(local_path):
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir=local_path,
            model='silero_vad',
            source='local',
            onnx=False
        )
        print(f"      ✓ 本地加载")
    else:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        print(f"      ✓ 网络加载")


def load_paraformer():
    """加载 Paraformer + SpeechBrain"""
    global paraformer_model, emotion_classifier
    
    if paraformer_model is not None:
        return
    
    print("\n[ASR] 加载 Paraformer...")
    from funasr import AutoModel
    
    paraformer_model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device="cpu"
    )
    print("      ✓ Paraformer 加载完成")
    
    # 加载情感模型
    if emotion_classifier is None:
        print("\n[EMO] 加载 SpeechBrain 情感模型...")
        sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = None
        from speechbrain.inference.interfaces import foreign_class
        
        emotion_classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
            run_opts={"device": "cpu"}
        )
        print("      ✓ SpeechBrain 情感模型加载完成")


def load_sensevoice():
    """加载 SenseVoice"""
    global sensevoice_model
    
    if sensevoice_model is not None:
        return
    
    print("\n[ASR+EMO] 加载 SenseVoice...")
    from funasr import AutoModel
    
    # 尝试多种加载方式
    load_methods = [
        {"model": "iic/SenseVoiceSmall", "trust_remote_code": True},
        {"model": "FunAudioLLM/SenseVoiceSmall", "trust_remote_code": True},
        {"model": "iic/SenseVoiceSmall", "trust_remote_code": True, "model_revision": "master"},
    ]
    
    for i, kwargs in enumerate(load_methods):
        try:
            sensevoice_model = AutoModel(**kwargs, device="cpu")
            print(f"      ✓ SenseVoice 加载完成 (方法 {i+1})")
            return
        except Exception as e:
            print(f"      方法 {i+1} 失败: {e}")
    
    raise RuntimeError("SenseVoice 加载失败，请检查 funasr 版本")


# ============== 音频处理函数 ==============
def capture_audio(duration=RECORD_SECONDS):
    """录音"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print(f"\n🎤 开始录音 ({duration}秒)...")
    frames = []
    total_chunks = int(SAMPLE_RATE / CHUNK * duration)
    
    for i in range(total_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
        
        # 进度条
        progress = int((i + 1) / total_chunks * 30)
        bar = "█" * progress + "░" * (30 - progress)
        print(f"\r   [{bar}] {(i+1)*CHUNK/SAMPLE_RATE:.1f}s", end="")
    
    print(f"\r   ✓ 录音完成 ({duration}s)                    ")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio = np.hstack(frames).astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


def apply_vad(audio_tensor):
    """VAD 处理"""
    get_speech_timestamps = vad_utils[0]
    collect_chunks = vad_utils[4]
    
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    )
    
    if not timestamps:
        print("   ⚠️ 未检测到语音")
        return audio_tensor, []
    
    speech_audio = collect_chunks(timestamps, audio_tensor)
    duration = sum(t['end'] - t['start'] for t in timestamps) / SAMPLE_RATE
    print(f"   ✓ VAD: {len(timestamps)} 段语音, 共 {duration:.2f}s")
    
    return speech_audio, timestamps


def save_audio(audio_tensor, filename="recording.wav"):
    """保存音频"""
    try:
        import soundfile as sf
        audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
        sf.write(filename, audio_np, SAMPLE_RATE)
        print(f"   ✓ 已保存: {filename}")
    except Exception as e:
        print(f"   ⚠️ 保存失败: {e}")


# ============== Paraformer 模式 ==============
def run_paraformer(audio_tensor):
    """Paraformer + SpeechBrain 模式"""
    print("\n" + "=" * 50)
    print("【模式: Paraformer + SpeechBrain】")
    print("=" * 50)
    
    # ASR 转录
    print("\n📝 语音识别 (Paraformer)...")
    audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
    result = paraformer_model.generate(input=audio_np, batch_size_s=300)
    
    transcript = ""
    if result and len(result) > 0:
        transcript = result[0].get('text', '')
    
    print(f"   转录: {transcript}")
    
    # 情感分析
    print("\n🎭 情感分析 (SpeechBrain)...")
    emotion_labels = ["中性 😐", "生气 😠", "开心 😊", "悲伤 😢"]
    
    audio_input = audio_tensor.unsqueeze(0) if audio_tensor.dim() == 1 else audio_tensor
    prob, score, index, emotion = emotion_classifier.classify_batch(audio_input)
    
    pred_idx = index[0].item()
    pred_score = score[0].item()
    
    print(f"   情绪: {emotion_labels[pred_idx]} ({emotion[0]})")
    print(f"   置信度: {pred_score:.2%}")
    
    print("\n   概率分布:")
    for i, label in enumerate(emotion_labels):
        bar_len = int(prob[0][i] * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"   {label[:5]:6} [{bar}] {prob[0][i]:.1%}")
    
    # 文本情感
    text_sentiment = None
    if transcript:
        try:
            from snownlp import SnowNLP
            s = SnowNLP(transcript)
            text_sentiment = {
                'score': s.sentiments,
                'label': "正面" if s.sentiments > 0.6 else "负面" if s.sentiments < 0.4 else "中性"
            }
            print(f"\n📊 文本情感 (SnowNLP): {text_sentiment['score']:.1%} → {text_sentiment['label']}")
        except:
            pass
    
    return {
        'mode': 'Paraformer + SpeechBrain',
        'transcript': transcript,
        'emotion': emotion_labels[pred_idx],
        'emotion_raw': emotion[0],
        'confidence': pred_score,
        'text_sentiment': text_sentiment
    }


# ============== SenseVoice 模式 ==============
def run_sensevoice(audio_tensor):
    """SenseVoice 一站式模式"""
    print("\n" + "=" * 50)
    print("【模式: SenseVoice 一站式】")
    print("=" * 50)
    
    # SenseVoice 识别
    print("\n📝🎭 语音识别 + 情感分析 (SenseVoice)...")
    audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
    
    result = sensevoice_model.generate(
        input=audio_np,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60
    )
    
    if not result or len(result) == 0:
        print("   ⚠️ 识别失败")
        return None
    
    raw_text = result[0].get('text', '')
    
    # 解析输出: <|lang|><|emotion|><|event|>text
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
    
    # 语言映射
    lang_map = {'zh': '中文', 'en': '英文', 'ja': '日语', 'ko': '韩语', 'yue': '粤语'}
    
    # 事件映射
    event_map = {
        'Speech': '语音',
        'Laughter': '笑声 😄',
        'Applause': '掌声 👏',
        'Cry': '哭泣 😭',
        'Music': '音乐 🎵',
    }
    
    # 提取标签
    pattern = r'<\|([^|]+)\|>'
    tags = re.findall(pattern, raw_text)
    clean_text = re.sub(pattern, '', raw_text).strip()
    
    language = tags[0] if len(tags) > 0 else 'unknown'
    emotion_raw = tags[1] if len(tags) > 1 else 'NEUTRAL'
    event_raw = tags[2] if len(tags) > 2 else 'Speech'
    
    emotion = emotion_map.get(emotion_raw, emotion_raw)
    event = event_map.get(event_raw, event_raw)
    lang = lang_map.get(language, language)
    
    print(f"   转录: {clean_text}")
    print(f"   语言: {lang}")
    print(f"   情绪: {emotion}")
    print(f"   事件: {event}")
    print(f"   原始: {raw_text}")
    
    # 文本情感
    text_sentiment = None
    if clean_text:
        try:
            from snownlp import SnowNLP
            s = SnowNLP(clean_text)
            text_sentiment = {
                'score': s.sentiments,
                'label': "正面" if s.sentiments > 0.6 else "负面" if s.sentiments < 0.4 else "中性"
            }
            print(f"\n📊 文本情感 (SnowNLP): {text_sentiment['score']:.1%} → {text_sentiment['label']}")
        except:
            pass
    
    return {
        'mode': 'SenseVoice',
        'transcript': clean_text,
        'language': lang,
        'emotion': emotion,
        'emotion_raw': emotion_raw,
        'event': event,
        'raw_output': raw_text,
        'text_sentiment': text_sentiment
    }


# ============== 对比模式 ==============
def run_both(audio_tensor):
    """同时运行两种模式进行对比"""
    print("\n" + "▓" * 50)
    print("【对比模式: 同时测试两种方案】")
    print("▓" * 50)
    
    results = {}
    
    # Paraformer 模式
    try:
        load_paraformer()
        results['paraformer'] = run_paraformer(audio_tensor)
    except Exception as e:
        print(f"\n❌ Paraformer 失败: {e}")
        results['paraformer'] = None
    
    # SenseVoice 模式
    try:
        load_sensevoice()
        results['sensevoice'] = run_sensevoice(audio_tensor)
    except Exception as e:
        print(f"\n❌ SenseVoice 失败: {e}")
        results['sensevoice'] = None
    
    # 对比总结
    print("\n" + "=" * 50)
    print("📊 对比总结")
    print("=" * 50)
    
    print("\n{:<20} {:<25} {:<25}".format("项目", "Paraformer+SpeechBrain", "SenseVoice"))
    print("-" * 70)
    
    p = results.get('paraformer') or {}
    s = results.get('sensevoice') or {}
    
    print("{:<20} {:<25} {:<25}".format(
        "转录文本",
        (p.get('transcript', '-') or '-')[:20],
        (s.get('transcript', '-') or '-')[:20]
    ))
    print("{:<20} {:<25} {:<25}".format(
        "情绪判断",
        p.get('emotion', '-'),
        s.get('emotion', '-')
    ))
    print("{:<20} {:<25} {:<25}".format(
        "置信度/详情",
        f"{p.get('confidence', 0):.1%}" if p.get('confidence') else '-',
        s.get('event', '-')
    ))
    
    return results


# ============== 主程序 ==============
def main():
    parser = argparse.ArgumentParser(
        description='语音情感识别 - 支持多模型切换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python emotion_recognition_cli.py                 # 交互式选择
  python emotion_recognition_cli.py -m paraformer  # Paraformer + SpeechBrain
  python emotion_recognition_cli.py -m sensevoice  # SenseVoice 一站式
  python emotion_recognition_cli.py -m both        # 两种都测试对比
  python emotion_recognition_cli.py -d 10          # 录音 10 秒
        """
    )
    parser.add_argument('-m', '--mode', choices=['paraformer', 'sensevoice', 'both'],
                        help='选择模型模式')
    parser.add_argument('-d', '--duration', type=int, default=RECORD_SECONDS,
                        help=f'录音时长(秒), 默认 {RECORD_SECONDS}')
    parser.add_argument('-f', '--file', type=str,
                        help='直接分析音频文件(跳过录音)')
    
    args = parser.parse_args()
    
    # 交互式选择模式
    if args.mode is None:
        print("=" * 50)
        print("语音情感识别 v5.0")
        print("=" * 50)
        print("\n选择模型模式:")
        print("  1. Paraformer + SpeechBrain (中文识别准，情感分开)")
        print("  2. SenseVoice 一站式 (语音+情感+事件检测)")
        print("  3. 两种都测试对比")
        print()
        
        while True:
            choice = input("请输入选择 (1/2/3): ").strip()
            if choice == '1':
                args.mode = 'paraformer'
                break
            elif choice == '2':
                args.mode = 'sensevoice'
                break
            elif choice == '3':
                args.mode = 'both'
                break
            else:
                print("无效选择，请输入 1, 2 或 3")
    
    # 加载 VAD
    load_silero_vad()
    
    # 加载对应模型
    if args.mode == 'paraformer':
        load_paraformer()
    elif args.mode == 'sensevoice':
        load_sensevoice()
    # both 模式在运行时加载
    
    print("\n" + "=" * 50)
    print(f"✅ 模型加载完成! 模式: {args.mode}")
    print("=" * 50)
    
    # 主循环
    while True:
        try:
            input("\n按 Enter 开始录音 (Ctrl+C 退出)...")
            
            # 获取音频
            if args.file:
                print(f"\n📂 加载文件: {args.file}")
                import soundfile as sf
                audio_np, sr = sf.read(args.file)
                if sr != SAMPLE_RATE:
                    import librosa
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=SAMPLE_RATE)
                audio = torch.from_numpy(audio_np.astype(np.float32))
            else:
                audio = capture_audio(args.duration)
            
            # VAD 处理
            print("\n🔍 VAD 语音检测...")
            speech_audio, timestamps = apply_vad(audio)
            
            if not timestamps:
                print("⚠️ 未检测到有效语音，请重试")
                continue
            
            # 保存音频
            save_audio(speech_audio)
            
            # 运行分析
            if args.mode == 'paraformer':
                result = run_paraformer(speech_audio)
            elif args.mode == 'sensevoice':
                result = run_sensevoice(speech_audio)
            else:
                result = run_both(speech_audio)
            
            # 询问是否继续
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 程序退出")
            break


if __name__ == "__main__":
    main()
