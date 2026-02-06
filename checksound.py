import pyaudio
import numpy as np
import torch

from speechbrain.inference.interfaces import foreign_class  # 这个是关键！
from speechbrain.utils.fetching import LocalStrategy  # 必须导入这个！
# 绕过 Flair Embeddings 的加载问题
import sys
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = None  # 假装已加载，但实际跳过


emotion_classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",          # 模型仓库自带的自定义文件，会自动下载
    classname="CustomEncoderWav2vec2Classifier",  # 自定义类名
    savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
    #local_strategy=LocalStrategy.COPY,  # 如果有权限问题，可加（但 foreign_class 支持有限，建议先不加）
    run_opts={"device": "cpu"}
)
print("情感识别模型加载完成！（如果第一次，会显示下载进度）")

# Function to capture audio from microphone
def capture_audio(duration=8, rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=1024)
    print("正在录音8s...")
    frames = []
    for _ in range(0, int(rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.int16))

    
    print("录音结束.")
    stream.stop_stream()
    stream.close()
    p.terminate()
   
    # 在 capture_audio 返回前加归一化
    audio = np.hstack(frames).astype(np.float32) / 32768.0

# 简单 VAD：去除低能量部分（静音）
    energy = np.abs(audio)
    energy_threshold = 0.015  # 可调节阈值
    mask = energy > energy_threshold
    if np.any(mask):
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        audio = audio[start:end]
        print(f"VAD 裁剪后有效长度: {len(audio)/16000:.2f} 秒")
    if len(audio) < rate:  # 至少 1 秒
        print("警告：有效语音太短！")
  
    return torch.from_numpy(audio).unsqueeze(0)
   

# Capture and analyze
audio_signal = capture_audio()

# 在转录前或后加这段保存调试音频
try:
    # 优先尝试 soundfile（最稳）
    import soundfile as sf
    sf.write("last_recording.wav", audio_signal.squeeze(0).cpu().numpy(), 16000)
    print("音频已保存为 last_recording.wav（使用 soundfile）")
except Exception as e:
    print(f"soundfile 保存失败: {e}")

    # fallback 到 torchaudio + soundfile 后端
    try:
        torchaudio.set_audio_backend("soundfile")
        torchaudio.save("last_recording.wav", audio_signal.cpu(), 16000)
        print("音频已保存为 last_recording.wav（torchaudio + soundfile 后端）")
    except Exception as e:
        print(f"torchaudio 保存也失败: {e}")
        print("请检查 soundfile 是否安装，或手动保存音频用于测试。")

# 录音函数保持不变（确保 audio_signal 是 [1, time] 的 float32 tensor，采样率 16kHz，归一化 [-1,1]）

# 推理（用 classify_batch 或 classify_file）


# 解包 prediction
prob, score, index, emotion = emotion_classifier.classify_batch(audio_signal)
pred_prob = prob[0]          # [4] 的 tensor
pred_index = index[0].item() # 预测的类别索引（整数）
pred_label = emotion[0]     # 字符串标签，如 'neu'

# 假设 IEMOCAP 的类别顺序是这个（最常见顺序，建议确认模型的 label_encoder.txt）
emotion_labels = [ "neu（中性）","ang（生气）", "hap（开心）", "sad（悲伤）"]
print("\n" + "="*40)
# 友好输出
print("\n1.=== 情绪分析结果 基于声音 ===")
print(f"预测情绪: {emotion_labels[pred_index]}  ({pred_label})")
print(f"置信度: {score[0]:.4f}")

print("\n各类别概率：")
for i, label in enumerate(emotion_labels):
    print(f"  {label:12} : {pred_prob[i]:.4%}")


print("\n2.=== 声音能力解析结果 ===")
# 计算 RMS 能量（音量代理）
rms = torch.sqrt(torch.mean(audio_signal**2))  # 简单 RMS
loudness_db = 20 * torch.log10(rms + 1e-8)     # 转分贝

print(f"音量水平: {loudness_db.item():.2f} dB")
print(f"音频长度: {audio_signal.shape[1] / 16000:.2f} 秒")
print(f"RMS 能量: {torch.sqrt(torch.mean(audio_signal**2)):.6f}")
print("\n.======================")

# 结合情绪做判断
if emotion == 'ang' and loudness_db > -20:  # 假设 -20 dB 是大声阈值
    print("大声生气！可能在争吵")
elif emotion == 'hap' and loudness_db > -25:
    print("兴奋开心，大声表达")
elif emotion == 'sad' and loudness_db < -35:
    print("低声伤心，很低落")
else:
    print("情绪正常音量")
# 简单判断建议
max_prob = pred_prob.max().item()
if max_prob > 0.8:
    print("→ 情绪判断很明确")
elif max_prob > 0.6:
    print("→ 情绪比较明显，但有一定不确定性")
else:
    print("→ 情绪较模糊，可能为中性或混合情绪")


### huggingface-cli login token removed for security
# For keyword analysis: Add ASR and NLP
# ASR 部分
from speechbrain.inference import EncoderDecoderASR

print("正在加载 ASR 模型：speechbrain/asr-wav2vec2-transformer-aishell")

asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-transformer-aishell",
    savedir="pretrained_models/asr-wav2vec2-transformer-aishell",
    local_strategy=LocalStrategy.COPY,
    run_opts={"device": "cpu"}
)

print("ASR 模型加载完成！（如果第一次，会下载 ~1.2GB）")

# 转录
transcript = asr_model.transcribe_batch(
    audio_signal,
    wav_lens=torch.ones(audio_signal.shape[0], device=audio_signal.device)
)[0][0]

print(f"转录文本：{transcript}")
print("\n" + "="*40)
print("【语音转录 & 文本情感分析】")
print("\nASR 模型加载完成！")


try:
    from snownlp import SnowNLP
    s = SnowNLP(transcript)
    score = s.sentiments
    sentiment_text = "正面/开心" if score > 0.6 else "负面/生气" if score < 0.4 else "中性/平静"
    print(f"文本情感得分 (SnowNLP)：{score:.4f}  →  {sentiment_text}")
except ImportError:
    print("建议 pip install snownlp 以获得更好的中文文本情感分析")

# TextBlob 作为备用（英文偏好）
from textblob import TextBlob
blob = TextBlob(transcript)
polarity = blob.sentiment.polarity
print(f"TextBlob polarity（英文参考）：{polarity:.4f} （>0 正面，<0 负面）")


print("\n=== 使用ASR语音转录及机器学习分析 分析结束===")