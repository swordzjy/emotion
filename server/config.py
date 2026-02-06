"""服务端配置常量"""

import os

# 项目根目录（emotion/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 音频参数
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # Silero VAD 推荐的每帧样本数
BYTES_PER_FRAME = CHUNK_SAMPLES * 4  # 512 float32 = 2048 bytes

# VAD 参数
SILENCE_THRESHOLD_SEC = 1.5   # 静音多久后停止接收（已弃用，改为手动停止）
MAX_RECORD_SECONDS = 60       # 最大录音时长（1分钟）
MIN_SPEECH_SEC = 0.5          # 最短有效语音
STREAMING_TRANSCRIBE_INTERVAL = 2.0  # 流式转录间隔（秒）

# VAD 检测参数
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 100

# 模型路径
SILERO_VAD_PATH = os.path.join(PROJECT_ROOT, "silero-vad")
PRETRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")
SPEECHBRAIN_EMOTION_DIR = os.path.join(PRETRAINED_MODELS_DIR, "emotion-recognition-wav2vec2-IEMOCAP")

# ffmpeg（部分模型需要）
FFMPEG_PATH = r"E:\Python\ffmpeg\bin"
if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

# 默认分析模式
DEFAULT_MODE = "sensevoice"

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000
