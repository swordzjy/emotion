# 语音情感识别程序 v2.0

## 主要改进

### 1. VAD 升级：简单能量阈值 → Silero VAD

| 对比项 | 旧版（能量阈值） | 新版（Silero VAD） |
|--------|-----------------|-------------------|
| 准确性 | 容易误判噪音为语音 | 基于深度学习，准确率高 |
| 静音处理 | 只能检测整体静音 | 精确到毫秒级语音段 |
| 多语言 | - | 语言无关 |
| 资源占用 | 极低 | 轻量（~1MB 模型） |

### 2. ASR 升级：SpeechBrain AISHELL → Whisper

| 对比项 | SpeechBrain AISHELL | OpenAI Whisper |
|--------|---------------------|----------------|
| 支持语言 | 仅中文 | 99+ 语言 |
| 模型大小 | ~1.2GB | tiny:39M ~ large:1.5G |
| 准确性 | 中文较好 | 多语言均优秀 |
| 自动语言检测 | ❌ | ✅ |
| 标点符号 | ❌ | ✅ |

## 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper
pip install pyaudio soundfile numpy
pip install speechbrain snownlp textblob

# 如果 pyaudio 安装失败 (Windows)
pip install pipwin
pipwin install pyaudio
```

## 使用方法

```bash
python emotion_recognition_v2.py
```

## 代码结构

```
emotion_recognition_v2.py
├── 配置参数
├── 模型加载
│   ├── Silero VAD (torch.hub)
│   ├── Whisper (openai-whisper)
│   └── SpeechBrain 情感识别
├── 核心函数
│   ├── capture_audio()      # 录音
│   ├── apply_silero_vad()   # VAD 处理
│   ├── transcribe_with_whisper()  # 语音转录
│   ├── analyze_emotion()    # 情感分析
│   └── analyze_text_sentiment()   # 文本情感
└── main()                   # 主流程
```

## Whisper 模型选择

| 模型 | 大小 | 速度 | 准确性 | 推荐场景 |
|------|------|------|--------|----------|
| tiny | 39M | 最快 | 一般 | 快速测试 |
| base | 74M | 快 | 较好 | **日常使用** |
| small | 244M | 中等 | 好 | 需要更高准确性 |
| medium | 769M | 慢 | 很好 | 专业场景 |
| large | 1.5G | 很慢 | 最好 | 最高质量需求 |

修改 `WHISPER_MODEL` 变量切换模型：
```python
WHISPER_MODEL = "base"  # 改为 "small", "medium" 等
```

## 输出示例

```
==================================================
【Silero VAD 语音检测】
==================================================
✓ VAD 检测到 2 个语音段
  有效语音时长: 3.45 秒
  段 1: 0.52s - 2.31s
  段 2: 3.10s - 4.88s

==================================================
【Whisper 语音转录】
==================================================
检测语言: zh
转录文本: 今天天气真不错，心情很好。

==================================================
【语音情感分析】
==================================================
预测情绪: hap（开心） (hap)
置信度: 0.7823

各类别概率：
  neu（中性）    : 15.23%
  ang（生气）    : 3.45%
  hap（开心）    : 78.23%
  sad（悲伤）    : 3.09%
```

## 常见问题

### Q: Silero VAD 首次运行慢？
A: 首次会从 GitHub 下载模型（~1MB），之后会缓存。

### Q: Whisper 首次运行慢？
A: 首次会下载模型，base 约 74MB。

### Q: 如何只使用 CPU？
A: 代码默认使用 CPU，无需修改。如有 GPU：
```python
whisper_model = whisper.load_model(WHISPER_MODEL, device="cuda")
```

### Q: 中文转录不准？
A: 可指定语言跳过检测：
```python
result = whisper_model.transcribe(audio, language="zh")
```
