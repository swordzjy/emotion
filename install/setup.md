lan: 客户端-服务器架构重构                                                        │     │                                                                                    │
     │ Context                                                                            │
     │                                                                                    │
     │ 当前项目是一组 CLI Python 脚本，所有 ML                                            │
     │ 推理和麦克风录音都在本地运行。目标是拆分为：                                       │
     │ - server/ — FastAPI 后端，承载所有 ML 模型推理                                     │
     │ - client/ — 纯 HTML/JS 前端，通过 Web Audio API 采集麦克风音频，WebSocket          │
     │ 实时流传到后端                                                                     │
     │                                                                                    │
     │ 以 alipay/emotion_recognition_cli.py（v5.0）为主要参考，支持两种分析模式：         │
     │ - paraformer — Paraformer（中文 ASR）+ SpeechBrain（4                              │
     │ 类情绪：中性/生气/开心/悲伤）                                                      │
     │ - sensevoice — SenseVoice 一站式（ASR + 7 类情绪 + 语音事件检测）                  │
     │                                                                                    │
     │ 现有 CLI 脚本保持不变。                                                            │
     │                                                                                    │
     │ 技术选型                                                                           │
     │ ┌──────┬────────────────────────────────────────────────────────┐                  │
     │ │  层  │                          技术                          │                  │
     │ ├──────┼────────────────────────────────────────────────────────┤                  │
     │ │ 后端 │ FastAPI + uvicorn + WebSocket                          │                  │
     │ ├──────┼────────────────────────────────────────────────────────┤                  │
     │ │ 前端 │ 纯 HTML/JS（Web Audio API）                            │                  │
     │ ├──────┼────────────────────────────────────────────────────────┤                  │
     │ │ 传输 │ WebSocket 实时流（float32 PCM, 16kHz, 512 samples/帧） │                  │
     │ ├──────┼────────────────────────────────────────────────────────┤                  │
     │ │ ASR  │ funasr（Paraformer / SenseVoice），替代 Whisper        │                  │
     │ └──────┴────────────────────────────────────────────────────────┘                  │
     │ 目录结构                                                                           │
     │                                                                                    │
     │ emotion/                                                                           │
     │ ├── server/                                                                        │
     │ │   ├── __init__.py                                                                │
     │ │   ├── app.py               # FastAPI 入口，lifespan 加载模型，静态文件           │
     │ │   ├── config.py            # 配置常量                                            │
     │ │   ├── models.py            # 单例 ModelManager（VAD / Paraformer / SenseVoice /  │
     │ SpeechBrain）                                                                      │
     │ │   ├── pipeline.py          # 两种分析管线：run_paraformer() / run_sensevoice()   │
     │ │   ├── ws_handler.py        # WebSocket 处理，流式 VAD 状态机                     │
     │ │   └── requirements.txt                                                           │
     │ │                                                                                  │
     │ └── client/                                                                        │
     │     ├── index.html                                                                 │
     │     ├── css/                                                                       │
     │     │   └── style.css                                                              │
     │     └── js/                                                                        │
     │         ├── app.js           # 主逻辑                                              │
     │         ├── audio.js         # Web Audio API 麦克风采集 + 16kHz + 512 样本分块     │
     │         ├── websocket.js     # WebSocket 管理                                      │
     │         └── ui.js            # DOM、波形、结果渲染                                 │
     │                                                                                    │
     │ 服务端实现                                                                         │
     │                                                                                    │
     │ server/config.py                                                                   │
     │                                                                                    │
     │ 复用 alipay/emotion_recognition_cli.py 的参数：                                    │
     │ - SAMPLE_RATE=16000, CHUNK_SAMPLES=512                                             │
     │ - VAD：SILENCE_THRESHOLD_SEC=1.5, MAX_RECORD_SECONDS=30, MIN_SPEECH_SEC=0.5        │
     │ - 模型路径指向项目根目录下的 pretrained_models/、silero-vad/                       │
     │                                                                                    │
     │ server/models.py — 单例 ModelManager                                               │
     │                                                                                    │
     │ 从 alipay/emotion_recognition_cli.py 的懒加载函数（44-133 行）提取：               │
     │                                                                                    │
     │ - load_silero_vad()（cli.py:44-71）→ Silero VAD，优先从本地 silero-vad/ 加载       │
     │ - load_paraformer()（cli.py:74-105）→ funasr.AutoModel(model="paraformer-zh",      │
     │ vad_model="fsmn-vad", punc_model="ct-punc") + SpeechBrain 情绪分类器               │
     │ - load_sensevoice()（cli.py:108-133）→                                             │
     │ funasr.AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True)              │
     │ - create_vad_iterator() → 每个 WebSocket 会话创建独立的 VADIterator                │
     │                                                                                    │
     │ 启动时根据配置的默认模式加载，也可全部加载支持运行时切换。                         │
     │                                                                                    │
     │ server/pipeline.py — 两种分析管线                                                  │
     │                                                                                    │
     │ 从 alipay/emotion_recognition_cli.py 提取：                                        │
     │                                                                                    │
     │ run_paraformer_analysis(audio_tensor)（cli.py:211-268）：                          │
     │ 1. VAD 过滤（apply_vad，cli.py:171-196）                                           │
     │ 2. paraformer_model.generate(input=audio_np, batch_size_s=300) → 转录文本          │
     │ 3. emotion_classifier.classify_batch(audio_input) → 4 类情绪概率                   │
     │ 4. 音频特征（RMS、响度 dB）                                                        │
     │ 5. SnowNLP 文本情感                                                                │
     │ 6. 返回 JSON dict                                                                  │
     │                                                                                    │
     │ run_sensevoice_analysis(audio_tensor)（cli.py:272-364）：                          │
     │ 1. VAD 过滤                                                                        │
     │ 2. sensevoice_model.generate(input=audio_np, cache={}, language="auto",            │
     │ use_itn=True) → 原始输出                                                           │
     │ 3. 解析 <|lang|><|emotion|><|event|>text 标签格式（cli.py:296-326）                │
     │   - 情绪映射：HAPPY/SAD/ANGRY/NEUTRAL/SURPRISE/FEARFUL/DISGUSTED（7 类）           │
     │   - 事件映射：Speech/Laughter/Applause/Cry/Music                                   │
     │   - 语言映射：zh/en/ja/ko/yue                                                      │
     │ 4. 音频特征 + SnowNLP 文本情感                                                     │
     │ 5. 返回 JSON dict                                                                  │
     │                                                                                    │
     │ 使用 threading.Lock 保护 VAD 模型的 get_speech_timestamps 调用（避免并发时         │
     │ reset_states 竞争）。                                                              │
     │                                                                                    │
     │ server/ws_handler.py — WebSocket 处理                                              │
     │                                                                                    │
     │ 从 emotion_recognition_realtime.py 的 RealtimeVADRecorder（68-148 行）改编：       │
     │                                                                                    │
     │ - 客户端连接时通过 query param 指定模式：ws://host:8000/ws/audio?mode=sensevoice   │
     │ - 接收二进制帧 → bytearray 缓冲 → 按 2048 字节（512 float32）切块                  │
     │ - 每块送入 VADIterator 做实时语音检测                                              │
     │ - 检测到 1.5s 静默后停止接收，调用对应管线                                         │
     │ - run_in_executor 将 CPU 密集推理放到线程池                                        │
     │ - 向客户端发送状态更新 + 最终结果 JSON                                             │
     │                                                                                    │
     │ server/app.py                                                                      │
     │                                                                                    │
     │ - lifespan 上下文管理器启动时加载模型                                              │
     │ - WebSocket：/ws/audio                                                             │
     │ - 健康检查：GET /health                                                            │
     │ - CORS 中间件                                                                      │
     │ - 挂载 client/ 为静态文件（/client）                                               │
     │                                                                                    │
     │ 客户端实现                                                                         │
     │                                                                                    │
     │ client/js/audio.js                                                                 │
     │                                                                                    │
     │ - AudioCapture 类，new AudioContext({ sampleRate: 16000 }) 自动重采样              │
     │ - ScriptProcessorNode（bufferSize=4096）累积样本，按 512 切块发送                  │
     │ - 波形可视化数据回调                                                               │
     │                                                                                    │
     │ client/js/websocket.js                                                             │
     │                                                                                    │
     │ - WebSocketManager 类                                                              │
     │ - 发送：二进制音频帧 + JSON 停止命令                                               │
     │ - 接收：JSON 状态 / 结果 / 错误                                                    │
     │                                                                                    │
     │ client/js/ui.js                                                                    │
     │                                                                                    │
     │ - 连接状态、录音计时器、波形 Canvas                                                │
     │ - 结果展示：转录文本、情绪标签、事件标签、音频特征、文本情感                       │
     │ - SenseVoice 模式下显示 7 类情绪 + 事件；Paraformer 模式下显示 4 类概率条          │
     │                                                                                    │
     │ client/js/app.js + index.html                                                      │
     │                                                                                    │
     │ - 页面顶部提供模式选择（paraformer / sensevoice）                                  │
     │ - 每次录音创建新的 WebSocket 连接，URL 中带 ?mode=xxx                              │
     │ - 服务端返回 silence_detected 时自动停止录音                                       │
     │                                                                                    │
     │ WebSocket 协议                                                                     │
     │                                                                                    │
     │ 连接：ws://host:8000/ws/audio?mode=sensevoice                                      │
     │                                                                                    │
     │ 客户端 → 服务端：binary（2048 字节 = 512 float32 样本，~32ms/帧）                  │
     │ 客户端 → 服务端：JSON {"type": "stop"}                                             │
     │                                                                                    │
     │ 服务端 → 客户端：JSON {"type": "status", "status":                                 │
     │ "listening|speech_start|speech_end|silence_detected|processing"}                   │
     │                                                                                    │
     │ 服务端 → 客户端（sensevoice 模式）：                                               │
     │ {                                                                                  │
     │   "type": "result",                                                                │
     │   "mode": "sensevoice",                                                            │
     │   "transcript": "今天天气真好",                                                    │
     │   "language": "中文",                                                              │
     │   "emotion": "开心",                                                               │
     │   "emotion_raw": "HAPPY",                                                          │
     │   "event": "语音",                                                                 │
     │   "audio_features": {"loudness_db": -25.3, "duration_sec": 3.45},                  │
     │   "text_sentiment": {"score": 0.82, "label": "正面"}                               │
     │ }                                                                                  │
     │                                                                                    │
     │ 服务端 → 客户端（paraformer 模式）：                                               │
     │ {                                                                                  │
     │   "type": "result",                                                                │
     │   "mode": "paraformer",                                                            │
     │   "transcript": "今天天气真好",                                                    │
     │   "emotion": {"label": "hap", "confidence": 0.78, "probabilities": {"neu": 0.15,   │
     │ "ang": 0.03, "hap": 0.78, "sad": 0.04}},                                           │
     │   "audio_features": {"loudness_db": -25.3, "duration_sec": 3.45},                  │
     │   "text_sentiment": {"score": 0.82, "label": "正面"}                               │
     │ }                                                                                  │
     │                                                                                    │
     │ 关键源文件参考                                                                     │
     │ 文件: alipay/emotion_recognition_cli.py                                            │
     │ 提取内容: 模型加载（44-133行）、apply_vad（171-196行）、run_paraformer（211-268行  │
     │ ）、run_sensevoice（272-364行）                                                    │
     │ ────────────────────────────────────────                                           │
     │ 文件: emotion_recognition_realtime.py                                              │
     │ 提取内容: 流式 VAD 录音循环 RealtimeVADRecorder（68-148行）                        │
     │ ────────────────────────────────────────                                           │
     │ 文件: alipay/emotion_recognition_sensevoice.py                                     │
     │ 提取内容: SenseVoice 输出解析 parse_sensevoice_output（170-230行）                 │
     │ 实现顺序                                                                           │
     │                                                                                    │
     │ 1. server/__init__.py + server/config.py                                           │
     │ 2. server/models.py（从 cli.py 提取模型加载）                                      │
     │ 3. server/pipeline.py（从 cli.py 提取 run_paraformer + run_sensevoice）            │
     │ 4. server/ws_handler.py（从 realtime.py 改编 WebSocket 流式处理）                  │
     │ 5. server/app.py + server/requirements.txt                                         │
     │ 6. client/index.html（含模式选择 UI）                                              │
     │ 7. client/css/style.css                                                            │
     │ 8. client/js/audio.js → websocket.js → ui.js → app.js                              │
     │                                                                                    │
     │ 新增依赖                                                                           │
     │                                                                                    │
     │ # server/requirements.txt                                                          │
     │ fastapi>=0.104.0                                                                   │
     │ uvicorn[standard]>=0.24.0                                                          │
     │ websockets>=12.0                                                                   │
     │ # 以下已在环境中安装                                                               │
     │ funasr                                                                             │
     │ modelscope                                                                         │
     │ torch>=2.0.0                                                                       │
     │ torchaudio>=2.0.0                                                                  │
     │ speechbrain                                                                        │
     │ snownlp                                                                            │
     │ numpy                                                                              │
     │ soundfile                                                                          │
     │                                                                                    │
     │ 验证方式                                                                           │
     │                                                                                    │
     │ 1. 启动：python -m uvicorn server.app:app --host 0.0.0.0 --port 8000               │
     │ 2. 浏览器打开 http://localhost:8000/client/index.html                              │
     │ 3. 选择模式（sensevoice / paraformer），点击录音                                   │
     │ 4. 对麦克风说话，观察：波形实时显示 → Speech detected → Processing → 结果展示      │
     │ 5. 切换模式重新测试，对比两种模式的输出差异                                        │
     │ 6. GET http://localhost:8000/health 确认服务状态     


     
  # Install server deps
  pip install -r server/requirements.txt

  # Start server
  python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

  # Open browser
  # http://localhost:8000/client/index.html
