# 服务端部署说明

## 模型目录定义

**重要**：并非所有模型都在 `model_cache` 下。各模型实际存储位置如下：

| 模型 | 存储路径 | 归属 | 说明 |
|------|----------|------|------|
| **Paraformer 三件套** (ASR+VAD+标点) | `{MODEL_CACHE_DIR}/models/iic/` 或 `{MODEL_CACHE_DIR}/hub/models/iic/` | MODEL_CACHE_DIR | ModelScope 格式 |
| **SenseVoice** | `{MODEL_CACHE_DIR}/huggingface/hub/` | MODEL_CACHE_DIR | HuggingFace 缓存 |
| **SpeechBrain 情感** | `pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/` | 固定路径 | 含 `wav2vec2-base/` 子目录（离线必需） |
| **Silero VAD** | `项目根/silero-vad/` | 固定路径 | 优先本地；无则从 torch hub 拉取 |

其中 `MODEL_CACHE_DIR` 默认 `pretrained_models/model_cache`，可通过环境变量覆盖。  
SpeechBrain 与 Silero **不在** `model_cache` 下，始终使用上述固定路径。

### 目录结构示例（项目根 = `emotion/`）

```
emotion/
├── silero-vad/                    # Silero VAD（固定）
├── pretrained_models/
│   ├── model_cache/               # MODEL_CACHE_DIR 默认值
│   │   ├── models/iic/            # Paraformer 三件套
│   │   │   ├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
│   │   │   ├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
│   │   │   └── punc_ct-transformer_cn-en-common-vocab471067-large/
│   │   ├── huggingface/hub/       # SenseVoice
│   │   └── torch_hub/             # 下载脚本缓存 Silero 时使用
│   └── emotion-recognition-wav2vec2-IEMOCAP/   # SpeechBrain 情感（固定）
│       ├── model.ckpt
│       ├── wav2vec2.ckpt
│       ├── hyperparams.yaml
│       ├── custom_interface.py
│       ├── label_encoder.txt
│       └── wav2vec2-base/          # 离线必需（含 config.json、pytorch_model.bin）
```
注意：`wav2vec2_checkpoints/` 为训练时的保存目录，推理不需要；只需 `wav2vec2-base/`。

---

## 预下载模型与离线加载

支持提前手动下载所有模型，启动服务时自动从本地加载，无需联网。

### 1. 预下载所有模型

在**有网络**的环境下执行（只需执行一次）：

```bash
# 激活虚拟环境后
python scripts/download_models.py

# 或指定 MODEL_CACHE_DIR 对应的缓存目录（仅影响 Paraformer、SenseVoice）
python scripts/download_models.py --cache-dir /data/emotion_models
```

下载结果按模型写入不同位置：
- **Paraformer、SenseVoice** → `--cache-dir` 指定的目录（默认 `pretrained_models/model_cache`）
- **SpeechBrain 情感** → `pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/`（固定）
- **Silero VAD** → `项目根/silero-vad/`（固定）

### 2. 启动服务时使用本地模型

当 `MODEL_CACHE_DIR` 存在时，Paraformer 与 SenseVoice 会从该目录加载；SpeechBrain 与 Silero 始终从固定路径加载：

```bash
# 默认 MODEL_CACHE_DIR=pretrained_models/model_cache
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 或指定（Ubuntu 常见）
export MODEL_CACHE_DIR=/data/emotion_models
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

未设置时，若 `pretrained_models/model_cache` 存在则自动使用；否则尝试 `/data/emotion_models`、`/data/emotion-models`。

### 3. 完全离线部署

1. 在有网络的机器上运行 `python scripts/download_models.py`
2. 将整个项目目录（含 `pretrained_models/`、`silero-vad/`）拷贝到离线服务器
3. 在离线服务器上启动服务，无需联网

---

## 内存不足（OOM）问题

连接时加载 Paraformer（ASR + VAD + 标点）和 SpeechBrain 情感模型会占用较多内存。若在 **Linux 上被 OOM Killer 终止**（日志中出现 `killed by the OOM killer` 或 `emotion.service: A process of this unit has been killed by the OOM killer`），请按以下方式处理。

### 1. 增加 swap（推荐）

为系统添加 swap，避免物理内存用尽时直接被杀进程：

```bash
# 查看当前 swap
free -h

# 创建 4GB swap 文件（路径与大小可按需修改）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效：写入 fstab
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2. 推荐内存与 swap

| 模式        | 建议物理内存 | 建议 swap |
|-------------|--------------|-----------|
| paraformer  | ≥ 4GB        | ≥ 2GB     |
| sensevoice  | ≥ 4GB        | ≥ 2GB     |
| 两者都使用  | ≥ 6GB        | ≥ 2GB     |

小内存机器（如 2GB）建议至少加 4GB swap，并启用下面的延迟加载。

### 3. 延迟加载情感模型（降低连接时峰值内存）

若连接阶段因内存不足被 OOM，可让 **情感模型在首次分析时再加载**，从而降低“连接”时的内存峰值：

```bash
# 启动前设置环境变量（仅 paraformer 模式生效）
export EMOTION_LAZY_LOAD=1
# 再启动服务，例如：
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

或在 systemd 中配置：

```ini
[Service]
Environment="EMOTION_LAZY_LOAD=1"
ExecStart=/path/to/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

效果：点击“连接服务器”时只加载 VAD + Paraformer，不加载 SpeechBrain；**第一次点击“停止录音”并得到分析结果时**再加载情感模型（该次会稍慢），之后正常。

### 4. 代码层面的优化

- 在加载 Paraformer 与 SpeechBrain 之间已执行 `gc.collect()`，减轻峰值内存。
- 设置 `HF_HUB_OFFLINE=1`（在 `load_for_mode` 中）可避免加载时联网检查模型新版本，减少异常与等待。
- 已内置 ModelScope 离线补丁：检测到本地 `model.pt` 时直接使用，避免联网验证。
- **SpeechBrain 离线**：hyperparams 引用 `facebook/wav2vec2-base` 需联网。`download_models.py --speechbrain` 会下载到 `wav2vec2-base/` 并 patch hyperparams；拷贝整个 `emotion-recognition-wav2vec2-IEMOCAP` 目录（含 `wav2vec2-base`）即可离线使用。
