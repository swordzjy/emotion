# 服务端部署说明

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
