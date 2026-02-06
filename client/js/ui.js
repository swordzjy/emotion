/**
 * UI — DOM 操作、波形可视化、结果渲染、历史记录管理
 */
const UI = {
  // 缓存 DOM 引用
  els: {},
  waveformBuffer: [],
  maxWaveformPoints: 200,

  init() {
    this.els = {
      connectionStatus: document.getElementById("connection-status"),
      statusText: document.getElementById("status-text"),
      timer: document.getElementById("timer"),
      waveform: document.getElementById("waveform"),
      btnRecord: document.getElementById("btn-record"),
      btnStop: document.getElementById("btn-stop"),
      resultPanel: document.getElementById("result-panel"),
      resultTranscript: document.getElementById("result-transcript"),
      resultLanguage: document.getElementById("result-language"),
      resultEmotion: document.getElementById("result-emotion"),
      eventSection: document.getElementById("event-section"),
      resultEvent: document.getElementById("result-event"),
      resultAudioFeatures: document.getElementById("result-audio-features"),
      sentimentSection: document.getElementById("sentiment-section"),
      resultTextSentiment: document.getElementById("result-text-sentiment"),
      errorPanel: document.getElementById("error-panel"),
      errorMessage: document.getElementById("error-message"),
      btnClearResult: document.getElementById("btn-clear-result"),
      btnDismissError: document.getElementById("btn-dismiss-error"),
      historyList: document.getElementById("history-list"),
      btnClearHistory: document.getElementById("btn-clear-history"),
      liveTranscriptPanel: document.getElementById("live-transcript-panel"),
      liveTranscriptText: document.getElementById("live-transcript-text"),
    };

    // 检查关键元素是否存在
    const missingElements = [];
    const requiredElements = [
      "connection-status", "status-text", "timer", "waveform",
      "btn-record", "btn-stop", "result-panel"
    ];
    
    requiredElements.forEach(id => {
      if (!document.getElementById(id)) {
        missingElements.push(id);
      }
    });
    
    if (missingElements.length > 0) {
      console.error("缺少必需的DOM元素:", missingElements);
      const errorMsg = `页面加载错误：缺少以下元素: ${missingElements.join(", ")}`;
      alert(errorMsg);
      return;
    }

    // 初始化波形Canvas
    if (this.els.waveform) {
      this._waveCtx = this.els.waveform.getContext("2d");
      this._resizeCanvas();
      window.addEventListener("resize", () => this._resizeCanvas());
      this.clearWaveform();
    } else {
      console.error("waveform元素不存在，无法初始化波形显示");
    }

    // 绑定事件（安全地检查元素是否存在）
    if (this.els.btnClearResult) {
      this.els.btnClearResult.addEventListener("click", () => this.hideResults());
    }
    if (this.els.btnDismissError) {
      this.els.btnDismissError.addEventListener("click", () => {
        if (this.els.errorPanel) {
          this.els.errorPanel.classList.add("hidden");
        }
      });
    }
    if (this.els.btnClearHistory) {
      this.els.btnClearHistory.addEventListener("click", () => this.clearHistory());
    }

    // 加载历史记录
    this.loadHistory();
  },

  _resizeCanvas() {
    if (!this.els.waveform || !this._waveCtx) {
      return;
    }
    const c = this.els.waveform;
    const container = c.parentElement;
    if (!container) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    c.width = (container.clientWidth - 32) * dpr; // padding
    c.height = 120 * dpr;
    c.style.width = (container.clientWidth - 32) + "px";
    c.style.height = "120px";
    this._waveCtx.scale(dpr, dpr);
  },

  // ---- 状态 ----
  setStatus(text, dotClass) {
    if (this.els.statusText) {
      this.els.statusText.textContent = text;
    }
    if (this.els.connectionStatus) {
      this.els.connectionStatus.className = "status-dot " + (dotClass || "disconnected");
    }
  },

  setTimer(seconds) {
    if (!this.els.timer) return;
    if (seconds == null) {
      this.els.timer.textContent = "";
    } else {
      const m = Math.floor(seconds / 60);
      const s = Math.floor(seconds % 60);
      this.els.timer.textContent = `${m}:${s.toString().padStart(2, "0")}`;
    }
  },

  setRecordingState(recording) {
    if (!this.els.btnRecord || !this.els.btnStop) return;
    const btn = this.els.btnRecord;
    const btnText = btn.querySelector(".btn-text");
    if (recording) {
      if (btnText) btnText.textContent = "录音中...";
      btn.classList.add("recording");
      if (this.els.btnStop) this.els.btnStop.disabled = false;
    } else {
      if (btnText) btnText.textContent = "开始录音";
      btn.classList.remove("recording");
      btn.disabled = false;
      if (this.els.btnStop) this.els.btnStop.disabled = true;
    }
  },

  // ---- 波形可视化（增强版） ----
  drawWaveform(samples) {
    if (!this.els.waveform || !this._waveCtx) {
      return;
    }
    // 计算RMS用于波形显示
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    const rms = Math.sqrt(sum / samples.length);
    
    // 添加到缓冲区
    this.waveformBuffer.push(rms);
    if (this.waveformBuffer.length > this.maxWaveformPoints) {
      this.waveformBuffer.shift();
    }

    this._drawWaveformFromBuffer();
  },

  _drawWaveformFromBuffer() {
    if (!this.els.waveform || !this._waveCtx) {
      return;
    }
    const ctx = this._waveCtx;
    const w = this.els.waveform.width / (window.devicePixelRatio || 1);
    const h = this.els.waveform.height / (window.devicePixelRatio || 1);

    // 清空画布
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
    ctx.fillRect(0, 0, w, h);

    // 绘制网格线
    ctx.strokeStyle = "rgba(59, 130, 246, 0.1)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (h / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // 绘制波形
    if (this.waveformBuffer.length > 1) {
      const step = w / this.waveformBuffer.length;
      
      // 创建渐变
      const gradient = ctx.createLinearGradient(0, 0, 0, h);
      gradient.addColorStop(0, "#3b82f6");
      gradient.addColorStop(0.5, "#8b5cf6");
      gradient.addColorStop(1, "#ec4899");

      ctx.strokeStyle = gradient;
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();

      for (let i = 0; i < this.waveformBuffer.length; i++) {
        const x = i * step;
        const value = Math.min(this.waveformBuffer[i] * 5, 1); // 放大显示
        const y = h / 2 - (value * h / 2);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // 填充区域
      ctx.lineTo(w, h / 2);
      ctx.lineTo(0, h / 2);
      ctx.closePath();
      const fillGradient = ctx.createLinearGradient(0, 0, 0, h);
      fillGradient.addColorStop(0, "rgba(59, 130, 246, 0.3)");
      fillGradient.addColorStop(1, "rgba(139, 92, 246, 0.1)");
      ctx.fillStyle = fillGradient;
      ctx.fill();
    }

    // 绘制中线
    ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
  },

  clearWaveform() {
    if (!this.els.waveform || !this._waveCtx) {
      return;
    }
    this.waveformBuffer = [];
    const ctx = this._waveCtx;
    const w = this.els.waveform.width / (window.devicePixelRatio || 1);
    const h = this.els.waveform.height / (window.devicePixelRatio || 1);
    
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
    ctx.fillRect(0, 0, w, h);

    // 绘制中线
    ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
  },

  // ---- 结果展示 ----
  showResult(data) {
    if (!this.els.resultPanel) {
      console.error("resultPanel元素不存在");
      return;
    }
    if (this.els.errorPanel) {
      this.els.errorPanel.classList.add("hidden");
    }
    this.els.resultPanel.classList.remove("hidden");

    // 转录
    if (this.els.resultTranscript) {
      this.els.resultTranscript.textContent = data.transcript || "(无文本)";
    }

    // 语言
    if (this.els.resultLanguage) {
      if (data.language) {
        this.els.resultLanguage.textContent = `语言: ${data.language}`;
        this.els.resultLanguage.classList.remove("hidden");
      } else {
        this.els.resultLanguage.classList.add("hidden");
      }
    }

    // 情感
    if (this.els.resultEmotion) {
      if (data.mode === "sensevoice") {
        this._renderSenseVoiceEmotion(data);
      } else {
        this._renderParaformerEmotion(data);
      }
    }

    // 事件
    if (this.els.eventSection && this.els.resultEvent) {
      if (data.event) {
        this.els.eventSection.classList.remove("hidden");
        const eventEmoji = this._getEventEmoji(data.event);
        this.els.resultEvent.innerHTML = `
          <span class="event-tag">
            ${eventEmoji} ${data.event}
          </span>
        `;
      } else {
        this.els.eventSection.classList.add("hidden");
      }
    }

    // 音频特征
    if (data.audio_features && this.els.resultAudioFeatures) {
      this._renderAudioFeatures(data.audio_features);
    }

    // 文本情感
    if (this.els.sentimentSection && this.els.resultTextSentiment) {
      if (data.text_sentiment) {
        this._renderTextSentiment(data.text_sentiment);
      } else {
        this.els.sentimentSection.classList.add("hidden");
      }
    }

    // 保存到历史记录
    this.addToHistory(data);
  },

  _getEventEmoji(event) {
    const emojiMap = {
      "语音": "🗣️",
      "笑声": "😂",
      "掌声": "👏",
      "哭声": "😢",
      "音乐": "🎵",
    };
    return emojiMap[event] || "🔔";
  },

  _renderAudioFeatures(features) {
    if (!this.els.resultAudioFeatures) {
      console.error("resultAudioFeatures元素不存在");
      return;
    }
    const items = [];
    if (features.loudness_db !== undefined) {
      items.push({
        label: "响度",
        value: `${features.loudness_db.toFixed(1)} dB`,
        icon: "🔊",
      });
    }
    if (features.duration_sec !== undefined) {
      items.push({
        label: "时长",
        value: `${features.duration_sec.toFixed(2)} 秒`,
        icon: "⏱️",
      });
    }
    if (features.rms_energy !== undefined) {
      items.push({
        label: "能量",
        value: features.rms_energy.toFixed(4),
        icon: "⚡",
      });
    }

    let html = '<div class="features-display">';
    items.forEach(item => {
      html += `
        <div class="feature-item">
          <div class="feature-label">${item.icon} ${item.label}</div>
          <div class="feature-value">${item.value}</div>
        </div>
      `;
    });
    html += "</div>";
    this.els.resultAudioFeatures.innerHTML = html;
  },

  _renderTextSentiment(sentiment) {
    if (!this.els.sentimentSection || !this.els.resultTextSentiment) {
      console.error("sentimentSection或resultTextSentiment元素不存在");
      return;
    }
    this.els.sentimentSection.classList.remove("hidden");
    const score = sentiment.score || 0;
    const label = sentiment.label || "中性";
    const percentage = Math.abs(score * 100);
    
    // 计算进度条位置（0-100%）
    const barPosition = score > 0 ? 50 + (score * 50) : 50 + (score * 50);

    let html = `
      <div class="sentiment-display">
        <div class="sentiment-score">${percentage.toFixed(0)}%</div>
        <div class="sentiment-label">${label}</div>
        <div class="sentiment-bar">
          <div class="sentiment-bar-fill" style="width: ${barPosition}%; margin-left: ${score < 0 ? (50 - barPosition) + '%' : '0'}"></div>
        </div>
      </div>
    `;
    this.els.resultTextSentiment.innerHTML = html;
  },

  _renderSenseVoiceEmotion(data) {
    if (!this.els.resultEmotion) {
      console.error("resultEmotion元素不存在");
      return;
    }
    const emotionRaw = (data.emotion_raw || "NEUTRAL").toLowerCase();
    const emotionText = data.emotion || data.emotion_raw || "中性";
    
    const classMap = {
      happy: "happy",
      sad: "sad",
      angry: "angry",
      neutral: "neutral",
      surprise: "surprise",
      fearful: "fearful",
      disgusted: "disgusted",
    };
    
    const emojiMap = {
      happy: "😊",
      sad: "😢",
      angry: "😠",
      neutral: "😐",
      surprise: "😲",
      fearful: "😨",
      disgusted: "🤢",
    };

    const cls = classMap[emotionRaw] || "neutral";
    const emoji = emojiMap[cls] || "😐";
    
    this.els.resultEmotion.innerHTML = `
      <span class="emotion-tag ${cls}">
        ${emoji} ${emotionText}
      </span>
    `;
  },

  _renderParaformerEmotion(data) {
    if (!this.els.resultEmotion) {
      console.error("resultEmotion元素不存在");
      return;
    }
    if (!data.emotion) return;
    const emo = data.emotion;
    const probs = emo.probabilities || {};

    const labelMap = {
      neu: { text: "中性", emoji: "😐" },
      ang: { text: "生气", emoji: "😠" },
      hap: { text: "开心", emoji: "😊" },
      sad: { text: "悲伤", emoji: "😢" },
    };
    
    const topKey = emo.label;
    const topLabel = labelMap[topKey] || { text: topKey, emoji: "😐" };

    let html = `
      <div style="margin-bottom: 1rem;">
        <span class="emotion-tag ${topKey === 'hap' ? 'happy' : topKey === 'sad' ? 'sad' : topKey === 'ang' ? 'angry' : 'neutral'}">
          ${topLabel.emoji} ${topLabel.text} (${(emo.confidence * 100).toFixed(1)}%)
        </span>
      </div>
      <div class="prob-bar-group">
    `;

    for (const key of ["neu", "ang", "hap", "sad"]) {
      const val = probs[key] || 0;
      const pct = (val * 100).toFixed(1);
      const isTop = key === topKey;
      const label = labelMap[key];
      
      html += `
        <div class="prob-bar-row">
          <span class="prob-bar-label">${label.emoji} ${label.text}</span>
          <div class="prob-bar-track">
            <div class="prob-bar-fill${isTop ? " top" : ""}" style="width:${pct}%"></div>
          </div>
          <span class="prob-bar-value">${pct}%</span>
        </div>
      `;
    }
    
    html += "</div>";
    this.els.resultEmotion.innerHTML = html;
  },

  showError(message) {
    if (this.els.resultPanel) {
      this.els.resultPanel.classList.add("hidden");
    }
    if (this.els.errorPanel) {
      this.els.errorPanel.classList.remove("hidden");
    }
    if (this.els.errorMessage) {
      this.els.errorMessage.textContent = message;
    } else {
      console.error("错误消息元素不存在:", message);
      alert("错误: " + message);
    }
  },

  hideResults() {
    if (this.els.resultPanel) {
      this.els.resultPanel.classList.add("hidden");
    }
    if (this.els.errorPanel) {
      this.els.errorPanel.classList.add("hidden");
    }
  },

  // ---- 实时转录 ----
  showLiveTranscript(text) {
    if (this.els.liveTranscriptPanel && this.els.liveTranscriptText) {
      this.els.liveTranscriptPanel.classList.remove("hidden");
      this.els.liveTranscriptText.textContent = text || "正在识别...";
    }
  },

  hideLiveTranscript() {
    if (this.els.liveTranscriptPanel) {
      this.els.liveTranscriptPanel.classList.add("hidden");
    }
  },

  // ---- 历史记录 ----
  addToHistory(data) {
    const history = this.getHistory();
    const item = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      mode: data.mode || "unknown",
      transcript: data.transcript || "(无文本)",
      emotion: data.emotion || data.emotion_raw || "未知",
      emotionRaw: data.emotion_raw || "NEUTRAL",
      language: data.language || "",
      event: data.event || "",
      audioFeatures: data.audio_features || {},
      textSentiment: data.text_sentiment || {},
    };

    history.unshift(item);
    // 最多保存50条
    if (history.length > 50) {
      history.pop();
    }

    localStorage.setItem("emotion_history", JSON.stringify(history));
    this.renderHistory();
  },

  getHistory() {
    try {
      const stored = localStorage.getItem("emotion_history");
      return stored ? JSON.parse(stored) : [];
    } catch (e) {
      return [];
    }
  },

  clearHistory() {
    if (confirm("确定要清空所有历史记录吗？")) {
      localStorage.removeItem("emotion_history");
      this.renderHistory();
    }
  },

  loadHistory() {
    this.renderHistory();
  },

  renderHistory() {
    if (!this.els.historyList) {
      console.error("historyList元素不存在");
      return;
    }
    const history = this.getHistory();
    const listEl = this.els.historyList;

    if (history.length === 0) {
      listEl.innerHTML = '<div class="history-empty">暂无历史记录</div>';
      return;
    }

    let html = "";
    history.forEach(item => {
      const date = new Date(item.timestamp);
      const timeStr = `${date.getMonth() + 1}/${date.getDate()} ${date.getHours().toString().padStart(2, "0")}:${date.getMinutes().toString().padStart(2, "0")}`;
      
      const emotionRaw = (item.emotionRaw || "NEUTRAL").toLowerCase();
      const emotionClass = emotionRaw === "happy" ? "happy" : 
                          emotionRaw === "sad" ? "sad" : 
                          emotionRaw === "angry" ? "angry" : "neutral";
      
      const emotionEmoji = emotionRaw === "happy" ? "😊" :
                          emotionRaw === "sad" ? "😢" :
                          emotionRaw === "angry" ? "😠" : "😐";

      html += `
        <div class="history-item" data-id="${item.id}">
          <div class="history-item-header">
            <span class="history-time">${timeStr}</span>
            <span class="history-emotion emotion-tag ${emotionClass}">
              ${emotionEmoji} ${item.emotion}
            </span>
          </div>
          <div class="history-transcript">${item.transcript}</div>
        </div>
      `;
    });

    listEl.innerHTML = html;

    // 绑定点击事件，点击历史记录项时显示详情
    listEl.querySelectorAll(".history-item").forEach(el => {
      el.addEventListener("click", () => {
        const id = parseInt(el.dataset.id);
        const item = history.find(h => h.id === id);
        if (item) {
          // 重新构造数据格式并显示
          const data = {
            mode: item.mode,
            transcript: item.transcript,
            emotion: item.emotion,
            emotion_raw: item.emotionRaw,
            language: item.language,
            event: item.event,
            audio_features: item.audioFeatures,
            text_sentiment: item.textSentiment,
          };
          this.showResult(data);
          // 滚动到结果面板
          document.getElementById("result-panel").scrollIntoView({ behavior: "smooth" });
        }
      });
    });
  },
};
