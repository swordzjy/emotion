/**
 * UI — DOM 操作、波形可视化、结果渲染
 */
const UI = {
  // 缓存 DOM 引用
  els: {},

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
    };
    this._waveCtx = this.els.waveform.getContext("2d");
    this._resizeCanvas();
    window.addEventListener("resize", () => this._resizeCanvas());
    this.clearWaveform();
  },

  _resizeCanvas() {
    const c = this.els.waveform;
    c.width = c.parentElement.clientWidth - 16; // padding
  },

  // ---- 状态 ----
  setStatus(text, dotClass) {
    this.els.statusText.textContent = text;
    this.els.connectionStatus.className = "status-dot " + (dotClass || "disconnected");
  },

  setTimer(seconds) {
    if (seconds == null) {
      this.els.timer.textContent = "";
    } else {
      const m = Math.floor(seconds / 60);
      const s = Math.floor(seconds % 60);
      this.els.timer.textContent = `${m}:${s.toString().padStart(2, "0")}`;
    }
  },

  setRecordingState(recording) {
    const btn = this.els.btnRecord;
    if (recording) {
      btn.textContent = "录音中...";
      btn.classList.add("recording");
      this.els.btnStop.disabled = false;
    } else {
      btn.textContent = "开始录音";
      btn.classList.remove("recording");
      btn.disabled = false;
      this.els.btnStop.disabled = true;
    }
  },

  // ---- 波形 ----
  drawWaveform(samples) {
    const ctx = this._waveCtx;
    const w = this.els.waveform.width;
    const h = this.els.waveform.height;

    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const step = Math.max(1, Math.floor(samples.length / w));
    for (let i = 0; i < w; i++) {
      const idx = i * step;
      const v = idx < samples.length ? samples[idx] : 0;
      const y = (1 - v) * h / 2;
      if (i === 0) ctx.moveTo(i, y);
      else ctx.lineTo(i, y);
    }
    ctx.stroke();
  },

  clearWaveform() {
    const ctx = this._waveCtx;
    const w = this.els.waveform.width;
    const h = this.els.waveform.height;
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, w, h);

    // 中线
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
  },

  // ---- 结果 ----
  showResult(data) {
    this.els.errorPanel.classList.add("hidden");
    this.els.resultPanel.classList.remove("hidden");

    // 转录
    this.els.resultTranscript.textContent = data.transcript || "(无文本)";

    // 语言
    if (data.language) {
      this.els.resultLanguage.textContent = `语言: ${data.language}`;
      this.els.resultLanguage.classList.remove("hidden");
    } else {
      this.els.resultLanguage.classList.add("hidden");
    }

    // 情感
    if (data.mode === "sensevoice") {
      this._renderSenseVoiceEmotion(data);
    } else {
      this._renderParaformerEmotion(data);
    }

    // 事件
    if (data.event) {
      this.els.eventSection.classList.remove("hidden");
      this.els.resultEvent.innerHTML = `<span class="event-tag">${data.event}</span>`;
    } else {
      this.els.eventSection.classList.add("hidden");
    }

    // 音频特征
    if (data.audio_features) {
      const af = data.audio_features;
      this.els.resultAudioFeatures.textContent =
        `响度: ${af.loudness_db} dB | 时长: ${af.duration_sec}s`;
    }

    // 文本情感
    if (data.text_sentiment) {
      this.els.sentimentSection.classList.remove("hidden");
      const ts = data.text_sentiment;
      this.els.resultTextSentiment.textContent =
        `${(ts.score * 100).toFixed(1)}% ${ts.label}`;
    } else {
      this.els.sentimentSection.classList.add("hidden");
    }
  },

  _renderSenseVoiceEmotion(data) {
    const emotionRaw = (data.emotion_raw || "NEUTRAL").toLowerCase();
    const classMap = {
      happy: "happy", sad: "sad", angry: "angry",
      neutral: "neutral", surprise: "surprise",
      fearful: "fearful", disgusted: "disgusted",
    };
    const cls = classMap[emotionRaw] || "neutral";
    this.els.resultEmotion.innerHTML =
      `<span class="emotion-tag ${cls}">${data.emotion || data.emotion_raw}</span>`;
  },

  _renderParaformerEmotion(data) {
    if (!data.emotion) return;
    const emo = data.emotion;
    const probs = emo.probabilities || {};

    const labelMap = { neu: "中性", ang: "生气", hap: "开心", sad: "悲伤" };
    const topKey = emo.label; // e.g. "hap"

    let html = `<p style="margin-bottom:0.5rem">
      ${emo.label_zh || emo.label} (${(emo.confidence * 100).toFixed(1)}%)</p>`;
    html += '<div class="prob-bar-group">';

    for (const key of ["neu", "ang", "hap", "sad"]) {
      const val = probs[key] || 0;
      const pct = (val * 100).toFixed(1);
      const isTop = key === topKey;
      html += `
        <div class="prob-bar-row">
          <span class="prob-bar-label">${labelMap[key]}</span>
          <div class="prob-bar-track">
            <div class="prob-bar-fill${isTop ? " top" : ""}" style="width:${pct}%"></div>
          </div>
          <span class="prob-bar-value">${pct}%</span>
        </div>`;
    }
    html += "</div>";
    this.els.resultEmotion.innerHTML = html;
  },

  showError(message) {
    this.els.resultPanel.classList.add("hidden");
    this.els.errorPanel.classList.remove("hidden");
    this.els.errorMessage.textContent = message;
  },

  hideResults() {
    this.els.resultPanel.classList.add("hidden");
    this.els.errorPanel.classList.add("hidden");
  },
};
