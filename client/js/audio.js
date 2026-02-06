/**
 * AudioCapture — Web Audio API 麦克风采集（AudioWorklet）
 * 输出 16kHz float32 PCM，按 512 样本分块
 */
class AudioCapture {
  constructor({ onAudioFrame, onWaveformData }) {
    this.onAudioFrame = onAudioFrame;       // (Float32Array) => void
    this.onWaveformData = onWaveformData;   // (Float32Array) => void  用于波形可视化
    this.audioContext = null;
    this.stream = null;
    this.workletNode = null;
    this.source = null;
    this.active = false;

    this.SAMPLE_RATE = 16000;
  }

  async start() {
    if (this.active) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: this.SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    this.audioContext = new AudioContext({ sampleRate: this.SAMPLE_RATE });

    // 加载 AudioWorklet 处理器
    await this.audioContext.audioWorklet.addModule("js/audio-processor.js");

    this.source = this.audioContext.createMediaStreamSource(this.stream);

    this.workletNode = new AudioWorkletNode(this.audioContext, "chunk-processor");
    this.workletNode.port.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "chunk" && this.onAudioFrame) {
        this.onAudioFrame(msg.data);
      } else if (msg.type === "waveform" && this.onWaveformData) {
        this.onWaveformData(msg.data);
      }
    };

    this.source.connect(this.workletNode);
    this.workletNode.connect(this.audioContext.destination);

    this.active = true;
  }

  stop() {
    this.active = false;

    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode.port.onmessage = null;
      this.workletNode = null;
    }
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}
