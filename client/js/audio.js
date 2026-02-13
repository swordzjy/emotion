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

    try {
      // 请求麦克风权限
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // 创建音频上下文
      this.audioContext = new AudioContext({ 
        sampleRate: this.SAMPLE_RATE,
        latencyHint: 'interactive'
      });

      // 如果上下文被暂停（浏览器策略），恢复它
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      // 加载 AudioWorklet 处理器
      try {
        // 使用相对于当前页面的路径
        const processorPath = new URL("js/audio-processor.js", window.location.href).href;
        await this.audioContext.audioWorklet.addModule(processorPath);
      } catch (error) {
        // 如果AudioWorklet不支持，抛出更友好的错误
        throw new Error("浏览器不支持AudioWorklet，请使用Chrome、Edge或Firefox最新版本");
      }

      this.source = this.audioContext.createMediaStreamSource(this.stream);

      this.workletNode = new AudioWorkletNode(this.audioContext, "chunk-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
      });

      this.workletNode.port.onmessage = (e) => {
        const msg = e.data;
        if (msg.type === "chunk" && this.onAudioFrame) {
          this.onAudioFrame(msg.data);
        } else if (msg.type === "waveform" && this.onWaveformData) {
          this.onWaveformData(msg.data);
        }
      };

      // 处理错误
      this.workletNode.onprocessorerror = (error) => {
        console.error("AudioWorklet处理错误:", error);
      };

      this.source.connect(this.workletNode);
      // 注意：不连接到destination，避免音频反馈
      // this.workletNode.connect(this.audioContext.destination);

      // 页面从后台切回时恢复 AudioContext（解决先访问 static 再访问本页时移动端无声音问题）
      this._onVisibilityChange = () => {
        if (document.visibilityState === "visible" && this.audioContext?.state === "suspended") {
          this.audioContext.resume().catch(() => {});
        }
      };
      document.addEventListener("visibilitychange", this._onVisibilityChange);

      this.active = true;
    } catch (error) {
      // 清理资源
      this.stop();
      throw error;
    }
  }

  stop() {
    this.active = false;

    try {
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
        this.stream.getTracks().forEach((t) => {
          t.stop();
          t.enabled = false;
        });
        this.stream = null;
      }
      if (this._onVisibilityChange) {
        document.removeEventListener("visibilitychange", this._onVisibilityChange);
        this._onVisibilityChange = null;
      }
      if (this.audioContext) {
        this.audioContext.close().catch(err => {
          console.warn("关闭AudioContext时出错:", err);
        });
        this.audioContext = null;
      }
    } catch (error) {
      console.error("停止音频捕获时出错:", error);
    }
  }
}
