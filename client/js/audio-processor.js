/**
 * AudioWorkletProcessor — 在音频线程中运行，将麦克风 PCM 按 512 样本分块发回主线程
 */
class ChunkProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    this.CHUNK_SAMPLES = 512;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // mono channel

    // 累积
    const merged = new Float32Array(this.buffer.length + samples.length);
    merged.set(this.buffer);
    merged.set(samples, this.buffer.length);
    this.buffer = merged;

    // 按 512 切块发回主线程
    while (this.buffer.length >= this.CHUNK_SAMPLES) {
      const chunk = this.buffer.slice(0, this.CHUNK_SAMPLES);
      this.buffer = this.buffer.slice(this.CHUNK_SAMPLES);
      this.port.postMessage({ type: "chunk", data: chunk });
    }

    // 发送波形数据（用于可视化）
    this.port.postMessage({ type: "waveform", data: samples });

    return true;
  }
}

registerProcessor("chunk-processor", ChunkProcessor);
