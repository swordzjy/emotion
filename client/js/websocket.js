/**
 * WebSocketManager — 管理与后端的 WebSocket 连接
 */
class WebSocketManager {
  constructor({ onStatus, onResult, onError, onClose }) {
    this.onStatus = onStatus;
    this.onResult = onResult;
    this.onError = onError;
    this.onClose = onClose;
    this.ws = null;
  }

  connect(mode) {
    return new Promise((resolve, reject) => {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${location.host}/ws/audio?mode=${mode}`;

      this.ws = new WebSocket(url);
      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => resolve();

      this.ws.onmessage = (e) => {
        if (typeof e.data === "string") {
          const msg = JSON.parse(e.data);
          if (msg.type === "status" && this.onStatus) {
            this.onStatus(msg.status);
          } else if (msg.type === "result" && this.onResult) {
            this.onResult(msg);
          } else if (msg.type === "error" && this.onError) {
            this.onError(msg.message);
          }
        }
      };

      this.ws.onerror = (e) => {
        reject(new Error("WebSocket 连接失败"));
      };

      this.ws.onclose = () => {
        if (this.onClose) this.onClose();
      };
    });
  }

  sendAudio(float32Array) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(float32Array.buffer);
    }
  }

  sendStop() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
  }

  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
