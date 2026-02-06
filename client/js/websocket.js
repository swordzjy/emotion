/**
 * WebSocketManager — 管理与后端的 WebSocket 连接
 */
class WebSocketManager {
  constructor({ onStatus, onResult, onError, onClose, onTranscript }) {
    this.onStatus = onStatus;
    this.onResult = onResult;
    this.onError = onError;
    this.onClose = onClose;
    this.onTranscript = onTranscript;
    this.ws = null;
  }

  connect(mode) {
    return new Promise((resolve, reject) => {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${location.host}/ws/audio?mode=${mode}`;

      try {
        this.ws = new WebSocket(url);
        this.ws.binaryType = "arraybuffer";

        // 设置超时
        const timeout = setTimeout(() => {
          if (this.ws.readyState === WebSocket.CONNECTING) {
            this.ws.close();
            reject(new Error("连接超时，请检查服务器是否运行"));
          }
        }, 10000); // 10秒超时

        this.ws.onopen = () => {
          clearTimeout(timeout);
          resolve();
        };

        this.ws.onmessage = (e) => {
          if (typeof e.data === "string") {
            try {
              const msg = JSON.parse(e.data);
              if (msg.type === "status" && this.onStatus) {
                this.onStatus(msg.status);
              } else if (msg.type === "transcript" && this.onTranscript) {
                this.onTranscript(msg.text, msg.is_final);
              } else if (msg.type === "result" && this.onResult) {
                this.onResult(msg);
              } else if (msg.type === "error" && this.onError) {
                this.onError(msg.message);
              }
            } catch (err) {
              console.error("解析WebSocket消息失败:", err);
            }
          }
        };

        this.ws.onerror = (e) => {
          clearTimeout(timeout);
          reject(new Error("WebSocket 连接失败，请检查服务器状态"));
        };

        this.ws.onclose = (e) => {
          clearTimeout(timeout);
          const wasClosedByUser = this.ws && this.ws._closedByUser;
          if (e.code !== 1000 && e.code !== 1001) {
            // 非正常关闭
            if (this.onError && !wasClosedByUser) {
              this.onError(`连接已断开 (代码: ${e.code})`);
            }
          }
          if (this.onClose) this.onClose();
        };
      } catch (error) {
        reject(new Error("无法创建WebSocket连接: " + error.message));
      }
    });
  }

  sendAudio(float32Array) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(float32Array.buffer);
    }
  }

  sendStop() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: "stop" }));
        console.log("✓ WebSocket sendStop: 成功发送stop消息");
      } catch (e) {
        console.error("✗ WebSocket sendStop: 发送失败", e);
      }
    } else {
      console.warn("✗ WebSocket sendStop: 连接未打开，状态:", this.ws?.readyState);
    }
  }

  close() {
    if (this.ws && this.ws.readyState !== WebSocket.CLOSED) {
      // 标记为用户主动关闭
      this.ws._closedByUser = true;
      try {
        if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
          this.ws.close(1000, "正常关闭");
        }
      } catch (e) {
        // 忽略关闭时的错误
        console.warn("关闭WebSocket时出错:", e);
      }
      // 延迟设置为null，确保onclose事件处理器能访问到_closedByUser
      setTimeout(() => {
        this.ws = null;
      }, 100);
    } else {
      this.ws = null;
    }
  }
}
