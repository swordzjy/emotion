/**
 * App — 主逻辑，串联 AudioCapture + WebSocketManager + UI
 */
(function () {
  let audioCapture = null;
  let wsManager = null;
  let timerInterval = null;
  let startTime = null;
  let isRecording = false;
  let isConnected = false;
  let eventSource = null;
  let connectionTimeoutId = null;

  function getSelectedMode() {
    const radio = document.querySelector('input[name="mode"]:checked');
    return radio ? radio.value : "sensevoice";
  }

  async function connectToServer() {
    if (isConnected) {
      disconnectFromServer();
      return;
    }

    const mode = getSelectedMode();
    const btnConnect = document.getElementById("btn-connect");
    const progressPanel = document.getElementById("connect-progress");
    const progressFill = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");

    // 禁用模式切换和连接按钮
    document.querySelectorAll('input[name="mode"]').forEach(r => r.disabled = true);
    btnConnect.disabled = true;
    btnConnect.classList.add("connecting");
    progressPanel.classList.remove("hidden");

    UI.setStatus("正在连接服务器...", "connected");
    progressText.textContent = "正在连接服务器...";
    progressFill.style.width = "10%";

    try {
      // 使用SSE连接加载模型
      const url = `/api/connect?mode=${mode}`;
      eventSource = new EventSource(url);

      const steps = {
        vad: { name: "VAD模型", progress: 20 },
        paraformer: { name: "Paraformer模型", progress: 50 },
        sensevoice: { name: "SenseVoice模型", progress: 50 },
        emotion: { name: "情感模型", progress: 80 },
        all: { name: "完成", progress: 100 },
      };

      let connectionCompleted = false;
      let lastMessageTime = Date.now();
      const connectionTimeout = 120000; // 2分钟超时
      
      // 设置连接超时
      connectionTimeoutId = setTimeout(() => {
        if (!connectionCompleted && eventSource) {
          console.warn("连接超时");
          eventSource.close();
          eventSource = null;
          connectionCompleted = true;
          
          btnConnect.disabled = false;
          btnConnect.classList.remove("connecting");
          progressPanel.classList.add("hidden");
          document.querySelectorAll('input[name="mode"]').forEach(r => r.disabled = false);
          
          UI.showError("连接超时，请检查网络或服务器状态");
          UI.setStatus("连接超时", "disconnected");
        }
      }, connectionTimeout);

      eventSource.onmessage = (event) => {
        try {
          lastMessageTime = Date.now();
          const data = JSON.parse(event.data);
          
          // 忽略心跳消息（但更新最后消息时间）
          if (data.step === "heartbeat") {
            return;
          }
          
          if (data.status === "error") {
            if (connectionTimeoutId) {
              clearTimeout(connectionTimeoutId);
              connectionTimeoutId = null;
            }
            connectionCompleted = true;
            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
            
            btnConnect.disabled = false;
            btnConnect.classList.remove("connecting");
            progressPanel.classList.add("hidden");
            document.querySelectorAll('input[name="mode"]').forEach(r => r.disabled = false);
            
            UI.showError("连接失败: " + (data.message || "未知错误"));
            UI.setStatus("连接失败", "disconnected");
            return;
          }

          const step = steps[data.step];
          if (step) {
            progressFill.style.width = step.progress + "%";
            progressText.textContent = `正在加载${step.name}...`;
          }

          if (data.step === "all" && data.status === "ready") {
            if (connectionTimeoutId) {
              clearTimeout(connectionTimeoutId);
              connectionTimeoutId = null;
            }
            connectionCompleted = true;
            progressFill.style.width = "100%";
            progressText.textContent = "连接成功！";
            
            // 延迟关闭连接，确保消息已接收
            setTimeout(() => {
              if (eventSource) {
                eventSource.close();
                eventSource = null;
              }
              
              isConnected = true;
              btnConnect.disabled = false;
              btnConnect.classList.remove("connecting");
              btnConnect.classList.add("connected");
              btnConnect.querySelector(".btn-text").textContent = "已连接";
              
              document.getElementById("btn-record").disabled = false;
              document.getElementById("record-hint").classList.add("hidden");
              
              progressPanel.classList.add("hidden");
              UI.setStatus("就绪，可以开始录音", "connected");
              
              // 恢复模式切换
              document.querySelectorAll('input[name="mode"]').forEach(r => r.disabled = false);
            }, 300);
          }
        } catch (err) {
          console.error("解析SSE消息失败:", err);
        }
      };

      eventSource.onerror = (error) => {
        // 如果连接已完成，不处理错误（可能是正常关闭）
        if (connectionCompleted) {
          return;
        }
        
        console.error("SSE连接错误:", error, "readyState:", eventSource?.readyState);
        
        // 检查连接状态
        if (eventSource && eventSource.readyState === EventSource.CLOSED) {
          // 连接已关闭，可能是正常完成或错误
          // 如果还没收到完成消息，则认为是错误
          if (!connectionCompleted) {
            if (connectionTimeoutId) {
              clearTimeout(connectionTimeoutId);
              connectionTimeoutId = null;
            }
            connectionCompleted = true;
            
            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
            
            btnConnect.disabled = false;
            btnConnect.classList.remove("connecting");
            progressPanel.classList.add("hidden");
            document.querySelectorAll('input[name="mode"]').forEach(r => r.disabled = false);
            
            // 检查是否超时
            const timeSinceLastMessage = Date.now() - lastMessageTime;
            if (timeSinceLastMessage > 10000) {
              UI.showError("连接超时，模型加载时间过长");
            } else {
              UI.showError("连接中断，请检查服务器状态后重试");
            }
            UI.setStatus("连接失败", "disconnected");
          }
        } else if (eventSource && eventSource.readyState === EventSource.CONNECTING) {
          // 正在重连，不处理
          console.log("SSE正在重连...");
        }
      };

    } catch (err) {
      console.error("连接失败:", err);
      if (connectionTimeoutId) {
        clearTimeout(connectionTimeoutId);
        connectionTimeoutId = null;
      }
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      
      btnConnect.disabled = false;
      btnConnect.classList.remove("connecting");
      progressPanel.classList.add("hidden");
      document.querySelectorAll('input[name="mode"]').forEach(r => r.disabled = false);
      
      UI.showError("连接失败: " + err.message);
      UI.setStatus("连接失败", "disconnected");
    }
  }

  function disconnectFromServer() {
    console.log("disconnectFromServer被调用, isRecording:", isRecording);
    
    if (connectionTimeoutId) {
      clearTimeout(connectionTimeoutId);
      connectionTimeoutId = null;
    }
    
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }

    // 注意：如果正在录音，不要在这里调用stopRecording
    // 因为stopRecording会发送stop消息，但连接可能已经断开
    // 应该在handleClose中处理
    if (wsManager) {
      wsManager.close();
      wsManager = null;
    }

    isConnected = false;
    const btnConnect = document.getElementById("btn-connect");
    if (btnConnect) {
      btnConnect.classList.remove("connected", "connecting");
      btnConnect.querySelector(".btn-text").textContent = "连接服务器";
    }
    
    const btnRecord = document.getElementById("btn-record");
    if (btnRecord) {
      btnRecord.disabled = true;
    }
    
    const recordHint = document.getElementById("record-hint");
    if (recordHint) {
      recordHint.classList.remove("hidden");
    }
    
    const progressPanel = document.getElementById("connect-progress");
    if (progressPanel) {
      progressPanel.classList.add("hidden");
    }
    
    UI.setStatus("未连接", "disconnected");
  }

  function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      UI.setTimer(elapsed);
    }, 100);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
    UI.setTimer(null);
  }

  async function startRecording() {
    if (isRecording) return;
    
    // 检查是否已连接
    if (!isConnected) {
      UI.showError("请先连接服务器后再开始录音");
      return;
    }
    
    // 检查浏览器支持
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      UI.showError("您的浏览器不支持麦克风访问，请使用现代浏览器（Chrome、Firefox、Edge等）");
      return;
    }

    isRecording = true;
    UI.hideResults();
    UI.setRecordingState(true);
    UI.setStatus("准备中...", "connected");

    const mode = getSelectedMode();

    // 禁用模式切换
    document.querySelectorAll('input[name="mode"]').forEach(
      (r) => (r.disabled = true)
    );

    try {
      // 建立 WebSocket
      wsManager = new WebSocketManager({
        onStatus: handleStatus,
        onResult: handleResult,
        onError: handleError,
        onClose: handleClose,
        onTranscript: handleTranscript,
      });
      
      UI.setStatus("连接服务器...", "connected");
      await wsManager.connect(mode);

      // 启动麦克风
      UI.setStatus("启动麦克风...", "connected");
      let audioFrameCount = 0;
      audioCapture = new AudioCapture({
        onAudioFrame: (chunk) => {
          audioFrameCount++;
          if (wsManager && wsManager.ws) {
            const wsState = wsManager.ws.readyState;
            if (wsState === WebSocket.OPEN) {
              wsManager.sendAudio(chunk);
              // 每100帧记录一次（约2秒）
              if (audioFrameCount % 100 === 0) {
                console.log(`✓ 已发送 ${audioFrameCount} 个音频帧，WebSocket状态: OPEN`);
              }
            } else {
              // WebSocket未打开，记录警告
              if (audioFrameCount % 100 === 0) {
                console.warn(`⚠ 音频帧 ${audioFrameCount}: WebSocket状态不是OPEN (${wsState})`);
              }
            }
          } else {
            if (audioFrameCount % 100 === 0) {
              console.warn(`⚠ 音频帧 ${audioFrameCount}: wsManager或ws不存在`);
            }
          }
        },
        onWaveformData: (samples) => UI.drawWaveform(samples),
      });
      
      await audioCapture.start();

      UI.setStatus("正在录音...", "recording");
      UI.showLiveTranscript("正在识别...");
      startTimer();
    } catch (err) {
      console.error("启动失败:", err);
      UI.showError("启动失败: " + err.message);
      stopRecording();
    }
  }

  function stopRecording() {
    console.log("=".repeat(50));
    console.log("停止录音，发送stop消息到服务器");
    console.log("WebSocket状态:", wsManager?.ws?.readyState, "OPEN=", WebSocket.OPEN);
    
    // 先发送停止消息到服务器（在停止音频采集之前）
    if (wsManager && wsManager.ws) {
      const wsState = wsManager.ws.readyState;
      console.log("当前WebSocket状态:", wsState);
      if (wsState === WebSocket.OPEN) {
        wsManager.sendStop();
        console.log("✓ 已发送stop消息到服务器");
      } else {
        console.warn("⚠ WebSocket未打开，无法发送stop消息。状态:", wsState);
      }
    } else {
      console.warn("⚠ wsManager或ws不存在，无法发送stop消息");
    }
    
    // 然后停止音频采集
    if (audioCapture) {
      audioCapture.stop();
      audioCapture = null;
      console.log("✓ 已停止音频采集");
    }
    
    stopTimer();
    isRecording = false;
    UI.setRecordingState(false);
    UI.clearWaveform();
    
    // 注意：不立即隐藏实时转录面板，等待最终转录结果
    // UI.hideLiveTranscript(); // 注释掉，让最终转录也能显示

    // 恢复模式切换
    document.querySelectorAll('input[name="mode"]').forEach(
      (r) => (r.disabled = false)
    );
  }

  function handleStatus(status) {
    const statusMap = {
      recording: { text: "正在录音...", class: "recording" },
      processing: { text: "正在分析情绪...", class: "processing" },
      timeout: { text: "录音超时（60秒）", class: "disconnected" },
    };
    
    const statusInfo = statusMap[status] || { text: status, class: "connected" };
    UI.setStatus(statusInfo.text, statusInfo.class);

    if (status === "timeout") {
      // 达到最大录音时长，自动停止
      stopRecording();
    }
  }

  function handleTranscript(text, isFinal) {
    console.log(`收到转录消息: isFinal=${isFinal}, text="${text}"`);
    if (isFinal) {
      // 最终转录，更新显示但保持面板可见
      UI.showLiveTranscript(text);
      // 延迟隐藏，让用户看到最终转录
      setTimeout(() => {
        UI.hideLiveTranscript();
      }, 1000);
    } else {
      // 实时转录，显示在实时转录面板
      UI.showLiveTranscript(text);
    }
  }

  function handleResult(data) {
    console.log("收到分析结果:", data);
    console.log("当前isRecording状态:", isRecording);
    
    // 如果还在录音状态，说明这是意外的结果（不应该发生）
    if (isRecording) {
      console.warn("警告：收到分析结果时仍在录音状态，这不应该发生！");
      // 停止录音，但不发送stop消息（因为服务器已经完成了分析）
      if (audioCapture) {
        audioCapture.stop();
        audioCapture = null;
      }
      stopTimer();
      isRecording = false;
      UI.setRecordingState(false);
      UI.clearWaveform();
    }
    
    UI.setStatus("分析完成", "connected");
    UI.hideLiveTranscript(); // 隐藏实时转录面板
    
    if (data.error) {
      UI.showError(data.error);
    } else {
      // 确保mode字段存在
      if (!data.mode) {
        data.mode = getSelectedMode();
      }
      UI.showResult(data);
      
      // 滚动到结果面板
      setTimeout(() => {
        const resultPanel = document.getElementById("result-panel");
        if (resultPanel) {
          resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      }, 100);
    }
    
    // 关闭 WebSocket（延迟关闭，确保所有消息都已处理）
    setTimeout(() => {
      if (wsManager) {
        wsManager.close();
        wsManager = null;
      }
    }, 100);
  }

  function handleError(message) {
    console.error("WebSocket错误:", message);
    UI.setStatus("连接错误", "disconnected");
    UI.showError(message || "连接出错，请检查网络或服务器状态");
    stopRecording();
    
    if (wsManager) {
      wsManager.close();
      wsManager = null;
    }
  }

  function handleClose() {
    console.log("WebSocket连接关闭, isRecording:", isRecording);
    
    // 如果还在录音状态，说明连接意外断开
    if (isRecording) {
      console.warn("警告：WebSocket连接关闭时仍在录音状态");
      // 停止本地录音，但不发送stop消息（连接已断开）
      if (audioCapture) {
        audioCapture.stop();
        audioCapture = null;
      }
      stopTimer();
      isRecording = false;
      UI.setRecordingState(false);
      UI.clearWaveform();
      UI.hideLiveTranscript();
      
      // 恢复模式切换
      document.querySelectorAll('input[name="mode"]').forEach(
        (r) => (r.disabled = false)
      );
      
      UI.showError("连接意外断开，请重新连接后重试");
    }
    
    UI.setStatus("连接已断开", "disconnected");
  }

  // 初始化
  document.addEventListener("DOMContentLoaded", () => {
    UI.init();

    // 绑定按钮事件
    const btnConnect = document.getElementById("btn-connect");
    const btnRecord = document.getElementById("btn-record");
    const btnStop = document.getElementById("btn-stop");

    if (btnConnect) {
      btnConnect.addEventListener("click", connectToServer);
    }

    if (btnRecord) {
      btnRecord.addEventListener("click", startRecording);
    }

    if (btnStop) {
      btnStop.addEventListener("click", stopRecording);
    }

    // 模式切换时断开连接
    document.querySelectorAll('input[name="mode"]').forEach(radio => {
      radio.addEventListener("change", () => {
        if (isConnected) {
          disconnectFromServer();
        }
      });
    });

    // 键盘快捷键支持
    document.addEventListener("keydown", (e) => {
      // 空格键开始/停止录音（不在输入框中时）
      if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "TEXTAREA") {
        e.preventDefault();
        if (isRecording) {
          stopRecording();
        } else {
          startRecording();
        }
      }
      
      // ESC键停止录音
      if (e.code === "Escape" && isRecording) {
        stopRecording();
      }
    });

    // 页面可见性变化时停止录音
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && isRecording) {
        stopRecording();
      }
    });

    // 检查WebSocket支持
    if (!window.WebSocket) {
      UI.showError("您的浏览器不支持WebSocket，请使用现代浏览器");
    }
  });
})();
