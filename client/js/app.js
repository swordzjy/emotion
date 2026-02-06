/**
 * App — 主逻辑，串联 AudioCapture + WebSocketManager + UI
 */
(function () {
  let audioCapture = null;
  let wsManager = null;
  let timerInterval = null;
  let startTime = null;
  let isRecording = false;

  function getSelectedMode() {
    const radio = document.querySelector('input[name="mode"]:checked');
    return radio ? radio.value : "sensevoice";
  }

  function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(() => {
      UI.setTimer((Date.now() - startTime) / 1000);
    }, 200);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  async function startRecording() {
    if (isRecording) return;
    isRecording = true;

    UI.hideResults();
    UI.setRecordingState(true);
    UI.setStatus("连接中...", "connected");

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
      });
      await wsManager.connect(mode);

      // 启动麦克风
      audioCapture = new AudioCapture({
        onAudioFrame: (chunk) => wsManager.sendAudio(chunk),
        onWaveformData: (samples) => UI.drawWaveform(samples),
      });
      await audioCapture.start();

      UI.setStatus("等待语音...", "recording");
      startTimer();
    } catch (err) {
      UI.showError("启动失败: " + err.message);
      stopRecording();
    }
  }

  function stopRecording() {
    if (audioCapture) {
      audioCapture.stop();
      audioCapture = null;
    }
    if (wsManager) {
      wsManager.sendStop();
      // 不立即关闭 ws，等待结果返回后 close
    }
    stopTimer();
    isRecording = false;
    UI.setRecordingState(false);

    // 恢复模式切换
    document.querySelectorAll('input[name="mode"]').forEach(
      (r) => (r.disabled = false)
    );
  }

  function handleStatus(status) {
    const map = {
      listening: ["等待语音...", "recording"],
      speech_start: ["检测到语音", "recording"],
      speech_end: ["语音段结束", "recording"],
      silence_detected: ["语音结束，正在分析...", "processing"],
      processing: ["分析中...", "processing"],
    };
    const [text, cls] = map[status] || [status, "connected"];
    UI.setStatus(text, cls);

    if (status === "silence_detected") {
      // 服务端检测到静音，停止客户端录音
      if (audioCapture) {
        audioCapture.stop();
        audioCapture = null;
      }
      stopTimer();
      UI.setRecordingState(false);
      UI.clearWaveform();
      isRecording = false;

      document.querySelectorAll('input[name="mode"]').forEach(
        (r) => (r.disabled = false)
      );
    }
  }

  function handleResult(data) {
    UI.setStatus("完成", "connected");
    if (data.error) {
      UI.showError(data.error);
    } else {
      UI.showResult(data);
    }
    // 关闭 WebSocket
    if (wsManager) {
      wsManager.close();
      wsManager = null;
    }
  }

  function handleError(message) {
    UI.setStatus("错误", "disconnected");
    UI.showError(message);
    stopRecording();
    if (wsManager) {
      wsManager.close();
      wsManager = null;
    }
  }

  function handleClose() {
    if (isRecording) {
      stopRecording();
    }
    UI.setStatus("就绪", "disconnected");
  }

  // 初始化
  document.addEventListener("DOMContentLoaded", () => {
    UI.init();

    document.getElementById("btn-record").addEventListener("click", startRecording);
    document.getElementById("btn-stop").addEventListener("click", stopRecording);
  });
})();
