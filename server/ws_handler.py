"""WebSocket 处理 — 流式 VAD 状态机"""

import asyncio
import json
import time
import logging
import struct

import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect

from .config import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    BYTES_PER_FRAME,
    SILENCE_THRESHOLD_SEC,
    MAX_RECORD_SECONDS,
    MIN_SPEECH_SEC,
)
from .models import ModelManager
from .pipeline import run_paraformer_analysis, run_sensevoice_analysis

logger = logging.getLogger(__name__)


async def send_status(ws: WebSocket, status: str):
    await ws.send_json({"type": "status", "status": status})


async def handle_audio_ws(ws: WebSocket, mode: str):
    """
    处理单个 WebSocket 音频会话。

    客户端发送二进制帧（float32 PCM 16kHz），服务端做流式 VAD，
    检测到语音结束后运行分析管线，返回 JSON 结果。
    """
    await ws.accept()
    logger.info(f"WebSocket 连接, mode={mode}")

    mm = ModelManager()

    # 确保所需模型已加载
    if mode == "paraformer" and mm.paraformer_model is None:
        mm.load_paraformer()
    elif mode == "sensevoice" and mm.sensevoice_model is None:
        mm.load_sensevoice()

    # 每个会话独立的 VADIterator
    vad_iterator = mm.create_vad_iterator()

    audio_chunks: list[np.ndarray] = []
    buffer = bytearray()
    speech_detected = False
    silence_start: float | None = None
    start_time = time.time()

    await send_status(ws, "listening")

    try:
        while True:
            # 超时检查
            elapsed = time.time() - start_time
            if elapsed > MAX_RECORD_SECONDS:
                logger.info("达到最大录音时长")
                await send_status(ws, "silence_detected")
                break

            # 接收数据（二进制或 JSON）
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if "text" in msg:
                data = json.loads(msg["text"])
                if data.get("type") == "stop":
                    logger.info("客户端发送 stop")
                    break
                continue

            if "bytes" not in msg:
                continue

            raw = msg["bytes"]
            buffer.extend(raw)

            # 按 BYTES_PER_FRAME 切块处理
            while len(buffer) >= BYTES_PER_FRAME:
                frame_bytes = bytes(buffer[:BYTES_PER_FRAME])
                del buffer[:BYTES_PER_FRAME]

                # float32 解码
                audio_float = np.array(
                    struct.unpack(f"<{CHUNK_SAMPLES}f", frame_bytes),
                    dtype=np.float32,
                )
                audio_chunks.append(audio_float)

                # VAD 检测
                audio_tensor = torch.from_numpy(audio_float)
                speech_dict = vad_iterator(audio_tensor)

                if speech_dict:
                    if "start" in speech_dict:
                        if not speech_detected:
                            speech_detected = True
                            silence_start = None
                            await send_status(ws, "speech_start")

                    if "end" in speech_dict:
                        silence_start = time.time()
                        await send_status(ws, "speech_end")

                # 静音检测
                if speech_detected and silence_start is not None:
                    if time.time() - silence_start >= SILENCE_THRESHOLD_SEC:
                        logger.info(f"静音 {SILENCE_THRESHOLD_SEC}s，停止接收")
                        await send_status(ws, "silence_detected")
                        # 跳出内层 while 和外层 while
                        buffer.clear()
                        break
            else:
                # 内层 while 正常结束（buffer 不够一帧），继续外层
                continue
            # 内层 while 被 break（静音检测），也跳出外层
            break

        # ---- 分析阶段 ----
        if not audio_chunks:
            await ws.send_json({"type": "error", "message": "未收到音频数据"})
            return

        full_audio = np.hstack(audio_chunks)
        duration = len(full_audio) / SAMPLE_RATE

        if not speech_detected or duration < MIN_SPEECH_SEC:
            await ws.send_json({"type": "error", "message": "未检测到有效语音，请重试"})
            return

        await send_status(ws, "processing")
        audio_tensor = torch.from_numpy(full_audio)

        # CPU 密集推理放到线程池
        loop = asyncio.get_event_loop()
        if mode == "paraformer":
            result = await loop.run_in_executor(None, run_paraformer_analysis, audio_tensor)
        else:
            result = await loop.run_in_executor(None, run_sensevoice_analysis, audio_tensor)

        await ws.send_json(result)

    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.exception(f"WebSocket 处理异常: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
