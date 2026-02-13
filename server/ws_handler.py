"""WebSocket 处理 — 流式转录 + 情绪分析"""

import asyncio
import gc
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
    MAX_RECORD_SECONDS,
    MIN_SPEECH_SEC,
    STREAMING_TRANSCRIBE_INTERVAL,
)
from .models import ModelManager
from .pipeline import run_paraformer_analysis, run_sensevoice_analysis

logger = logging.getLogger(__name__)


async def send_status(ws: WebSocket, status: str):
    await ws.send_json({"type": "status", "status": status})


async def send_transcript(ws: WebSocket, text: str, is_final: bool = False):
    """发送实时转录文本"""
    await ws.send_json({
        "type": "transcript",
        "text": text,
        "is_final": is_final,
    })


def _stream_transcribe_sync(audio_chunks: list[np.ndarray], mode: str, mm: ModelManager) -> str:
    """同步转录函数（在线程池中运行）"""
    if not audio_chunks:
        return ""
    
    try:
        full_audio = np.hstack(audio_chunks)
        audio_np = full_audio.astype(np.float32)
        
        if mode == "paraformer":
            if mm.paraformer_model is None:
                return ""
            result = mm.paraformer_model.generate(input=audio_np, batch_size_s=300)
            if result and len(result) > 0:
                return result[0].get("text", "")
        elif mode == "sensevoice":
            if mm.sensevoice_model is None:
                return ""
            result = mm.sensevoice_model.generate(
                input=audio_np,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            if result and len(result) > 0:
                raw_text = result[0].get("text", "")
                # 移除标签，只返回文本
                import re
                pattern = r"<\|([^|]+)\|>"
                clean_text = re.sub(pattern, "", raw_text).strip()
                return clean_text
    except Exception as e:
        logger.warning(f"流式转录失败: {e}")
    
    return ""


async def stream_transcribe(audio_chunks: list[np.ndarray], mode: str, mm: ModelManager, loop: asyncio.AbstractEventLoop) -> str:
    """异步转录函数（在线程池中执行）"""
    return await loop.run_in_executor(None, _stream_transcribe_sync, audio_chunks, mode, mm)


async def handle_audio_ws(ws: WebSocket, mode: str):
    """
    处理单个 WebSocket 音频会话。
    
    工作流程：
    1. 接收音频数据，累积到缓冲区
    2. 每2秒进行一次流式转录，发送实时转录结果
    3. 收到stop消息后，停止接收音频
    4. 对完整音频进行情绪分析
    """
    await ws.accept()
    logger.info(f"WebSocket 连接, mode={mode}")

    mm = ModelManager()

    # 确保VAD模型已加载（所有模式都需要）
    if mm.vad_model is None:
        logger.info("[VAD] WebSocket连接时加载VAD模型...")
        mm.load_silero_vad()

    # 确保所需模型已加载
    if mode == "paraformer" and mm.paraformer_model is None:
        logger.info("[ASR] WebSocket连接时加载Paraformer模型...")
        mm.load_paraformer()
    elif mode == "sensevoice" and mm.sensevoice_model is None:
        logger.info("[ASR+EMO] WebSocket连接时加载SenseVoice模型...")
        mm.load_sensevoice()

    audio_chunks: list[np.ndarray] = []
    buffer = bytearray()
    start_time = time.time()
    last_transcribe_time = start_time
    should_stop = False
    speech_detected = False
    loop = asyncio.get_event_loop()

    await send_status(ws, "recording")

    # 流式转录任务（仅实时转录，不进行情绪分析）
    async def periodic_transcribe():
        nonlocal last_transcribe_time, speech_detected
        transcribe_count = 0
        while not should_stop:
            await asyncio.sleep(STREAMING_TRANSCRIBE_INTERVAL)
            
            if should_stop:
                logger.info(f"流式转录任务收到停止信号 (should_stop={should_stop})")
                break
            
            # 检查是否有足够的音频进行转录（至少1秒）
            if audio_chunks and len(audio_chunks) > 0:
                # 创建音频副本，避免在转录时音频被修改
                chunks_copy = audio_chunks.copy()
                total_samples = sum(len(chunk) for chunk in chunks_copy)
                duration = total_samples / SAMPLE_RATE
                
                if duration >= 1.0:  # 至少1秒音频才转录
                    transcribe_count += 1
                    try:
                        logger.debug(f"[流式转录 #{transcribe_count}] 执行转录，音频时长: {duration:.2f}秒, 音频块数: {len(chunks_copy)}")
                        transcript = await stream_transcribe(chunks_copy, mode, mm, loop)
                        if transcript:
                            logger.debug(f"[流式转录 #{transcribe_count}] 转录结果: {transcript}")
                            await send_transcript(ws, transcript, is_final=False)
                            speech_detected = True
                        else:
                            logger.debug(f"[流式转录 #{transcribe_count}] 返回空结果")
                    except asyncio.CancelledError:
                        logger.info(f"[流式转录 #{transcribe_count}] 任务被取消")
                        raise
                    except Exception as e:
                        logger.warning(f"[流式转录 #{transcribe_count}] 转录失败: {e}", exc_info=True)
            else:
                logger.debug(f"[流式转录] 等待音频数据... (音频块数: {len(audio_chunks) if audio_chunks else 0})")
            
            last_transcribe_time = time.time()
        
        logger.info(f"流式转录任务结束，共执行 {transcribe_count} 次转录")

    # 启动定期转录任务
    transcribe_task = asyncio.create_task(periodic_transcribe())

    receive_count = 0
    try:
        while True:
            # 超时检查（60秒）
            elapsed = time.time() - start_time
            if elapsed > MAX_RECORD_SECONDS:
                logger.warning("=" * 50)
                logger.warning(f"⚠ 达到最大录音时长（60秒），停止接收音频")
                logger.warning(f"当前音频块: {len(audio_chunks)}, 总时长: {sum(len(chunk) for chunk in audio_chunks) / SAMPLE_RATE:.2f}秒")
                logger.warning("=" * 50)
                await send_status(ws, "timeout")
                should_stop = True
                break

            # 接收数据（二进制或 JSON）
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
                receive_count += 1
            except asyncio.TimeoutError:
                # 超时是正常的，继续等待（客户端可能暂时没有数据发送）
                # 每10次超时记录一次（约5秒）
                if receive_count % 10 == 0:
                    logger.debug(f"[接收循环] 等待数据中... (已接收 {receive_count} 条消息, 音频块: {len(audio_chunks)})")
                continue
            except WebSocketDisconnect:
                raise
            except Exception as e:
                err_str = str(e)
                # 客户端已断开后再次 receive 会触发此错误，视为正常断开
                if "disconnect" in err_str.lower():
                    logger.info("客户端已断开连接，停止接收")
                    should_stop = True
                    break
                logger.warning(f"接收消息时出现异常（继续等待）: {e}")
                await asyncio.sleep(0.1)
                continue

            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    logger.info(f"[接收 #{receive_count}] 收到文本消息: {data}")
                    if data.get("type") == "stop":
                        logger.info("=" * 50)
                        logger.info("✓ 客户端发送 stop，停止录音，准备开始分析")
                        logger.info(f"当前音频块数量: {len(audio_chunks)}, 总时长: {sum(len(chunk) for chunk in audio_chunks) / SAMPLE_RATE:.2f}秒")
                        logger.info(f"录音开始时间: {start_time}, 当前时间: {time.time()}, 实际耗时: {time.time() - start_time:.2f}秒")
                        logger.info(f"共接收 {receive_count} 条消息")
                        logger.info("=" * 50)
                        should_stop = True
                        # 停止接收新的音频数据
                        break
                    else:
                        logger.warning(f"收到未知的文本消息: {data}")
                except json.JSONDecodeError as e:
                    logger.error(f"解析JSON消息失败: {e}, 原始消息: {msg.get('text', '')}")
                continue

            if "bytes" not in msg:
                logger.warning(f"[接收 #{receive_count}] 收到非二进制非文本消息: {list(msg.keys())}")
                continue

            # 收到二进制音频数据
            raw = msg["bytes"]
            buffer.extend(raw)
            
            # 每100条消息记录一次
            if receive_count % 100 == 0:
                logger.debug(f"[接收 #{receive_count}] 收到 {len(raw)} 字节音频数据，缓冲区: {len(buffer)} 字节，音频块: {len(audio_chunks)}")

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

        # 停止定期转录任务，等待当前转录完成
        should_stop = True
        total_duration = sum(len(chunk) for chunk in audio_chunks) / SAMPLE_RATE
        logger.info("=" * 50)
        logger.info("⚠ 跳出主接收循环，准备停止录音")
        logger.info(f"最终音频块数量: {len(audio_chunks)}, 总时长: {total_duration:.2f}秒")
        logger.info(f"录音开始时间: {start_time}, 当前时间: {time.time()}, 实际耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"共接收 {receive_count} 条消息")
        logger.info(f"跳出循环的原因: should_stop={should_stop}")
        logger.info("=" * 50)
        
        # 等待一小段时间，让正在进行的转录完成
        await asyncio.sleep(0.5)
        
        # 取消转录任务
        if not transcribe_task.done():
            transcribe_task.cancel()
            try:
                await transcribe_task
            except asyncio.CancelledError:
                pass

        logger.info("开始最终分析和情绪分析...")

        # ---- 最终转录和情绪分析阶段（仅在停止后执行）----
        if not audio_chunks:
            await ws.send_json({"type": "error", "message": "未收到音频数据"})
            return

        full_audio = np.hstack(audio_chunks)
        duration = len(full_audio) / SAMPLE_RATE

        if duration < MIN_SPEECH_SEC:
            await ws.send_json({"type": "error", "message": "录音时长太短，请重试"})
            return

        # 发送处理状态
        await send_status(ws, "processing")
        logger.info(f"开始处理音频，时长: {duration:.2f}秒")
        
        # 最终转录（使用完整音频）
        logger.info("执行最终转录...")
        final_transcript = await stream_transcribe(audio_chunks, mode, mm, loop)
        
        if final_transcript:
            logger.info(f"最终转录结果: {final_transcript}")
            await send_transcript(ws, final_transcript, is_final=True)
        elif not speech_detected:
            logger.warning("未检测到有效语音")
            await ws.send_json({"type": "error", "message": "未检测到有效语音，请重试"})
            return

        # 情绪分析（使用完整音频，仅在停止后执行）
        logger.info("开始情绪分析...")
        audio_tensor = torch.from_numpy(full_audio)
        
        if mode == "paraformer":
            result = await loop.run_in_executor(None, run_paraformer_analysis, audio_tensor)
        else:
            result = await loop.run_in_executor(None, run_sensevoice_analysis, audio_tensor)
        
        # 确保转录文本使用最终转录结果
        if final_transcript:
            result["transcript"] = final_transcript

        logger.info("分析完成，发送结果")
        await ws.send_json(result)

        # 单用户多次使用后释放内存，避免进程无法建立新连接
        gc.collect()

        # 正常关闭连接
        try:
            await ws.close()
        except Exception:
            pass

    except WebSocketDisconnect as e:
        logger.warning("=" * 50)
        logger.warning("客户端断开连接（WebSocketDisconnect）")
        logger.warning(f"断开时音频块数量: {len(audio_chunks)}, 总时长: {sum(len(chunk) for chunk in audio_chunks) / SAMPLE_RATE:.2f}秒")
        logger.warning(f"should_stop状态: {should_stop}")
        logger.warning("注意：连接断开时不应该发送结果，直接返回")
        logger.warning("=" * 50)
        should_stop = True
        if not transcribe_task.done():
            transcribe_task.cancel()
        # 直接返回，不执行后续的分析逻辑
        return
    except Exception as e:
        logger.exception(f"WebSocket 处理异常: {e}")
        should_stop = True
        if not transcribe_task.done():
            transcribe_task.cancel()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
