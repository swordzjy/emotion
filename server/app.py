"""FastAPI 入口 — WebSocket、SSE 连接端点、静态文件"""

import asyncio
import json
import logging
import queue

from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, DEFAULT_MODE
from .models import ModelManager
from .ws_handler import handle_audio_ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="语音情感识别服务")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 健康检查
@app.get("/health")
async def health():
    mm = ModelManager()
    return {
        "status": "ok",
        "models": {
            "vad": mm.vad_model is not None,
            "paraformer": mm.paraformer_model is not None,
            "sensevoice": mm.sensevoice_model is not None,
            "emotion_classifier": mm.emotion_classifier is not None,
        },
    }


# SSE 连接端点 — 按需加载模型并推送进度
@app.get("/api/connect")
async def connect_models(mode: str = Query(default=DEFAULT_MODE)):
    if mode not in ("paraformer", "sensevoice", "both"):
        return StreamingResponse(
            iter([f"data: {json.dumps({'step': 'error', 'status': 'error', 'message': f'未知模式: {mode}'})}\n\n"]),
            media_type="text/event-stream",
        )

    progress_queue: queue.Queue = queue.Queue()

    def on_progress(step: str, status: str):
        progress_queue.put({"step": step, "status": status})

    async def event_generator():
        loop = asyncio.get_event_loop()
        mm = ModelManager()

        # 在线程池中加载模型
        load_task = loop.run_in_executor(None, mm.load_for_mode, mode, on_progress)

        while True:
            # 从队列取进度事件
            try:
                while True:
                    evt = progress_queue.get_nowait()
                    yield f"data: {json.dumps(evt)}\n\n"
            except queue.Empty:
                pass

            if load_task.done():
                # 检查是否有异常
                exc = load_task.exception()
                if exc:
                    yield f"data: {json.dumps({'step': 'error', 'status': 'error', 'message': str(exc)})}\n\n"
                else:
                    # 排空剩余事件
                    while not progress_queue.empty():
                        evt = progress_queue.get_nowait()
                        yield f"data: {json.dumps(evt)}\n\n"
                    yield f"data: {json.dumps({'step': 'all', 'status': 'ready'})}\n\n"
                break

            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# WebSocket 端点
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket, mode: str = DEFAULT_MODE):
    if mode not in ("paraformer", "sensevoice"):
        await ws.accept()
        await ws.send_json({"type": "error", "message": f"未知模式: {mode}"})
        await ws.close()
        return
    await handle_audio_ws(ws, mode)


# 挂载前端静态文件
import os

client_dir = os.path.join(PROJECT_ROOT, "client")
if os.path.isdir(client_dir):
    app.mount("/client", StaticFiles(directory=client_dir, html=True), name="client")
