"""FastAPI 入口 — WebSocket、SSE 连接端点、静态文件"""

import asyncio
import json
import logging
import queue
import time
import os
import sys
import warnings

# 忽略 transformers + torch 的 pytree 弃用警告（库兼容性问题，不影响功能）
warnings.filterwarnings(
    "ignore",
    message=r".*_register_pytree_node.*deprecated.*",
    category=FutureWarning,
)

# 确保项目根目录在 Python 路径中（解决从不同目录启动时的 ModuleNotFoundError）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    PROJECT_ROOT,
    DEFAULT_MODE,
    MODEL_CACHE_DIR,
    SILERO_VAD_PATH,
    SPEECHBRAIN_EMOTION_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 尽早打印路径，便于 Ubuntu/Windows 迁移排查（在 models 导入前）
logger.info(
    "项目路径: PROJECT_ROOT=%s | MODEL_CACHE_DIR=%s | SILERO=%s | SPEECHBRAIN=%s",
    PROJECT_ROOT,
    MODEL_CACHE_DIR,
    SILERO_VAD_PATH,
    SPEECHBRAIN_EMOTION_DIR,
)
try:
    from .models import ModelManager
    from .ws_handler import handle_audio_ws
except ModuleNotFoundError as e:
    _srv = __import__("server")
    _dp = getattr(_srv, "__path__", [])
    _mp = os.path.join(_dp[0], "models.py") if _dp else "(unknown)"
    raise RuntimeError(
        f"无法导入 server.models: {e}\n"
        f"models.py 预期路径: {_mp} (存在: {os.path.isfile(_mp) if _dp else '?'})\n"
        f"sys.path 前 3 项: {sys.path[:3]}"
    ) from e


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
        last_heartbeat = time.time()
        heartbeat_interval = 1.0  # 每1秒发送心跳，避免首次加载 SpeechBrain（较慢）时连接被判定超时

        # 在线程池中加载模型
        load_task = loop.run_in_executor(None, mm.load_for_mode, mode, on_progress)

        try:
            while True:
                current_time = time.time()
                
                # 从队列取进度事件
                try:
                    while True:
                        evt = progress_queue.get_nowait()
                        yield f"data: {json.dumps(evt)}\n\n"
                        last_heartbeat = current_time
                except queue.Empty:
                    pass

                # 检查任务是否完成
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

                # 发送心跳保持连接活跃
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield f"data: {json.dumps({'step': 'heartbeat', 'status': 'loading'})}\n\n"
                    last_heartbeat = current_time

                await asyncio.sleep(0.1)
        except Exception as e:
            logger.exception(f"SSE事件生成器异常: {e}")
            yield f"data: {json.dumps({'step': 'error', 'status': 'error', 'message': str(e)})}\n\n"

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
client_dir = os.path.join(PROJECT_ROOT, "client")
logger.info(f"客户端目录路径: {client_dir}")
logger.info(f"客户端目录是否存在: {os.path.isdir(client_dir)}")

if os.path.isdir(client_dir):
    # 检查关键文件是否存在
    index_html = os.path.join(client_dir, "index.html")
    logger.info(f"index.html路径: {index_html}")
    logger.info(f"index.html是否存在: {os.path.exists(index_html)}")
    
    # 挂载静态文件
    try:
        app.mount("/client", StaticFiles(directory=client_dir, html=True), name="client")
        logger.info("✓ 成功挂载客户端静态文件")
    except Exception as e:
        logger.error(f"✗ 挂载客户端静态文件失败: {e}", exc_info=True)
    
    # 添加根路径重定向到客户端
    @app.get("/")
    async def root():
        """根路径重定向到客户端"""
        index_path = os.path.join(client_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            return {"error": "客户端文件未找到", "client_dir": client_dir}
    
    # 添加调试页面
    @app.get("/debug")
    async def debug_page():
        """调试页面"""
        debug_path = os.path.join(client_dir, "debug.html")
        if os.path.exists(debug_path):
            return FileResponse(debug_path)
        else:
            return {"error": "调试页面未找到"}
else:
    logger.error(f"✗ 客户端目录不存在: {client_dir}")
