#!/usr/bin/env python3
"""Ubuntu/Windows 环境诊断，用于定位 uvicorn 启动失败原因"""

import os
import sys

# 项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

def step(name):
    print(f"\n--- {name} ---")

print("Python:", sys.version)
print("CWD:", os.getcwd())
print("PROJECT_ROOT:", PROJECT_ROOT)

step("1. config")
try:
    from server.config import PROJECT_ROOT, MODEL_CACHE_DIR, SILERO_VAD_PATH, SPEECHBRAIN_EMOTION_DIR
    print("  PROJECT_ROOT:", PROJECT_ROOT)
    print("  MODEL_CACHE_DIR:", MODEL_CACHE_DIR)
    print("  SILERO_VAD_PATH:", SILERO_VAD_PATH)
    print("  SPEECHBRAIN_EMOTION_DIR:", SPEECHBRAIN_EMOTION_DIR)
    print("  model_cache exists:", os.path.isdir(MODEL_CACHE_DIR))
    print("  silero exists:", os.path.isdir(SILERO_VAD_PATH))
    print("  speechbrain exists:", os.path.isdir(SPEECHBRAIN_EMOTION_DIR))
    print("  OK")
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

step("2. torch / torchaudio")
try:
    import torch
    import torchaudio
    print("  OK")
except Exception as e:
    print("  FAIL:", e)
    sys.exit(1)

step("3. models (ModelManager)")
try:
    from server.models import ModelManager
    print("  OK")
except Exception as e:
    print("  FAIL:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

step("4. app")
try:
    from server.app import app
    print("  OK")
except Exception as e:
    print("  FAIL:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n--- 全部通过 ---")
print("若 uvicorn 仍失败，请将完整报错贴出，尤其是最后一行的 Exception 信息。")
