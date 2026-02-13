#!/bin/bash
# 确保在项目根目录启动服务，避免 ModuleNotFoundError
cd "$(dirname "$0")"
exec python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 "$@"
