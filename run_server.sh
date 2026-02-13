#!/bin/bash
# 确保在项目根目录启动服务，避免 ModuleNotFoundError
# --limit-concurrency 4：限制并发，单用户多次使用更稳定
# --timeout-keep-alive 30：保持连接超时，避免空闲连接占用
cd "$(dirname "$0")"
exec python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --limit-concurrency 4 --timeout-keep-alive 30 "$@"
