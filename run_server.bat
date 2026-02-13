@echo off
REM 确保在项目根目录启动服务
REM --limit-concurrency 4 --timeout-keep-alive 30 单用户多次使用更稳定
cd /d "%~dp0"
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --limit-concurrency 4 --timeout-keep-alive 30 %*
