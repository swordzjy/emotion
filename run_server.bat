@echo off
REM 确保在项目根目录启动服务
cd /d "%~dp0"
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 %*
