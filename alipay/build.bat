@echo off
chcp 65001 >nul
echo ================================================
echo 中文语音识别 - FunASR / SenseVoice 安装
echo ================================================
echo.
echo 这些是阿里达摩院的开源模型，中文效果远超 Whisper
echo.

echo [1/5] 安装 PyTorch (CPU)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo [2/5] 安装 FunASR (阿里语音识别工具包)...
pip install funasr modelscope

echo.
echo [3/5] 安装音频处理库...
pip install soundfile numpy

echo.
echo [4/5] 安装 PyAudio...
pip install pyaudio
if errorlevel 1 (
    pip install pipwin
    pipwin install pyaudio
)

echo.
echo [5/5] 安装文本分析库...
pip install snownlp

echo.
echo ================================================
echo ✅ 安装完成！
echo.
echo 推荐使用:
echo   方案A (Paraformer): python emotion_recognition_v4_funasr.py
echo   方案B (SenseVoice): python emotion_recognition_sensevoice.py
echo.
echo SenseVoice 自带情感识别，是一站式方案！
echo ================================================
pause