# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A real-time speech emotion recognition system that captures microphone audio and analyzes user emotional state through a multi-model ML pipeline: Voice Activity Detection (Silero VAD) → Speech-to-Text (OpenAI Whisper) → Emotion Classification (SpeechBrain Wav2Vec2) → Text Sentiment Analysis (SnowNLP + TextBlob).

## Running the Programs

```bash
# Activate virtual environment (Python 3.12)
.myenv312\Scripts\activate   # Windows

# Main program - fixed 8-second recording with full analysis pipeline
python emotion_recognition_v2.py

# Real-time version - auto-stops after 1.5s silence, max 30s
python emotion_recognition_realtime.py

# Quick audio test/demo
python checksound.py
```

## Installing Dependencies

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# If pyaudio fails on Windows:
pip install pipwin && pipwin install pyaudio
```

## Architecture

The processing pipeline flows through these stages:

1. **Audio Capture** — PyAudio records PCM 16-bit mono at 16kHz (512-byte chunks)
2. **Silero VAD** — Deep learning-based voice activity detection filters silence/noise (threshold: 0.5, min speech: 250ms, min silence: 100ms)
3. **Whisper ASR** — Transcribes speech to text with auto-language detection (base model from `./whisper_models/base.pt`)
4. **SpeechBrain Emotion** — Wav2Vec2-IEMOCAP classifies into 4 emotions: neutral, angry, happy, sad (from `./pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/`)
5. **Audio Features** — Extracts RMS energy, loudness (dB), zero-crossing rate
6. **Text Sentiment** — SnowNLP (Chinese) and TextBlob (English) analyze transcribed text
7. **Final Report** — Synthesizes voice emotion + text sentiment + audio features

## Key Source Files

- `emotion_recognition_v2.py` — Main program with fixed-duration recording and full pipeline
- `emotion_recognition_realtime.py` — `RealtimeVADRecorder` class with streaming VAD and auto-stop
- `checksound.py` — Legacy demo using energy-based VAD (not Silero)

## Model Locations

| Model | Path | Purpose |
|-------|------|---------|
| Silero VAD | loaded via torch.hub (cached) | Voice activity detection |
| Whisper base | `whisper_models/base.pt` | Speech-to-text |
| SpeechBrain emotion | `pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/` | 4-class emotion classification |

Change Whisper model size by modifying the `WHISPER_MODEL` variable (options: tiny/base/small/medium/large).

## Important Notes

- All models run on CPU by default — no GPU required
- A SpeechBrain workaround exists: `sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = None` bypasses a Flair import issue
- The `.env` file contains WeChat OAuth config for a separate web integration
- Intermediate audio files (`raw_recording.wav`, `vad_processed.wav`, `last_recording.wav`) are saved for debugging
- No automated test suite exists — testing is done manually by running the scripts with microphone input
