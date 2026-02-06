"""单例 ModelManager — 管理所有 ML 模型的加载和访问"""

import os
import sys
import threading
import logging

import torch

from .config import (
    SAMPLE_RATE, SILERO_VAD_PATH, SPEECHBRAIN_EMOTION_DIR,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """线程安全的模型管理器（单例）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.vad_model = None
        self.vad_utils = None
        self.paraformer_model = None
        self.sensevoice_model = None
        self.emotion_classifier = None

        # 保护 VAD get_speech_timestamps 的锁（reset_states 非线程安全）
        self.vad_lock = threading.Lock()

    # ---- Silero VAD ----
    def load_silero_vad(self, on_progress=None):
        if self.vad_model is not None:
            if on_progress:
                on_progress("vad", "done")
            return

        logger.info("[VAD] 加载 Silero VAD ...")
        if on_progress:
            on_progress("vad", "loading")
        if os.path.exists(SILERO_VAD_PATH):
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir=SILERO_VAD_PATH,
                model="silero_vad",
                source="local",
                onnx=False,
            )
            logger.info("[VAD] 本地加载完成")
        else:
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            logger.info("[VAD] 网络加载完成")
        if on_progress:
            on_progress("vad", "done")

    def create_vad_iterator(self):
        """为每个 WebSocket 会话创建独立的 VADIterator"""
        if self.vad_utils is None:
            raise RuntimeError("VAD 模型未加载，请先调用 load_silero_vad()")
        VADIterator = self.vad_utils[3]
        return VADIterator(self.vad_model, sampling_rate=SAMPLE_RATE)

    # ---- Paraformer + SpeechBrain ----
    def load_paraformer(self, on_progress=None):
        if self.paraformer_model is not None:
            if on_progress:
                on_progress("paraformer", "done")
            self._load_emotion_classifier(on_progress)
            return

        logger.info("[ASR] 加载 Paraformer ...")
        if on_progress:
            on_progress("paraformer", "loading")
        from funasr import AutoModel

        self.paraformer_model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            device="cpu",
        )
        logger.info("[ASR] Paraformer 加载完成")
        if on_progress:
            on_progress("paraformer", "done")

        self._load_emotion_classifier(on_progress)

    def _load_emotion_classifier(self, on_progress=None):
        if self.emotion_classifier is not None:
            if on_progress:
                on_progress("emotion", "done")
            return

        logger.info("[EMO] 加载 SpeechBrain 情感模型 ...")
        if on_progress:
            on_progress("emotion", "loading")
        sys.modules["speechbrain.integrations.nlp.flair_embeddings"] = None
        from speechbrain.inference.interfaces import foreign_class

        self.emotion_classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=SPEECHBRAIN_EMOTION_DIR,
            run_opts={"device": "cpu"},
        )
        logger.info("[EMO] SpeechBrain 情感模型加载完成")
        if on_progress:
            on_progress("emotion", "done")

    # ---- SenseVoice ----
    def load_sensevoice(self, on_progress=None):
        if self.sensevoice_model is not None:
            if on_progress:
                on_progress("sensevoice", "done")
            return

        logger.info("[ASR+EMO] 加载 SenseVoice ...")
        if on_progress:
            on_progress("sensevoice", "loading")
        from funasr import AutoModel

        load_methods = [
            {"model": "iic/SenseVoiceSmall", "trust_remote_code": True},
            {"model": "FunAudioLLM/SenseVoiceSmall", "trust_remote_code": True},
            {"model": "iic/SenseVoiceSmall", "trust_remote_code": True, "model_revision": "master"},
        ]

        for i, kwargs in enumerate(load_methods):
            try:
                self.sensevoice_model = AutoModel(**kwargs, device="cpu")
                logger.info(f"[ASR+EMO] SenseVoice 加载完成 (方法 {i + 1})")
                if on_progress:
                    on_progress("sensevoice", "done")
                return
            except Exception as e:
                logger.warning(f"[ASR+EMO] 方法 {i + 1} 失败: {e}")

        raise RuntimeError("SenseVoice 加载失败，请检查 funasr 版本")

    # ---- 按模式加载 ----
    def load_for_mode(self, mode: str, on_progress=None):
        """根据模式加载所需的全部模型"""
        self.load_silero_vad(on_progress)
        if mode == "paraformer":
            self.load_paraformer(on_progress)
        elif mode == "sensevoice":
            self.load_sensevoice(on_progress)
        elif mode == "both":
            self.load_paraformer(on_progress)
            self.load_sensevoice(on_progress)
        else:
            raise ValueError(f"未知模式: {mode}")
