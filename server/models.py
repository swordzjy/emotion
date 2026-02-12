"""单例 ModelManager — 管理所有 ML 模型的加载和访问"""

import gc
import os
import sys
import threading
import logging

import torch

# torchaudio 2.9+ 移除了 list_audio_backends，SpeechBrain 依赖此函数，必须在导入 SpeechBrain 前补丁
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from .config import (
    SAMPLE_RATE, SILERO_VAD_PATH, SPEECHBRAIN_EMOTION_DIR, MODEL_CACHE_DIR,
)

logger = logging.getLogger(__name__)

_modelscope_patch_applied = False


def _apply_modelscope_offline_patch():
    """ModelScope/FunASR 在有本地缓存时仍会联网检查，需拦截 snapshot_download 直接返回本地路径"""
    global _modelscope_patch_applied
    if _modelscope_patch_applied:
        return
    try:
        import modelscope
        _original = modelscope.snapshot_download

        def _patched_snapshot_download(model_id=None, model=None, *args, **kwargs):
            mid = model_id or model or (args[0] if args else None)
            if not mid:
                return _original(model_id=model_id, model=model, *args, **kwargs)
            cache_dir = kwargs.get("cache_dir") or os.environ.get("MODELSCOPE_CACHE") or os.environ.get("MODEL_CACHE_DIR")
            if not cache_dir:
                return _original(model_id=model_id, model=model, *args, **kwargs)
            # ModelScope 结构: cache_dir/models/org/model_name 或 cache_dir/hub/models/org/model_name
            for sub in ("models", os.path.join("hub", "models")):
                local_dir = os.path.join(cache_dir, sub, str(mid).replace("/", os.sep))
                model_pt = os.path.join(local_dir, "model.pt")
                if os.path.isfile(model_pt):
                    logger.info(f"[MODEL] 使用本地缓存: {local_dir}")
                    return local_dir
            return _original(model_id=model_id, model=model, *args, **kwargs)

        modelscope.snapshot_download = _patched_snapshot_download
        # FunASR 可能从 hub 子模块导入，需同步 patch
        if hasattr(modelscope, "hub") and hasattr(modelscope.hub, "snapshot_download"):
            modelscope.hub.snapshot_download = _patched_snapshot_download
        _modelscope_patch_applied = True
        logger.info("[MODEL] 已应用 ModelScope 离线补丁")
    except Exception as e:
        logger.warning(f"[MODEL] ModelScope 离线补丁未生效: {e}")


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
                source="github",
                force_reload=False,
                skip_validation=True,
                trust_repo=True,
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
            disable_update=True
        )
        logger.info("[ASR] Paraformer 加载完成")
        if on_progress:
            on_progress("paraformer", "done")

        # 降低峰值内存：释放 Paraformer 加载过程中的临时对象后再加载情感模型
        gc.collect()

        # 内存紧张时可设置环境变量 EMOTION_LAZY_LOAD=1，情感模型在首次分析时再加载
        if os.environ.get("EMOTION_LAZY_LOAD", "").strip() in ("1", "true", "yes"):
            logger.info("[EMO] 已启用延迟加载，情感模型将在首次分析时加载")
        else:
            self._load_emotion_classifier(on_progress)

    def ensure_emotion_loaded(self, on_progress=None):
        """确保情感模型已加载（用于延迟加载场景，首次分析时调用）"""
        self._load_emotion_classifier(on_progress)

    def _speechbrain_savedir_complete(self, savedir: str) -> bool:
        """检查 savedir 是否已包含 SpeechBrain 情感模型全部必需文件"""
        required = ("custom_interface.py", "hyperparams.yaml", "model.ckpt", "wav2vec2.ckpt")
        label_enc = ("label_encoder.ckpt", "label_encoder.txt")
        if not os.path.isdir(savedir):
            return False
        existing = set(os.listdir(savedir))
        if not all(f in existing for f in required):
            return False
        return any(e in existing for e in label_enc)

    def _patch_hyperparams_for_offline_wav2vec2(self, savedir: str) -> None:
        """若存在本地 wav2vec2-base，将 hyperparams 中的 wav2vec2_hub 指向本地路径，以支持离线"""
        import re
        w2v_dir = os.path.abspath(os.path.join(savedir, "wav2vec2-base"))
        config_path = os.path.join(w2v_dir, "config.json")
        if not os.path.isfile(config_path):
            logger.warning(
                "[EMO] wav2vec2-base 未找到，将尝试联网。请将 pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/wav2vec2-base/ "
                "从 Windows 拷到 Ubuntu（含 config.json、pytorch_model.bin 等）"
            )
            return
        hp_path = os.path.join(savedir, "hyperparams.yaml")
        if not os.path.isfile(hp_path):
            return
        with open(hp_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 统一 wav2vec2_hub 为当前机器的绝对路径（处理从 Windows 拷到 Ubuntu 后路径失效）
        local_path = w2v_dir.replace("\\", "/")
        new_line = f"wav2vec2_hub: {local_path}"
        if re.search(r"wav2vec2_hub:\s*[^\n]+", content):
            old_line = re.search(r"wav2vec2_hub:\s*[^\n]+", content).group(0)
            if old_line.strip() != new_line.strip():
                content = re.sub(r"wav2vec2_hub:\s*[^\n]+", new_line, content)
                with open(hp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info("[EMO] 已 patch hyperparams → wav2vec2_hub 指向本地 %s", local_path)

    def _ensure_speechbrain_savedir(self, savedir: str) -> None:
        """若 savedir 不完整，通过 HF snapshot_download 补全（临时允许联网）"""
        if self._speechbrain_savedir_complete(savedir):
            return
        logger.info("[EMO] savedir 不完整，尝试从 HuggingFace 补全 ...")
        orig_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            from huggingface_hub import snapshot_download
            os.makedirs(savedir, exist_ok=True)
            snapshot_download(
                repo_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                local_dir=savedir,
                local_dir_use_symlinks=False,
            )
            if self._speechbrain_savedir_complete(savedir):
                logger.info("[EMO] 补全完成")
            else:
                raise RuntimeError("补全后 savedir 仍缺少必需文件，请运行: python scripts/download_models.py --speechbrain")
        except Exception as e:
            raise RuntimeError(
                f"SpeechBrain 模型文件不完整且无法联网补全: {e}\n"
                "离线环境请在有网络机器上运行: python scripts/download_models.py --speechbrain\n"
                "然后将 pretrained_models/emotion-recognition-wav2vec2-IEMOCAP 目录复制到本机"
            ) from e
        finally:
            if orig_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = orig_offline

    def _load_emotion_classifier(self, on_progress=None):
        if self.emotion_classifier is not None:
            if on_progress:
                on_progress("emotion", "done")
            return

        logger.info("[EMO] 加载 SpeechBrain 情感模型 ...")
        if on_progress:
            on_progress("emotion", "loading")

        self._ensure_speechbrain_savedir(SPEECHBRAIN_EMOTION_DIR)
        self._patch_hyperparams_for_offline_wav2vec2(SPEECHBRAIN_EMOTION_DIR)

        sys.modules["speechbrain.integrations.nlp.flair_embeddings"] = None
        from speechbrain.inference.interfaces import foreign_class

        try:
            self.emotion_classifier = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir=SPEECHBRAIN_EMOTION_DIR,
                run_opts={"device": "cpu"},
            )
        except Exception as e:
            logger.exception(f"[EMO] SpeechBrain 加载失败: {e}")
            raise RuntimeError(
                f"SpeechBrain 情感模型加载失败: {e}\n"
                "请确保 pretrained_models/emotion-recognition-wav2vec2-IEMOCAP 下有完整文件 "
                "(model.ckpt, wav2vec2.ckpt, hyperparams.yaml, custom_interface.py, label_encoder.txt)，"
                "以及 wav2vec2-base/ 子目录（离线必需）。"
                "在有网络机器上运行: python scripts/download_models.py --speechbrain"
            ) from e

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
         #   {"model": "iic/SenseVoiceSmall", "trust_remote_code": True},
          #  {"model": "FunAudioLLM/SenseVoiceSmall", "trust_remote_code": True},
            {"model": "iic/SenseVoiceSmall", "trust_remote_code": True, "model_revision": "master",  "disable_update":True }
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
        """根据模式加载所需的全部模型（优先本地/缓存，不联网检查新版本）"""
        # 统一模型缓存目录：预下载后服务从此加载
        if os.path.isdir(MODEL_CACHE_DIR):
            os.environ["MODELSCOPE_CACHE"] = MODEL_CACHE_DIR
            hf_cache = os.path.join(MODEL_CACHE_DIR, "huggingface")
            if os.path.isdir(hf_cache):
                os.environ["HF_HOME"] = hf_cache
                os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache, "hub")
            logger.info(f"[MODEL] 使用本地缓存: {MODEL_CACHE_DIR}")
            _apply_modelscope_offline_patch()
        # 禁用 HuggingFace 在线检查，仅使用本地/缓存模型
        os.environ["HF_HUB_OFFLINE"] = "1"
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
