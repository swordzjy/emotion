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


def _log_memory():
    """输出当前内存占用（便于诊断 8GB 等低内存环境）"""
    try:
        if sys.platform == "linux":
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:") or line.startswith("VmSize:") or line.startswith("VmPeak:"):
                        logger.info("[MEM] %s", line.strip())
    except Exception as e:
        logger.debug("[MEM] 无法获取内存信息: %s", e)


_modelscope_patch_applied = False

# 服务启动时即设置缓存路径和离线补丁，避免首次连接时 FunASR 仍尝试联网
def _init_modelscope_offline():
    if os.path.isdir(MODEL_CACHE_DIR):
        os.environ["MODELSCOPE_CACHE"] = MODEL_CACHE_DIR
        _apply_modelscope_offline_patch()


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
            cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir)))
            if not os.path.isdir(cache_dir):
                return _original(model_id=model_id, model=model, *args, **kwargs)
            # ModelScope 结构: cache_dir/models/org/model_name 或 cache_dir/hub/models/org/model_name
            for sub in ("models", os.path.join("hub", "models")):
                local_dir = os.path.join(cache_dir, sub, str(mid).replace("/", os.sep))
                model_pt = os.path.join(local_dir, "model.pt")
                if os.path.isfile(model_pt):
                    logger.info(f"[MODEL] 使用本地缓存: {local_dir}")
                    return local_dir
            raise RuntimeError(
                f"[MODEL] 未找到 {mid} 的 model.pt (cache_dir={cache_dir})。"
                "需尝试联网下载，请先运行: python scripts/download_models.py"
            )

        modelscope.snapshot_download = _patched_snapshot_download
        # FunASR 可能从 hub 子模块导入，需同步 patch
        if hasattr(modelscope, "hub") and hasattr(modelscope.hub, "snapshot_download"):
            modelscope.hub.snapshot_download = _patched_snapshot_download
        _modelscope_patch_applied = True
        logger.info("[MODEL] 已应用 ModelScope 离线补丁")
    except Exception as e:
        logger.warning(f"[MODEL] ModelScope 离线补丁未生效: {e}")


# 模块加载时即初始化，确保 patch 在 FunASR 导入前生效
_init_modelscope_offline()


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
            raise RuntimeError(
                f"[VAD] Silero VAD 目录不存在: {SILERO_VAD_PATH}。"
                "需尝试联网下载，请先运行: python scripts/download_models.py --silero"
            )
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
        """若存在本地 wav2vec2-base，将 hyperparams 中的 wav2vec2_hub、pretrained_path 指向本地路径，以支持离线"""
        import re
        savedir_abs = os.path.abspath(savedir)
        local_base = savedir_abs.replace("\\", "/")
        w2v_dir = os.path.join(savedir_abs, "wav2vec2-base")
        config_path = os.path.join(w2v_dir, "config.json")
        if not os.path.isfile(config_path):
            logger.warning(
                "[EMO] wav2vec2-base 未找到，将尝试联网。请将 pretrained_models/emotion-recognition-wav2vec2-IEMOCAP/wav2vec2-base/ "
                "从 Windows 拷到 Ubuntu（含 config.json、pytorch_model.bin 等）"
            )
        hp_path = os.path.join(savedir, "hyperparams.yaml")
        if not os.path.isfile(hp_path):
            return
        with open(hp_path, "r", encoding="utf-8") as f:
            content = f.read()
        changed = False
        # 1. wav2vec2_hub → 本地 wav2vec2-base 路径（跨机器拷贝后路径会失效）
        w2v_line = f"wav2vec2_hub: {w2v_dir.replace(os.sep, '/')}"
        if re.search(r"wav2vec2_hub:\s*[^\n]+", content):
            old = re.search(r"wav2vec2_hub:\s*[^\n]+", content).group(0)
            if old.strip() != w2v_line.strip():
                content = re.sub(r"wav2vec2_hub:\s*[^\n]+", w2v_line, content)
                changed = True
        # 2. pretrained_path → 本地 savedir（关键：否则 Pretrainer 会从 HF 拉 model.ckpt，Ubuntu 上会卡住）
        if re.search(r"pretrained_path:\s*[^\n]+", content):
            old = re.search(r"pretrained_path:\s*[^\n]+", content).group(0)
            new_line = f"pretrained_path: {local_base}"
            if "speechbrain/" in old:
                content = re.sub(r"pretrained_path:\s*[^\n]+", new_line, content)
                changed = True
        if changed:
            with open(hp_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("[EMO] 已 patch hyperparams → wav2vec2_hub 与 pretrained_path 指向本地 %s", local_base)

    def _ensure_speechbrain_savedir(self, savedir: str) -> None:
        """若 savedir 不完整，通过 HF snapshot_download 补全。HF_HUB_OFFLINE=1 时禁止联网，仅提示"""
        if self._speechbrain_savedir_complete(savedir):
            return
        if os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true"):
            raise RuntimeError(
                "[EMO] SpeechBrain 模型文件不完整且已禁用联网。"
                "需尝试联网下载，请先运行: python scripts/download_models.py --speechbrain"
            )
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
        _log_memory()
        if on_progress:
            on_progress("emotion", "loading")

        logger.info("[EMO] 步骤 1/5: 检查 savedir ...")
        self._ensure_speechbrain_savedir(SPEECHBRAIN_EMOTION_DIR)
        if on_progress:
            on_progress("emotion", "loading")
        logger.info("[EMO] 步骤 2/5: patch hyperparams（wav2vec2/pretrained_path）...")
        self._patch_hyperparams_for_offline_wav2vec2(SPEECHBRAIN_EMOTION_DIR)
        _log_memory()

        sys.modules["speechbrain.integrations.nlp.flair_embeddings"] = None
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        # Ubuntu 上 HF_HUB_OFFLINE=1 可能导致 fetch 卡住，暂时解除（本地模型完整时不会触发下载）
        orig_hf = os.environ.pop("HF_HUB_OFFLINE", None)
        orig_tf = os.environ.pop("TRANSFORMERS_OFFLINE", None)
        logger.info("[EMO] 步骤 3/5: 导入 foreign_class ...")
        from speechbrain.inference.interfaces import foreign_class
        savedir_abs = os.path.abspath(SPEECHBRAIN_EMOTION_DIR)
        # FetchConfig/LocalStrategy 在旧版 SpeechBrain 中可能不存在，需兼容
        extra_kw = {}
        try:
            from speechbrain.utils.fetching import FetchConfig, LocalStrategy
            allow_net = os.environ.get("SB_ALLOW_NETWORK", "1") in ("1", "true")
            extra_kw["fetch_config"] = FetchConfig(
                allow_network=allow_net,
                overwrite=False,
                allow_updates=False,
                huggingface_cache_dir=savedir_abs,
            )
            extra_kw["local_strategy"] = LocalStrategy.NO_LINK
            logger.info("[EMO] 步骤 4/5: 调用 foreign_class（fetch_config+local_strategy）...")
        except ImportError:
            logger.info("[EMO] 步骤 4/5: 调用 foreign_class（兼容旧版 SpeechBrain，无 FetchConfig）...")
        _log_memory()
        # 设置 SB_DEBUG=1 可开启 SpeechBrain 详细日志，便于定位卡住位置
        _sb_log = logging.getLogger("speechbrain")
        _orig_level = _sb_log.level
        if os.environ.get("SB_DEBUG", "").strip() in ("1", "true"):
            _sb_log.setLevel(logging.DEBUG)
        try:
            self.emotion_classifier = foreign_class(
                source=savedir_abs,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir=savedir_abs,
                run_opts={"device": "cpu"},
                **extra_kw,
            )
        except Exception as e:
            _log_memory()
            logger.exception("[EMO] SpeechBrain 加载失败: %s", e)
            raise RuntimeError(
                f"SpeechBrain 情感模型加载失败: {e}\n"
                "8GB 内存机器建议: 1) 添加 4GB swap; 2) 设置 EMOTION_LAZY_LOAD=1 延迟加载情感模型。"
                "在有网络机器上运行: python scripts/download_models.py --speechbrain"
            ) from e
        finally:
            _sb_log.setLevel(_orig_level)
            if orig_hf is not None:
                os.environ["HF_HUB_OFFLINE"] = orig_hf
            if orig_tf is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = orig_tf

        logger.info("[EMO] 步骤 5/5: SpeechBrain 情感模型加载完成")
        _log_memory()
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
