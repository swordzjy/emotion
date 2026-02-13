#!/usr/bin/env python3
"""
预下载所有模型到本地目录，启动服务时可自动从本地加载（无需联网）。

用法:
    python scripts/download_models.py [--cache-dir PATH] [--check-only]

默认缓存目录: 与 server 一致（pretrained_models/model_cache 或 MODEL_CACHE_DIR 环境变量）
--check-only: 仅检查离线模型是否完整，不下载（缺失时打印提示）
--cache-dir: 覆盖默认，需与启动服务时的 MODEL_CACHE_DIR 一致以兼容

若 ~/.cache/modelscope、~/.cache/huggingface、~/.cache/torch 中已有模型，
会优先复制过来，避免重复下载。
"""

import argparse
import os
import shutil
import sys

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

# 与 server 一致的缓存目录解析（Windows/Ubuntu 兼容）
def _get_model_cache_dir(cache_dir_arg=None):
    """返回规范化后的 cache_dir，与 server/config 逻辑一致"""
    if cache_dir_arg:
        return os.path.abspath(os.path.expanduser(os.path.normpath(cache_dir_arg)))
    try:
        from server.config import MODEL_CACHE_DIR
        return MODEL_CACHE_DIR
    except ImportError:
        default = os.path.join(PROJECT_ROOT, "pretrained_models", "model_cache")
        return os.path.abspath(os.path.expanduser(os.environ.get("MODEL_CACHE_DIR", default)))

# 默认缓存根目录（通常 ~/.cache）
DEFAULT_CACHE_ROOT = os.path.expanduser("~/.cache")

# ModelScope 可能使用的目录结构（hub/models 或 models）
MODELSCOPE_IIC_MODELS = [
    "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc_ct-transformer_cn-en-common-vocab471067-large",
]


def _find_ms_iic_paths(cache_dir: str = None) -> list:
    """收集所有可能包含 ModelScope iic 模型的目录（用于检查是否已存在，避免重复下载）"""
    candidates = []
    # 1. 脚本 cache_dir 或与 server 一致的 MODEL_CACHE_DIR
    base = cache_dir or _get_model_cache_dir(None)
    base = os.path.abspath(os.path.expanduser(str(base)))
    for parts in [("hub", "models", "iic"), ("models", "iic")]:
        p = os.path.join(base, *parts)
        if os.path.isdir(p):
            candidates.append(p)
    # 2. ~/.cache/modelscope
    for parts in [("hub", "models", "iic"), ("models", "iic")]:
        p = os.path.join(DEFAULT_CACHE_ROOT, "modelscope", *parts)
        if os.path.isdir(p):
            candidates.append(p)
    # 3. 常见部署路径（Ubuntu 上常设为 /data/emotion_models）
    for extra in ("/data/emotion_models", "/data/emotion-models", "/data/models"):
        if os.path.isdir(extra):
            for parts in [("hub", "models", "iic"), ("models", "iic")]:
                p = os.path.join(extra, *parts)
                if os.path.isdir(p):
                    candidates.append(p)
    return candidates


def _paraformer_models_exist(cache_dir: str = None) -> tuple:
    """检查 Paraformer 三件套是否在任一已知路径下已存在，返回 (是否存在, 找到的路径)"""
    for iic_dir in _find_ms_iic_paths(cache_dir):
        if all(os.path.isdir(os.path.join(iic_dir, m)) for m in MODELSCOPE_IIC_MODELS):
            return True, iic_dir
    return False, None


def _copy_or_move_dir(src: str, dst: str, use_move: bool = False) -> bool:
    """复制或移动目录，跳过已存在且有内容的 dst"""
    if not os.path.isdir(src):
        return False
    if os.path.exists(dst):
        if os.path.isdir(dst) and os.listdir(dst):
            return True  # 视为已有，成功
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    try:
        if use_move:
            shutil.move(src, dst)
        else:
            shutil.copytree(src, dst, symlinks=False)
        return True
    except Exception as e:
        print(f"    警告: {e}")
        return False


def migrate_from_default_cache(cache_dir: str, use_move: bool = False) -> dict:
    """
    检查 ~/.cache 下的 modelscope、huggingface、torch，若已有模型则复制到目标 cache_dir。
    返回 {"modelscope": bool, "huggingface": bool, "silero": bool} 表示各是否已迁移。
    """
    result = {"modelscope": False, "huggingface": False, "silero": False}

    # 1. ModelScope: ~/.cache/modelscope 或 /data/emotion_models 等 -> cache_dir
    # 支持两种结构：hub/models/org 与 models/org
    ms_srcs = [
        os.path.join(DEFAULT_CACHE_ROOT, "modelscope", "hub", "models"),
        os.path.join(DEFAULT_CACHE_ROOT, "modelscope", "models"),
    ]
    for extra_root in ("/data/emotion_models", "/data/emotion-models"):
        if os.path.isdir(extra_root):
            ms_srcs.extend([
                os.path.join(extra_root, "hub", "models"),
                os.path.join(extra_root, "models"),
            ])
    ms_hub_src = None
    for src in ms_srcs:
        if os.path.isdir(src) and os.listdir(src):
            ms_hub_src = src
            break
    if ms_hub_src:
        # 目标保持与源一致：若源含 hub 则用 hub/models，否则用 models
        is_hub = os.path.sep + "hub" + os.path.sep in ms_hub_src or ms_hub_src.rstrip(os.path.sep).endswith("hub")
        dst_parts = ("hub", "models") if is_hub else ("models",)
        ms_hub_dst = os.path.join(cache_dir, *dst_parts)
        os.makedirs(ms_hub_dst, exist_ok=True)
        for org in os.listdir(ms_hub_src):
            org_src = os.path.join(ms_hub_src, org)
            org_dst = os.path.join(ms_hub_dst, org)
            if os.path.isdir(org_src) and (not os.path.exists(org_dst) or not os.listdir(org_dst)):
                _copy_or_move_dir(org_src, org_dst, use_move=use_move)
        result["modelscope"] = True

    # 2. HuggingFace: ~/.cache/huggingface -> cache_dir/huggingface
    hf_src = os.path.join(DEFAULT_CACHE_ROOT, "huggingface")
    if os.path.isdir(hf_src) and os.listdir(hf_src):
        hf_dst = os.path.join(cache_dir, "huggingface")
        if not (os.path.isdir(hf_dst) and os.listdir(hf_dst)) and _copy_or_move_dir(hf_src, hf_dst, use_move=use_move):
            result["huggingface"] = True
        elif os.path.isdir(hf_dst) and os.listdir(hf_dst):
            result["huggingface"] = True

    # 3. Silero (torch hub): ~/.cache/torch/hub/snakers4_silero-vad_master -> silero-vad/ 及 cache_dir/torch_hub
    torch_hub = os.path.join(DEFAULT_CACHE_ROOT, "torch", "hub")
    silero_src = os.path.join(torch_hub, "snakers4_silero-vad_master") if os.path.isdir(torch_hub) else None
    silero_dst = os.path.join(PROJECT_ROOT, "silero-vad")
    th_dst = os.path.join(cache_dir, "torch_hub", "snakers4_silero-vad_master")
    if silero_src and os.path.isdir(silero_src):
        # 先复制到 cache_dir/torch_hub
        if not os.path.exists(th_dst):
            _copy_or_move_dir(silero_src, th_dst, use_move=False)
        # 再复制或移动到项目 silero-vad/
        if not (os.path.exists(silero_dst) and os.path.isdir(silero_dst) and os.listdir(silero_dst)):
            _copy_or_move_dir(silero_src, silero_dst, use_move=use_move)
        result["silero"] = True

    return result


# SpeechBrain 情感模型必需文件（HF 仓库为 label_encoder.txt，部分环境保存为 .ckpt，二者有其一即可）
SPEECHBRAIN_REQUIRED_FILES = [
    "custom_interface.py",
    "hyperparams.yaml",
    "model.ckpt",
    "wav2vec2.ckpt",
]
SPEECHBRAIN_LABEL_ENCODER = ("label_encoder.ckpt", "label_encoder.txt")  # 二选一
WAV2VEC2_BASE_REPO = "facebook/wav2vec2-base"  # hyperparams 引用的架构，离线需本地化


def _wav2vec2_base_local_complete(speechbrain_dir: str) -> bool:
    """检查 wav2vec2-base 是否已本地化（用于离线加载）"""
    w2v_dir = os.path.join(speechbrain_dir, "wav2vec2-base")
    return os.path.isfile(os.path.join(w2v_dir, "config.json")) if os.path.isdir(w2v_dir) else False


def _ensure_wav2vec2_base_local(speechbrain_dir: str) -> bool:
    """下载 facebook/wav2vec2-base 到 savedir 并 patch hyperparams，以实现完全离线"""
    if _wav2vec2_base_local_complete(speechbrain_dir):
        return True
    try:
        from huggingface_hub import snapshot_download
        w2v_dir = os.path.join(speechbrain_dir, "wav2vec2-base")
        os.makedirs(w2v_dir, exist_ok=True)
        print("  下载 facebook/wav2vec2-base（架构，约 360MB）...")
        snapshot_download(
            repo_id=WAV2VEC2_BASE_REPO,
            local_dir=w2v_dir,
            local_dir_use_symlinks=False,
        )
        if not os.path.isfile(os.path.join(w2v_dir, "config.json")):
            return False
        # patch hyperparams：将 facebook/wav2vec2-base 替换为本地路径
        hp_path = os.path.join(speechbrain_dir, "hyperparams.yaml")
        if os.path.isfile(hp_path):
            with open(hp_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 使用相对于 hyperparams 的路径，便于跨机器拷贝
            local_path = os.path.join(os.path.dirname(hp_path), "wav2vec2-base").replace("\\", "/")
            if "facebook/wav2vec2-base" in content:
                content = content.replace("facebook/wav2vec2-base", local_path)
                with open(hp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("  已 patch hyperparams.yaml → 使用本地 wav2vec2-base")
        return True
    except Exception as e:
        print(f"  wav2vec2-base 本地化失败: {e}")
        return False


def _speechbrain_savedir_complete(speechbrain_dir: str) -> bool:
    """检查 savedir 是否已包含 SpeechBrain 情感模型全部必需文件"""
    if not os.path.isdir(speechbrain_dir):
        return False
    existing = set(os.listdir(speechbrain_dir))
    if not all(f in existing for f in SPEECHBRAIN_REQUIRED_FILES):
        return False
    return any(enc in existing for enc in SPEECHBRAIN_LABEL_ENCODER)


def migrate_speechbrain_from_hf_cache(speechbrain_dir: str) -> bool:
    """从 ~/.cache/huggingface 提取 SpeechBrain 情感模型到 savedir（HF 缓存已有时）"""
    hf_hub = os.path.join(DEFAULT_CACHE_ROOT, "huggingface", "hub")
    if not os.path.isdir(hf_hub):
        return False
    # HF 缓存目录格式: models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP
    cache_name = "models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP"
    cache_path = os.path.join(hf_hub, cache_name)
    if not os.path.isdir(cache_path) or not os.listdir(cache_path):
        return False
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(speechbrain_dir, exist_ok=True)
        # 从 HF 缓存/网络 materialize 到 local_dir（HF 仓库含 label_encoder.txt，hyperparams 已引用）
        snapshot_download(
            repo_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            local_dir=speechbrain_dir,
            local_dir_use_symlinks=False,
        )
        return _speechbrain_savedir_complete(speechbrain_dir)
    except Exception as e:
        print(f"    从 HF 缓存迁移 SpeechBrain 失败: {e}")
        return False


def run_model_check(cache_dir: str) -> dict:
    """全面检查离线模型是否完整，返回各模块状态 {"paraformer": bool, "sensevoice": bool, "speechbrain": bool, "silero": bool}"""
    result = {}
    cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir)))

    # Paraformer
    result["paraformer"], _ = _paraformer_models_exist(cache_dir)

    # SenseVoice（HuggingFace hub 内需有 SenseVoice 相关缓存）
    hf_hub = os.path.join(cache_dir, "huggingface", "hub")
    result["sensevoice"] = bool(os.path.isdir(hf_hub) and os.listdir(hf_hub))

    # SpeechBrain（固定路径）
    speechbrain_dir = os.path.join(PROJECT_ROOT, "pretrained_models", "emotion-recognition-wav2vec2-IEMOCAP")
    result["speechbrain"] = _speechbrain_savedir_complete(speechbrain_dir) and _wav2vec2_base_local_complete(speechbrain_dir)

    # Silero
    silero_path = os.path.join(PROJECT_ROOT, "silero-vad")
    result["silero"] = os.path.isdir(silero_path) and bool(os.listdir(silero_path))

    return result


def main():
    parser = argparse.ArgumentParser(description="预下载语音情感识别所需模型")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="覆盖默认缓存目录（默认与 server 一致: pretrained_models/model_cache 或 MODEL_CACHE_DIR）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查离线模型是否完整，不下载；缺失时打印提示",
    )
    parser.add_argument("--paraformer", action="store_true", default=True, help="下载 Paraformer 相关模型（默认开启）")
    parser.add_argument("--no-paraformer", action="store_false", dest="paraformer")
    parser.add_argument("--sensevoice", action="store_true", default=True, help="下载 SenseVoice 模型（默认开启）")
    parser.add_argument("--no-sensevoice", action="store_false", dest="sensevoice")
    parser.add_argument("--speechbrain", action="store_true", default=True, help="下载 SpeechBrain 情感模型（默认开启）")
    parser.add_argument("--no-speechbrain", action="store_false", dest="speechbrain")
    parser.add_argument("--silero", action="store_true", default=True, help="下载 Silero VAD（默认开启）")
    parser.add_argument("--no-silero", action="store_false", dest="silero")
    parser.add_argument(
        "--move",
        action="store_true",
        help="从 ~/.cache 迁移时使用移动而非复制（节省空间，原缓存将被清空）",
    )
    args = parser.parse_args()

    cache_dir = _get_model_cache_dir(args.cache_dir)
    print(f"模型缓存目录: {cache_dir}")

    # --check-only: 仅检查，不下载
    if args.check_only:
        status = run_model_check(cache_dir)
        all_ok = all(status.values())
        print("\n--- 离线模型检查结果 ---")
        for name, ok in status.items():
            print(f"  {name}: {'✓ 完整' if ok else '✗ 缺失'}")
        if not all_ok:
            missing = [k for k, v in status.items() if not v]
            print("\n缺失模型，请运行以下命令下载：")
            print("  python scripts/download_models.py")
            print("或分别执行: python scripts/download_models.py --paraformer --sensevoice --speechbrain --silero")
        else:
            print("\n✓ 所有模型已完整，启动服务时无需联网。")
        sys.exit(0 if all_ok else 1)

    os.makedirs(cache_dir, exist_ok=True)

    # 优先从 ~/.cache 迁移已有模型
    migrated = migrate_from_default_cache(cache_dir, use_move=args.move)
    if any(migrated.values()):
        print("\n--- 从 ~/.cache 迁移已有模型 ---")
        for name, ok in migrated.items():
            if ok:
                print(f"  ✓ {name} 已迁移到 {cache_dir}")
    else:
        print("\n未检测到 ~/.cache 下的已有模型，将按需下载。")

    # 设置缓存目录，供后续下载使用
    os.environ["MODELSCOPE_CACHE"] = cache_dir

    # 1. ModelScope 模型（Paraformer 模式）
    if args.paraformer:
        ms_ok, ms_path = _paraformer_models_exist(cache_dir)
        if migrated.get("modelscope") or ms_ok:
            print("\n--- Paraformer 相关模型 (ModelScope) ---")
            if ms_path:
                print(f"  已存在于 {ms_path}，跳过下载")
            else:
                print("  已存在，跳过下载")
        else:
            print("\n--- 下载 Paraformer 相关模型 (ModelScope) ---")
            from modelscope import snapshot_download

            for model_id in ["iic/" + m for m in MODELSCOPE_IIC_MODELS]:
                print(f"  下载: {model_id}")
                try:
                    snapshot_download(model_id=model_id, cache_dir=cache_dir)
                    print(f"  ✓ 完成")
                except Exception as e:
                    print(f"  ✗ 失败: {e}")

    # 2. SenseVoice（HuggingFace）
    if args.sensevoice:
        hf_cache = os.path.join(cache_dir, "huggingface")
        hf_hub = os.path.join(hf_cache, "hub")
        sensevoice_cached = os.path.isdir(hf_hub) and bool(os.listdir(hf_hub))
        if migrated.get("huggingface") or sensevoice_cached:
            print("\n--- SenseVoice (HuggingFace) ---")
            print("  已存在，跳过下载")
        else:
            print("\n--- 下载 SenseVoice (HuggingFace) ---")
            try:
                from huggingface_hub import snapshot_download as hf_snapshot_download
                os.makedirs(hf_cache, exist_ok=True)
                os.environ["HF_HOME"] = hf_cache
                os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub
                hf_snapshot_download(
                    repo_id="iic/SenseVoiceSmall",
                    revision="master",
                    cache_dir=hf_hub,
                )
                print("  ✓ 完成")
            except Exception as e:
                print(f"  ✗ 失败: {e}")

    # 3. SpeechBrain 情感模型（与 server 一致：pretrained_models/emotion-recognition-wav2vec2-IEMOCAP）
    if args.speechbrain:
        print("\n--- SpeechBrain 情感模型 ---")
        speechbrain_dir = os.path.join(PROJECT_ROOT, "pretrained_models", "emotion-recognition-wav2vec2-IEMOCAP")
        os.makedirs(speechbrain_dir, exist_ok=True)

        if _speechbrain_savedir_complete(speechbrain_dir):
            print(f"  已存在（含 {len(SPEECHBRAIN_REQUIRED_FILES)} 个必需文件）")
            if not _wav2vec2_base_local_complete(speechbrain_dir):
                if _ensure_wav2vec2_base_local(speechbrain_dir):
                    print("  ✓ 已补充 wav2vec2-base（离线必需）")
                else:
                    print("  ⚠ 无法下载 facebook/wav2vec2-base，离线加载可能失败")
            else:
                print("  跳过下载")
        else:
            # 优先从 ~/.cache/huggingface 迁移（Ubuntu 上 HF 可能已缓存）
            if migrate_speechbrain_from_hf_cache(speechbrain_dir):
                print("  已从 ~/.cache/huggingface 迁移到 pretrained_models")
                _ensure_wav2vec2_base_local(speechbrain_dir)
            else:
                # 使用与 server/models.py 完全一致的方式下载
                try:
                    os.environ["HF_HUB_OFFLINE"] = "0"  # 允许下载
                    sys.modules["speechbrain.integrations.nlp.flair_embeddings"] = None
                    # torchaudio 2.9+ 移除了 list_audio_backends，SpeechBrain 需要此函数
                    import torchaudio
                    if not hasattr(torchaudio, "list_audio_backends"):
                        torchaudio.list_audio_backends = lambda: ["soundfile"]
                    from speechbrain.inference.interfaces import foreign_class
                    os.chdir(PROJECT_ROOT)
                    foreign_class(
                        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                        pymodule_file="custom_interface.py",
                        classname="CustomEncoderWav2vec2Classifier",
                        savedir=speechbrain_dir,
                        run_opts={"device": "cpu"},
                    )
                    if _speechbrain_savedir_complete(speechbrain_dir):
                        print("  ✓ 完成（与 Windows savedir 一致）")
                    else:
                        print("  ✓ 下载完成")
                    _ensure_wav2vec2_base_local(speechbrain_dir)
                except Exception as e:
                    print(f"  foreign_class 失败: {e}")
                    # 降级：直接用 HF snapshot_download
                    try:
                        from huggingface_hub import snapshot_download
                        snapshot_download(
                            repo_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                            local_dir=speechbrain_dir,
                            local_dir_use_symlinks=False,
                        )
                        print("  ✓ 已通过 huggingface_hub 下载")
                        _ensure_wav2vec2_base_local(speechbrain_dir)
                    except Exception as e2:
                        print(f"  ✗ 失败: {e2}")

    # 4. Silero VAD（下载到 cache_dir/torch_hub，或项目 silero-vad/）
    if args.silero:
        print("\n--- Silero VAD ---")
        silero_path = os.path.join(PROJECT_ROOT, "silero-vad")
        if migrated.get("silero") or (os.path.exists(silero_path) and os.path.isdir(silero_path)):
            print(f"  已存在: {silero_path}，跳过")
        else:
            try:
                import torch
                torch_hub_cache = os.path.join(cache_dir, "torch_hub")
                os.makedirs(torch_hub_cache, exist_ok=True)
                torch.hub.set_dir(torch_hub_cache)
                vad_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    source="github",
                    skip_validation=True,
                    trust_repo=True,
                    onnx=False,
                )
                # 复制到项目 silero-vad/ 以便 source="local" 加载
                hub_dir = torch.hub.get_dir()
                src = os.path.join(hub_dir, "snakers4_silero-vad_master")
                if os.path.isdir(src):
                    import shutil
                    if os.path.exists(silero_path):
                        shutil.rmtree(silero_path)
                    shutil.copytree(src, silero_path)
                    print(f"  ✓ 已复制到 {silero_path}")
                else:
                    print(f"  ✓ 已缓存到 {hub_dir}")
            except Exception as e:
                print(f"  ✗ 失败: {e}")

    print("\n" + "=" * 50)
    print("预下载完成。")
    print("Paraformer/SenseVoice 启动时设置: export MODEL_CACHE_DIR=<cache_dir>")
    print(f"  当前 cache_dir: {cache_dir}")
    print("SpeechBrain 与 Silero 使用固定路径，见 server/DEPLOY.md")
    print("=" * 50)


if __name__ == "__main__":
    main()
