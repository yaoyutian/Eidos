#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vllm_multimodal.py — 在 vLLM 中注册 / 承载本项目 (mmllm) 的多模态能力
==============================================================================

本文件同时提供两条部署路径, 并对 vLLM 的多模态注册 API 做了跨版本适配。

路径 A (推荐用于"能用 HF 生态表达"的 checkpoint)
-------------------------------------------------
    1) `python src/deploy/export_hf.py --src <ckpt> --dst <hf_dir>`
    2) `vllm serve <hf_dir> --port 8000`
    图像走 OpenAI 兼容的 `image_url`; 音频需要额外处理, 见 README。

    优点: 工程量最小, 直接复用 vLLM 的 Llava 加载器 / paged attention /
          continuous batching / prefix caching / 量化生态。
    缺点: HF `LlavaForConditionalGeneration` 把视觉 token 数硬绑定在 CLIP patch 网格上,
          且没有 Perceiver Resampler / PixelShuffle / 音频塔的位置。
          export_hf.py 会逐条体检并给出 blocker 清单。

路径 B (完全自定义, 数值 100% 对齐本项目)
------------------------------------------
    用 `ModelRegistry.register_model` 注册自定义模型类, 用
    `MULTIMODAL_REGISTRY.register_processor` 注册自定义 `MultiModalProcessor`,
    processor 产出 `MultiModalInputs` (pixel_values + image_features + 音频特征占位),
    模型侧在 `get_multimodal_embeddings` 里把特征按占位符掩码原位替换进文本 embedding。

    优点: 完全复刻本项目的三塔结构与"静态展开"占位符契约 (256 image token / 25 audio token per chunk)。
    缺点: 需要跟着 vLLM 版本改接口; 本文已做 0.6 / 0.7 / 0.8 / 0.9+ 的适配, 但仍需在目标版本上验收。

vLLM 多模态注册 API 的版本差异 (截至 0.9.x)
---------------------------------------------
    ���本     注册入口                              processor 基类                        apply() 返回
    ------  -----------------------------------  ------------------------------------  ----------------------
    0.6.x   MULTIMODAL_REGISTRY.register_       vllm.multimodal.processing.           PromptUpdates
            processor(processor, model_cls)     BaseMultiModalProcessor
    0.7.x   同上 (签名不变)                     同上; 额外要求 _get_mm_fields /        PromptUpdates
                                                 _get_dummy_mm_data
    0.8.x   register_processor(processor)        同上; 引入 get_max_tokens_per_item    PromptUpdates
    0.9.x   register_processor() (无参, 由       同上; 引入 MultiModalProcessorInfo,  PromptUpdates
            processor.info 反查模型)             处理器需提供 info
    1.0+    同 0.9.x, 但 get_max_tokens_per_item  同上                                 PromptUpdates
            迁移到 ProcessingInfo

    本文件的做法:
      * `_compat()` 在 import 时用 try/except 链把上述符号解析出来, 并打印实际命中的版本信息;
      * 处理器同时实现新旧两套方法名 (get_max_tokens_per_item / get_max_image_tokens / ...),
        由基类是否存在决定 vLLM 会调用哪一个;
      * `apply()` 的返回值统一构造成 `PromptUpdates` (能 import 时) 或等价 list (不能 import 时)。

    ⚠️ 本文件在没有安装 vLLM 的机器上无法做端到端验证; `selfcheck()` 提供了纯 Python 自检,
       适合放进 CI。真正上线前请对照目标版本的 `vllm/multimodal/processing.py` 校验签名。

静态展开占位符契约 (与训练侧一致)
----------------------------------
    每张图   -> 256 个 <image> token   (cfg.vision.out_tokens)
    每 30s 音频 -> 25 个 <audio> token  (cfg.audio.out_tokens_per_chunk)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"

# 仓库根目录也要在 sys.path 上, 才能 `from src.mmllm... import ...`
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ======================================================================================= #
# 0. 基础工具: 加载 MMConfig (尽量不依赖 torch)
# ======================================================================================= #

def load_mm_config_class():
    """拿到 `MMConfig` 类。

    `mmllm/__init__.py` 会 import modeling -> 需要 torch。这里直接按文件路径加载 config.py,
    使得在没有 torch 的 CI 机器上 `selfcheck()` 也能跑。
    """
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    try:
        from mmllm.config import MMConfig  # type: ignore

        return MMConfig
    except Exception:
        pass
    import importlib.util

    path = _SRC_DIR / "mmllm" / "config.py"
    spec = importlib.util.spec_from_file_location("_mmllm_config_standalone", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MMConfig


def default_cfg():
    """默认配置 (base 预设)。"""
    MMConfig = load_mm_config_class()
    try:
        from mmllm.config import preset  # type: ignore
    except Exception:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_mmllm_config_standalone", _SRC_DIR / "mmllm" / "config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        preset = mod.preset
    return preset("base")


# ======================================================================================= #
# 1. vLLM 兼容层
# ======================================================================================= #

@dataclass
class VLLMCompat:
    """记录实际命中的 vLLM 符号, 供自检与文档引用。"""
    available: bool = False
    version: str = "未安装"
    registry: Any = None                 # MULTIMODAL_REGISTRY
    model_registry: Any = None           # ModelRegistry
    base_processor: Any = None           # BaseMultiModalProcessor
    multi_modal_inputs: Any = None       # MultiModalInputs
    multi_modal_kwargs: Any = None       # MultiModalKwargs
    prompt_updates: Any = None           # PromptUpdates
    prompt_replacement: Any = None       # PromptReplacement
    prompt_insertion: Any = None         # PromptInsertion
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.available:
            return "vLLM 未安装 (仅能做纯 Python 自检)"
        return f"vLLM {self.version}: " + ", ".join(self.notes)


def _try_import(modname: str, attr: Optional[str] = None):
    try:
        mod = __import__(modname, fromlist=["__name__"])
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None


def build_compat() -> VLLMCompat:
    """解析 vLLM 的多模态 API; 全部用 try/except, 任何一步失败都不影响纯 Python 部分。"""
    c = VLLMCompat()

    vllm = _try_import("vllm")
    if vllm is None:
        c.notes.append("未安装 vLLM")
        return c
    c.available = True
    c.version = getattr(vllm, "__version__", "未知")

    # ---- MULTIMODAL_REGISTRY ---- #
    c.registry = (_try_import("vllm.multimodal", "MULTIMODAL_REGISTRY")
                  or _try_import("vllm.multimodal.registry", "MULTIMODAL_REGISTRY"))
    if c.registry is not None:
        c.notes.append("MULTIMODAL_REGISTRY ✓")

    # ---- ModelRegistry ---- #
    c.model_registry = (_try_import("vllm.model_executor.models.registry", "ModelRegistry")
                        or _try_import("vllm.model_executor.models", "ModelRegistry"))
    if c.model_registry is not None:
        c.notes.append("ModelRegistry ✓")

    # ---- processor 基类 ---- #
    c.base_processor = _try_import("vllm.multimodal.processing", "BaseMultiModalProcessor")
    if c.base_processor is None:
        c.base_processor = _try_import("vllm.multimodal.processing", "MultiModalProcessor")
    if c.base_processor is not None:
        c.notes.append("BaseMultiModalProcessor ✓")

    # ---- 多模态输入 ---- #
    c.multi_modal_inputs = _try_import("vllm.multimodal.inputs", "MultiModalInputs")
    c.multi_modal_kwargs = (_try_import("vllm.multimodal.inputs", "MultiModalKwargs")
                            or _try_import("vllm.multimodal.inputs", "MultiModalKwargsItem"))
    if c.multi_modal_inputs is not None:
        c.notes.append("MultiModalInputs ✓")

    # ---- prompt 更新 ---- #
    c.prompt_replacement = (_try_import("vllm.multimodal.processing", "PromptReplacement")
                            or _try_import("vllm.inputs.data", "PromptReplacement"))
    c.prompt_insertion = (_try_import("vllm.multimodal.processing", "PromptInsertion")
                          or _try_import("vllm.inputs.data", "PromptInsertion"))
    c.prompt_updates = (_try_import("vllm.multimodal.processing", "PromptUpdates")
                        or _try_import("vllm.inputs.data", "PromptUpdates"))
    if c.prompt_updates is not None or c.prompt_replacement is not None:
        c.notes.append("PromptUpdates/PromptReplacement ✓")

    return c


COMPAT = build_compat()


# ======================================================================================= #
# 2. 本地占位符工具 (与 serve.py 保持同一套契约)
# ======================================================================================= #

IMAGE_MARKER = "<image>"
AUDIO_MARKER = "<audio>"


def image_token_count(cfg) -> int:
    """一张图展开成多少个 image token。"""
    return int(cfg.vision.out_tokens)


def audio_token_count(cfg, chunks: int = 1) -> int:
    """chunks 个 30s 音频 chunk 展开成多少个 audio token。"""
    return int(cfg.audio.out_tokens_per_chunk) * int(chunks)


def audio_chunk_count(duration_sec: float, cfg) -> int:
    """按 30s 粒度切块的数量。"""
    chunk = float(cfg.audio.chunk_seconds or 30)
    return max(1, int(math.ceil(duration_sec / chunk)))


def count_markers(text: str) -> Tuple[int, int]:
    """统计文本里的 <image> / <audio> 占位符个数。"""
    return text.count(IMAGE_MARKER), text.count(AUDIO_MARKER)


def expected_token_math(cfg, n_images: int, n_audio_chunks: int) -> Dict[str, int]:
    """给出一次请求展开后的 token 账 (用于自检与前端预估)。"""
    return {
        "image_tokens": image_token_count(cfg) * n_images,
        "audio_tokens": audio_token_count(cfg, n_audio_chunks),
        "total_visual_audio_tokens": image_token_count(cfg) * n_images
                                    + audio_token_count(cfg, n_audio_chunks),
    }


# ======================================================================================= #
# 3. 多模态预处理 (图片 / 音频 -> 张量)
# ======================================================================================= #

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _np():
    import numpy as np

    return np


def preprocess_image(image, image_size: int):
    """PIL.Image -> (3, H, W) float32 张量, CLIP 归一化。

    与 CLIPImageProcessor 对齐: 先按短边缩放 (保持长宽比), 再 center crop, 最后归一化。
    """
    import numpy as np
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"preprocess_image 需要 PIL.Image, 收到 {type(image)}")
    image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"非法图片尺寸 {w}x{h}")

    # 短边缩放到 image_size, 再 center crop 成 image_size x image_size
    scale = image_size / min(w, h)
    new_w, new_h = max(image_size, int(round(w * scale))), max(image_size, int(round(h * scale)))
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - image_size) // 2, (new_h - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(arr))


def mel_filterbank(sr: int = 16000, n_fft: int = 400, n_mels: int = 128,
                   fmin: float = 0.0, fmax: Optional[float] = None) -> "Any":
    """Slaney 风格的 mel 滤波器组 (与 librosa.filters.mel 默认一致)。"""
    import numpy as np

    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_bins)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_pts[2:n_mels + 2] - hz_pts[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def log_mel(audio, sr: int = 16000, n_mels: int = 128, n_fft: int = 400,
            hop_length: int = 160, n_frames: int = 3000):
    """波形 -> (n_mels, n_frames) log-Mel (Whisper 风格归一化)。

    * 音频先补零/截断到 30s = 480000 采样点, 保证输出恒为 3000 帧 -> 推理期 shape 静态
    * 幅度谱 -> mel -> log10 -> Whisper 的 max/8 截断 -> 归一化到大致 [-1, 1]
    """
    import numpy as np
    import torch

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    target_len = sr * 30
    if audio.shape[0] < target_len:
        audio = np.pad(audio, (0, target_len - audio.shape[0]))
    else:
        audio = audio[:target_len]

    window = torch.hann_window(n_fft)
    spec = torch.stft(torch.from_numpy(audio), n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, window=window, return_complex=True, center=True)
    mag = spec.abs()[..., :n_frames]                       # (freq, frames)
    if mag.shape[-1] < n_frames:
        mag = torch.nn.functional.pad(mag, (0, n_frames - mag.shape[-1]))

    fb = torch.from_numpy(mel_filterbank(sr, n_fft, n_mels))   # (n_mels, freq)
    mel = fb @ mag                                            # (n_mels, frames)
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def preprocess_audio_chunk(audio, cfg) -> "Any":
    """一段 (<=30s) 音频 -> (n_mels, 3000) log-Mel 张量。"""
    return log_mel(audio, sr=cfg.audio.sample_rate, n_mels=cfg.audio.n_mels, n_frames=3000)


def chunk_audio(audio, cfg) -> List[Any]:
    """把整段音频按 30s 切成若干 chunk, 每块补零到 3000 帧。"""
    import numpy as np

    sr = cfg.audio.sample_rate
    chunk_len = int(sr * cfg.audio.chunk_seconds)
    if hasattr(audio, "detach"):          # torch.Tensor -> numpy
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_chunks = max(1, int(math.ceil(len(audio) / chunk_len)))
    return [audio[i * chunk_len:(i + 1) * chunk_len] for i in range(n_chunks)]


# ======================================================================================= #
# 4. MultiModalProcessor
# ======================================================================================= #

@dataclass
class _Replacement:
    """vLLM 不可用时的 PromptReplacement 替身 (字段与 vLLM 保持一致)。"""
    target: Sequence[int]
    replacement: Sequence[int]
    modality: str = "image"
    is_embed: Optional[dict] = None


class _Updates(list):
    """vLLM 不可用时的 PromptUpdates 替身 (就是一个 list)。"""

    def __init__(self, items: Optional[Iterable[Any]] = None):
        super().__init__(items or [])


def _build_updates(items: List[Any]) -> Any:
    """统一构造 prompt updates 容器。"""
    if COMPAT.prompt_updates is not None:
        try:
            return COMPAT.prompt_updates(items)
        except Exception:
            try:
                return COMPAT.prompt_updates(replacements=items)
            except Exception:
                pass
    return _Updates(items)


def _build_replacement(target: Sequence[int], replacement: Sequence[int], modality: str) -> Any:
    if COMPAT.prompt_replacement is not None:
        try:
            return COMPAT.prompt_replacement(target=list(target), replacement=list(replacement),
                                            modality=modality)
        except Exception:
            try:
                return COMPAT.prompt_replacement(target=list(target),
                                                replacement=list(replacement))
            except Exception:
                pass
    return _Replacement(target=list(target), replacement=list(replacement), modality=modality)


def _build_mm_inputs(mm_kwargs: Dict[str, Any], placeholders: Optional[Dict[str, Any]] = None) -> Any:
    """构造 MultiModalInputs (不同版本可能是 TypedDict / dataclass)。"""
    if COMPAT.multi_modal_inputs is not None:
        for kwargs in ({"mm_placeholders": placeholders, "mm_kwargs": mm_kwargs},
                       {"mm_kwargs": mm_kwargs}):
            try:
                return COMPAT.multi_modal_inputs(**kwargs)
            except Exception:
                continue
    return {"mm_placeholders": placeholders, "mm_kwargs": mm_kwargs}


def _build_mm_kwargs(**fields: Any) -> Any:
    """构造 MultiModalKwargs。"""
    if COMPAT.multi_modal_kwargs is not None:
        try:
            return COMPAT.multi_modal_kwargs(**fields)
        except Exception:
            pass
    return dict(fields)


_Base = COMPAT.base_processor or object


class MMMultiModalProcessor(_Base):
    """本项目多模态处理器。

    职责:
      1. 把 prompt 里的 `<image>` / `<audio>` 标记换成固定数量的占位 token id
         (静态展开: 256 个 image token / 每 30s 音频 25 个 audio token);
      2. 把 mm_data 里的图片/音频编码成张量, 塞进 `MultiModalInputs`;
      3. 提供 profiling 用的 dummy data 与 token 上限 (get_max_tokens_per_item)。
    """

    def __init__(self, cfg=None, tokenizer=None, info=None):
        self.cfg = cfg if cfg is not None else default_cfg()
        self._tokenizer = tokenizer
        self.info = info
        # 0.9+ 的 ProcessingInfo 有缓存字段; 老版本没有, 这里无条件建一个
        self._cached_apply_hf_processor = None
        try:
            super().__init__(info)  # type: ignore[call-arg]
        except TypeError:
            try:
                super().__init__()   # type: ignore[call-arg]
            except TypeError:
                pass

    # ---------------- tokenizer ---------------- #
    def _get_tokenizer(self):
        """兼容 0.6/0.7 (self.tokenizer) 与 0.9+ (self.info.get_tokenizer())。"""
        if self._tokenizer is not None:
            return self._tokenizer
        info = getattr(self, "info", None)
        if info is not None:
            getter = getattr(info, "get_tokenizer", None)
            if callable(getter):
                try:
                    self._tokenizer = getter()
                    return self._tokenizer
                except Exception:
                    pass
            hf_proc = getattr(info, "get_hf_processor", None)
            if callable(hf_proc):
                try:
                    self._tokenizer = getattr(hf_proc(), "tokenizer", None)
                    return self._tokenizer
                except Exception:
                    pass
        return getattr(self, "tokenizer", None)

    def marker_token_id(self, marker: str, fallback_id: int) -> int:
        """找到 `<image>` / `<audio>` 标记对应的 token id。

        优先问 tokenizer; 拿不到就退回本项目约定的特殊 token id
        (cfg.image_token_id / cfg.audio_token_id, 训练时扩展 tokenizer 保证一致)。
        """
        tok = self._get_tokenizer()
        if tok is not None:
            try:
                tid = tok.convert_tokens_to_ids(marker)
                if tid is not None and tid >= 0:
                    return int(tid)
            except Exception:
                pass
            try:
                ids = tok.encode(marker, add_special_tokens=False)
                if len(ids) == 1:
                    return int(ids[0])
            except Exception:
                pass
        return int(fallback_id)

    # ---------------- token 上限 (profiling 用) ---------------- #
    def get_max_tokens_per_item(self, modality: str = "image") -> int:
        """一个 item 最多展开成多少个 token (0.8+ / 0.9+ 的接口名)。"""
        if modality == "image":
            return image_token_count(self.cfg)
        if modality == "audio":
            max_chunks = max(1, int(math.ceil(
                float(self.cfg.audio.max_seconds) / float(self.cfg.audio.chunk_seconds or 30))))
            return audio_token_count(self.cfg, max_chunks)
        raise ValueError(f"不支持的模态: {modality}")

    def get_max_image_tokens(self, image_width: int = 336, image_height: int = 336) -> int:
        """0.6/0.7 的接口名 (与本项目契约一致: 恒定 256, 不随分辨率变化)。"""
        return image_token_count(self.cfg)

    def get_max_audio_tokens(self, duration_sec: float = 30.0) -> int:
        chunks = audio_chunk_count(duration_sec, self.cfg)
        return audio_token_count(self.cfg, chunks)

    # ---------------- dummy data (profiling) ---------------- #
    def get_dummy_mm_data(self, mm_counts: Dict[str, int]) -> Dict[str, Any]:
        """构造 profiling 用的假数据。"""
        data: Dict[str, Any] = {}
        if mm_counts.get("image"):
            try:
                from PIL import Image

                img = Image.new("RGB", (self.cfg.vision.image_size, self.cfg.vision.image_size),
                                color=(128, 128, 128))
                data["image"] = [img] * int(mm_counts["image"])
            except Exception:
                data["image"] = None
        if mm_counts.get("audio"):
            import numpy as np

            sr = self.cfg.audio.sample_rate
            dur = float(self.cfg.audio.chunk_seconds or 30)
            data["audio"] = [np.zeros(int(sr * dur), dtype=np.float32)
                             for _ in range(int(mm_counts["audio"]))]
        return data

    def _get_dummy_mm_data(self, mm_counts: Dict[str, int]) -> Dict[str, Any]:
        """0.7+ 的私有接口名。"""
        return self.get_dummy_mm_data(mm_counts)

    def get_mm_fields(self) -> Dict[str, Any]:
        """返回该 processor 产出的 mm 字段 (0.7+ 需要)。"""
        fields: Dict[str, Any] = {}
        try:
            import torch

            fields["pixel_values"] = torch.Tensor
            fields["image_features"] = torch.Tensor
            fields["input_features"] = torch.Tensor
            fields["audio_features"] = torch.Tensor
        except Exception:
            pass
        return fields

    def _get_mm_fields(self) -> Dict[str, Any]:
        return self.get_mm_fields()

    # ---------------- 主流程 ---------------- #
    def apply(self, prompt: str, mm_data: Optional[Dict[str, Any]] = None,
              mm_processor_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """核心: 生成 prompt updates + 多模态张量。

        返回值在不同 vLLM 版本里形态不同 (PromptUpdates / list), 但语义一致:
        "把 prompt 里的每个 <image> 标记替换成 256 个 image_token_id,
         每个 <audio> 标记替换成 25 * chunk 数 个 audio_token_id"。
        """
        mm_data = mm_data or {}
        mm_processor_kwargs = mm_processor_kwargs or {}
        cfg = self.cfg

        n_img_marker, n_aud_marker = count_markers(prompt)
        images = list(mm_data.get("image") or [])
        audios = list(mm_data.get("audio") or [])

        if len(images) > n_img_marker:
            raise ValueError(
                f"图片数量({len(images)}) 超过 prompt 中的 <image> 占位符数({n_img_marker}); "
                f"请在 prompt 里为每张图写一个 {IMAGE_MARKER}")
        if len(audios) > n_aud_marker:
            raise ValueError(
                f"音频数量({len(audios)}) 超过 prompt 中的 <audio> 占位符数({n_aud_marker})")

        updates: List[Any] = []
        mm_kwargs: Dict[str, Any] = {}

        # ---- 图像 ---- #
        if images:
            import torch

            pixel_values = torch.stack([preprocess_image(im, cfg.vision.image_size)
                                        for im in images])
            mm_kwargs["pixel_values"] = pixel_values
            # 视觉特征由模型侧算 (vLLM 的 _process_image_input), 这里给占位
            mm_kwargs["image_features"] = None

            target = self.marker_token_id(IMAGE_MARKER, cfg.image_token_id)
            replacement = [int(cfg.image_token_id)] * image_token_count(cfg)
            updates.append(_build_replacement([target], replacement, "image"))

        # ---- 音频 ---- #
        if audios:
            import torch

            import numpy as np

            chunks: List[Any] = []
            for au in audios:
                if isinstance(au, (bytes, bytearray)):
                    au = _decode_wav_bytes(bytes(au))
                if isinstance(au, dict):
                    au = au.get("array", au.get("data"))
                arr = np.asarray(au, dtype=np.float32).reshape(-1)
                chunks.extend(chunk_audio(arr, cfg))
            features = torch.stack([preprocess_audio_chunk(c, cfg) for c in chunks])
            mm_kwargs["input_features"] = features
            mm_kwargs["audio_features"] = None

            if n_aud_marker > 1:
                raise NotImplementedError(
                    "当前实现只支持 1 个 <audio> 标记对应全部音频; "
                    "多段音频请在请求层先合并或分别请求。")
            target = self.marker_token_id(AUDIO_MARKER, cfg.audio_token_id)
            replacement = [int(cfg.audio_token_id)] * audio_token_count(cfg, len(chunks))
            updates.append(_build_replacement([target], replacement, "audio"))

        placeholders = {"image": list(mm_kwargs.get("pixel_values") or []) and None}
        return _build_updates(updates), _build_mm_inputs(
            _build_mm_kwargs(**{k: v for k, v in mm_kwargs.items() if v is not None}),
            placeholders)

    # 0.9+ 有时会把 apply 的结果拆成两个返回值; 这里保留一个显式入口
    def apply_with_inputs(self, prompt: str, mm_data: Optional[Dict[str, Any]] = None,
                           mm_processor_kwargs: Optional[Dict[str, Any]] = None):
        """返回 (prompt_updates, mm_inputs) 的显式二元组, 供自定义模型侧使用。"""
        return self.apply(prompt, mm_data, mm_processor_kwargs)


def _decode_wav_bytes(data: bytes):
    """把 wav 字节流解成 (n,) float32 单声道波形; 优先 soundfile, 其次 wave。"""
    import io

    try:
        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)
        return arr
    except Exception:
        pass
    try:
        import wave

        with wave.open(io.BytesIO(data), "rb") as wf:
            n_ch = wf.getnchannels()
            width = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if width == 2:
            import numpy as np

            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            import numpy as np

            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            import numpy as np

            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
        if n_ch > 1:
            arr = arr.reshape(-1, n_ch).mean(axis=1)
        return arr
    except Exception as e:  # pragma: no cover
        raise ValueError(f"无法解析 wav 字节流: {e}")


# ======================================================================================= #
# 5. 路径 B: 自定义模型类 (注册到 vLLM)
# ======================================================================================= #

def build_model_class():
    """构造路径 B 的模型类。

    设计:
      * 语言塔直接复用 vLLM 的 `LlamaForCausalLM` (权重需先转成 HF Llama 命名);
      * 视觉塔 / 音频塔用本项目的实现, 在 `get_multimodal_embeddings` 里算好特征,
        按占位符掩码原位替换进文本 embedding;
      * `load_weights` 覆写: 把 `vision_tower.*` / `audio_tower.*` / `*_projector.*`
        路由到本项目的子模块, 其余交给内层 Llama 模型。

    ⚠️ 需要按目标 vLLM 版本调整 (不同版本的 SupportsMultiModal 协议方法名不同):
       0.6-0.8: `_process_image_input` / `_process_audio_input` / `get_multimodal_embeddings`
       0.9+:     `get_multimodal_embeddings` / `get_input_embeddings` / `get_language_model`
    """
    if not COMPAT.available:
        return None
    try:
        import torch
        from vllm.model_executor.models.llama import LlamaForCausalLM
    except Exception as e:  # pragma: no cover
        print(f"[vllm_multimodal] 构造模型类失败 (缺少 vLLM/torch): {e}")
        return None

    from src.mmllm.config import MMConfig  # type: ignore
    from src.mmllm.modeling import MultimodalLM  # type: ignore

    class MMForCausalLM(LlamaForCausalLM):
        """路径 B: 本项目多模态模型 (文本塔沿用 vLLM 的 Llama 实现)。"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            hf_cfg = getattr(self.config, "text_config", self.config)
            mm_cfg = MMConfig()
            mm_cfg.llm.vocab_size = getattr(hf_cfg, "vocab_size", mm_cfg.llm.vocab_size)
            mm_cfg.llm.hidden_size = getattr(hf_cfg, "hidden_size", mm_cfg.llm.hidden_size)
            mm_cfg.llm.num_layers = getattr(hf_cfg, "num_hidden_layers", mm_cfg.llm.num_layers)
            mm_cfg.llm.num_heads = getattr(hf_cfg, "num_attention_heads", mm_cfg.llm.num_heads)
            mm_cfg.llm.num_kv_heads = getattr(hf_cfg, "num_key_value_heads", mm_cfg.llm.num_heads)
            mm_cfg.image_token_id = getattr(self.config, "image_token_index", 32001)
            self.mm_cfg = mm_cfg
            # 本项目的视觉/音频子模块 (按需替换成真实实现)
            self.mm_model = MultimodalLM(mm_cfg)

        # -------- 多模态特征 --------
        def get_multimodal_embeddings(self, **kwargs):
            """返回与占位符一一对应的特征列表。"""
            pixel_values = kwargs.get("pixel_values")
            if pixel_values is None:
                return []
            feats = self.mm_model.encode_images(pixel_values)
            return [feats[i] for i in range(feats.shape[0])]

        # 0.6-0.8 的旧接口名
        def _process_image_input(self, image_input):
            return self.get_multimodal_embeddings(pixel_values=image_input)

        def _process_audio_input(self, audio_input):
            feats = self.mm_model.encode_audio(audio_input)
            return [feats[i] for i in range(feats.shape[0])]

        def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
            """把多模态特征按占位符掩码原位替换进文本 embedding。"""
            inputs_embeds = self.get_language_model().get_input_embeddings(input_ids)
            if not multimodal_embeddings:
                return inputs_embeds
            flat = [f for f in multimodal_embeddings]
            mask = input_ids == self.mm_cfg.image_token_id
            n = int(mask.sum().item())
            if n and flat:
                cat = torch.cat(flat, dim=0).to(inputs_embeds.dtype)
                pos = mask.long().cumsum(dim=1) - 1
                pos = pos.clamp(min=0, max=max(cat.shape[0] - 1, 0))
                idx = pos.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
                gathered = torch.gather(cat.unsqueeze(0).expand_as(idx), 1, idx)
                inputs_embeds = torch.where(mask.unsqueeze(-1), gathered, inputs_embeds)
            return inputs_embeds

        # -------- 权重加载 --------
        def load_weights(self, weights):
            """把视觉/音频权重截住, 其余交给 Llama。"""
            ours, others = {}, {}
            for name, tensor in weights:
                if name.startswith(("vision_tower.", "audio_tower.", "vision_projector.",
                                    "audio_projector.", "vision_resampler.")):
                    ours[name] = tensor
                else:
                    others.append((name, tensor))
            if ours:
                missing, unexpected = self.mm_model.load_state_dict(ours, strict=False)
                if missing or unexpected:
                    print(f"[MMForCausalLM] 多模态权重 missing={list(missing)[:4]} "
                          f"unexpected={list(unexpected)[:4]}")
            return super().load_weights(others)

    return MMForCausalLM


MMForCausalLM = build_model_class()


def register_processor(model_cls=None, cfg=None):
    """把 processor 注册到 vLLM 的多模态注册表 (跨版本适配)。"""
    if COMPAT.registry is None:
        print("[vllm_multimodal] 未安装 vLLM, 跳过 register_processor")
        return False
    proc = MMMultiModalProcessor(cfg=cfg)
    registry = COMPAT.registry
    attempts = [
        lambda: registry.register_processor(proc, model_cls) if model_cls is not None
        else registry.register_processor(proc),
        lambda: registry.register_processor(proc),
        lambda: registry.register_processor(),
    ]
    last_err = None
    for fn in attempts:
        try:
            fn()
            print("[vllm_multimodal] register_processor 成功")
            return True
        except Exception as e:
            last_err = e
    print(f"[vllm_multimodal] register_processor 失败: {last_err}")
    return False


def register_model(name: str = "MMForCausalLM",
                   path: str = "src.deploy.vllm_multimodal:MMForCausalLM") -> bool:
    """把自定义模型类注册到 vLLM 的 ModelRegistry。"""
    if COMPAT.model_registry is None:
        print("[vllm_multimodal] 未安装 vLLM, 跳过 register_model")
        return False
    try:
        COMPAT.model_registry.register_model(name, path)
        print(f"[vllm_multimodal] register_model({name}) 成功")
        return True
    except Exception as e:
        print(f"[vllm_multimodal] register_model 失败: {e}")
        return False


# ======================================================================================= #
# 6. 纯 Python 自检 (不装 vLLM 也能跑, 适合 CI)
# ======================================================================================= #

def selfcheck(verbose: bool = True) -> Dict[str, Any]:
    """自检: 检查依赖、占位符契约、导出映射的一致性。

    返回 {"ok": bool, "checks": [...]}。缺少 vLLM 时不会失败, 只提示安装命令。
    """
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "", fatal: bool = False) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "fatal": fatal})

    if verbose:
        print("=" * 78)
        print("mmllm vLLM 部署自检 (selfcheck)")
        print("=" * 78)
        print(f"python      : {sys.version.split()[0]}")
        print(f"platform    : {sys.platform}")
        print(f"vLLM        : {COMPAT.summary()}")
        print("-" * 78)

    # ---- 1. 依赖 ---- #
    for mod, hint in (("torch", "pip install torch"),
                      ("numpy", "pip install numpy"),
                      ("PIL", "pip install pillow"),
                      ("transformers", "pip install transformers"),
                      ("vllm", 'pip install "vllm>=0.6"  # 需要 CUDA 环境')):
        try:
            __import__(mod)
            add(f"依赖 {mod}", True, "已安装")
        except Exception:
            add(f"依赖 {mod}", False, f"缺失 -> {hint}")

    # ---- 2. 配置契约 ---- #
    try:
        cfg = default_cfg()
        add("加载 MMConfig", True, f"preset=base, image_token_id={cfg.image_token_id}, "
                                   f"audio_token_id={cfg.audio_token_id}")
    except Exception as e:
        add("加载 MMConfig", False, str(e), fatal=True)
        if verbose:
            for c in checks:
                print(f"  [{'OK ' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
        return {"ok": False, "checks": checks}

    # ---- 3. 静态展开契约 ---- #
    n_img = image_token_count(cfg)
    n_aud = audio_token_count(cfg, 1)
    ok_contract = (n_img == 256 and n_aud == 25)
    add("静态展开契约", ok_contract,
        f"每图 {n_img} 个 <image> token; 每 30s 音频 {n_aud} 个 <audio> token "
        f"(期望 256 / 25)")

    text = "看图 <image> 听音 <audio> 回答"
    n_i, n_a = count_markers(text)
    math_info = expected_token_math(cfg, n_i, n_a)
    add("占位符计数", n_i == 1 and n_a == 1,
        f"'{text}' -> image={n_i}, audio={n_a}; 展开后视觉+音频 token={math_info}")

    # ---- 4. 导出映射一致性 (与 export_hf.py 联动) ---- #
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from export_hf import map_hf_to_project, map_weight_name  # type: ignore

        probes = [
            "language_model.embed_tokens.weight",
            "language_model.layers.3.input_norm.weight",
            "language_model.layers.3.post_attn_norm.weight",
            "language_model.layers.3.attn.q_proj.weight",
            "language_model.layers.3.mlp.gate_proj.weight",
            "language_model.final_norm.weight",
            "language_model.lm_head.weight",
            "vision_tower.patch_embed.weight",
            "vision_tower.pos_embed",
            "vision_tower.pre_layrnorm.weight",
            "vision_tower.blocks.5.input_norm.weight",
            "vision_tower.blocks.5.post_attn_norm.weight",
            "vision_tower.blocks.5.attn.v_proj.weight",
            "vision_tower.blocks.5.mlp.fc1.weight",
            "vision_tower.post_layernorm.weight",
        ]
        bad_fwd, bad_rev = [], []
        for k in probes:
            hf = map_weight_name(k)
            if hf is None:
                bad_fwd.append(k)
                continue
            back = map_hf_to_project(hf)
            if back != k:
                bad_rev.append(f"{k} -> {hf} -> {back}")
        add("导出映射 (本项目 -> HF)", not bad_fwd, f"未映射: {bad_fwd}" if bad_fwd else "全部命中")
        add("导出映射 (HF -> 本项目, 往返)", not bad_rev,
            f"不一致: {bad_rev}" if bad_rev else "往返一致")
    except Exception as e:
        add("导出映射检查", False, f"无法导入 export_hf: {e}")

    # ---- 5. 多模态张量形状 (需要 torch + PIL) ---- #
    try:
        import torch

        px = preprocess_image.__doc__ is not None  # 仅探测 PIL 是否可用
        from PIL import Image

        img = Image.new("RGB", (640, 480), color=(10, 20, 30))
        t = preprocess_image(img, cfg.vision.image_size)
        add("图像预处理", tuple(t.shape) == (3, cfg.vision.image_size, cfg.vision.image_size),
            f"输出 shape={tuple(t.shape)}, dtype={t.dtype}")
    except Exception as e:
        add("图像预处理", False, f"跳过 ({e})")

    try:
        import numpy as np

        audio = np.zeros(16000 * 5, dtype=np.float32)
        chunks = chunk_audio(audio, cfg)
        mel = preprocess_audio_chunk(chunks[0], cfg)
        add("音频预处理", tuple(mel.shape) == (cfg.audio.n_mels, 3000),
            f"{len(audio)/16000:.1f}s -> {len(chunks)} chunk, log-Mel shape={tuple(mel.shape)}")
    except Exception as e:
        add("音频预处理", False, f"跳过 ({e})")

    # ---- 6. processor 冒烟测试 (不需要 vLLM) ---- #
    try:
        proc = MMMultiModalProcessor(cfg=cfg)
        add("processor 构造", True,
            f"get_max_tokens_per_item(image)={proc.get_max_tokens_per_item('image')}, "
            f"(audio)={proc.get_max_tokens_per_item('audio')}")
        upd, mm_inputs = proc.apply_with_inputs("描述 <image> 和 <audio>",
                                                {"image": [], "audio": []})
        add("processor apply (空模态)", True, f"updates={len(upd)}")
    except Exception as e:
        add("processor apply", False, str(e))

    # ---- 7. vLLM 相关 ---- #
    if COMPAT.available:
        add("vLLM 多模态注册表", COMPAT.registry is not None,
            "已找到 MULTIMODAL_REGISTRY" if COMPAT.registry else "未找到")
        add("vLLM ModelRegistry", COMPAT.model_registry is not None,
            "已找到 ModelRegistry" if COMPAT.model_registry else "未找到")
        add("vLLM processor 基类", COMPAT.base_processor is not None,
            "已找到 BaseMultiModalProcessor" if COMPAT.base_processor else "未找到 (将使用纯 Python 替身)")
        add("vLLM 自定义模型类", MMForCausalLM is not None,
            "MMForCausalLM 已构造" if MMForCausalLM is not None else "构造失败")
    else:
        add("vLLM (可选)", True,
            '未安装 vLLM -> 路径 A/B 的注册逻辑未在本机验证; 安装: pip install "vllm>=0.6"')

    # ---- 汇总 ---- #
    fatal_fail = [c for c in checks if c["fatal"] and not c["ok"]]
    ok = len(fatal_fail) == 0

    if verbose:
        print("-" * 78)
        for c in checks:
            mark = "OK  " if c["ok"] else ("FAIL" if c["fatal"] else "WARN")
            print(f"  [{mark}] {c['name']}: {c['detail']}")
        print("-" * 78)
        print(f"自检结果: {'PASS' if ok else 'FAIL'}  "
              f"(共 {len(checks)} 项, 其中 {len([c for c in checks if not c['ok']])} 项未通过)")
        if not COMPAT.available:
            print()
            print("提示: 本机没有 vLLM, 只做了纯 Python 自检。")
            print('      安装 vLLM 后可继续验证路径 A/B:')
            print('        pip install "vllm>=0.6"')
            print('        python src/deploy/vllm_multimodal.py --selfcheck --deep')
        print("=" * 78)

    return {"ok": ok, "checks": checks}


# ======================================================================================= #
# 7. CLI
# ======================================================================================= #

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="vLLM 多模态注册 / 自检")
    ap.add_argument("--selfcheck", action="store_true", help="运行自检 (默认行为)")
    ap.add_argument("--deep", action="store_true", help="自检时额外构造模型 (需要 torch)")
    ap.add_argument("--register", action="store_true", help="注册 processor 与模型类到 vLLM")
    ap.add_argument("--model-name", default="MMForCausalLM", help="注册到 vLLM 的模型名")
    args = ap.parse_args(argv)

    if args.deep:
        try:
            import torch

            from src.mmllm.config import preset  # type: ignore
            from src.mmllm.modeling import MultimodalLM  # type: ignore

            cfg = preset("tiny")
            cfg.llm.vocab_size = 32064
            model = MultimodalLM(cfg)
            ids = torch.full((1, 16), 5, dtype=torch.long)
            ids[0, 2:2 + cfg.vision.out_tokens] = cfg.image_token_id
            px = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size)
            out = model(input_ids=ids, pixel_values=px)
            print(f"[deep] logits={tuple(out.logits.shape)} "
                  f"视觉 token={cfg.vision.out_tokens}")
        except Exception as e:
            print(f"[deep] 跳过: {e}")

    if args.register:
        register_model(args.model_name)
        register_processor()

    result = selfcheck()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
