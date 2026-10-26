#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
serve.py — mmllm 多模态推理服务 (FastAPI, OpenAI 兼容)
======================================================

提供三个后端形态 (--backend / MM_BACKEND):

    1. `transformers` : 本地直接加载 `MultimodalLM.from_pretrained(...)` 推理。
       优点: 不依赖 vLLM, 支持完整的 图像 + 音频 (Whisper 塔), 便于功能验证;
       缺点: 无 continuous batching / paged attention, 并发能力弱, 只能当参考实现。
    2. `vllm`        : 走 vLLM。三种子模式 (MM_VLLM_MODE):
         - `subprocess` : 拉起 `vllm serve <model_dir>` 子进程, 本服务做 OpenAI 协议转发 (推荐)
         - `engine`     : 进程内用 `AsyncLLMEngine` / `AsyncLLM` (vLLM <=0.9 / >=0.10)
         - `proxy`      : 转发到已有的 OpenAI 兼容服务 (vLLM / SGLang / 任何兼容网关)
    3. `auto`        : 有 vllm 就用 vllm, 否则回落到 transformers。

接口
----
    POST /v1/chat/completions   OpenAI 兼容对话接口 (支持 image_url / audio / stream)
    POST /generate              裸接口: prompt + images + audios
    GET  /healthz               健康检查
    GET  /metrics               Prometheus 文本格式指标 (TTFT / TPOT / QPS / GPU 显存 ...)

请求侧的多模态预处理 (与训练侧契约严格一致)
------------------------------------------
    图片: 下载/解码 -> 缩放 + center crop 到 `cfg.vision.image_size` -> CLIP 归一化 -> (3,H,W)
    音频: 解码 (soundfile / librosa / wave) -> 16kHz -> 30s 分块 -> log-Mel(n_mels=128, 3000 帧) -> 归一化
    占位符静态展开: 每张图 -> 256 个 image_token_id; 每 30s 音频 -> 25 个 audio_token_id

配置 (命令行 > 环境变量 > 默认值)
--------------------------------
    MM_MODEL_DIR / --model-dir      checkpoint 目录 (含 config.json + model.safetensors)
    MM_BACKEND   / --backend        transformers | vllm | auto
    MM_HOST      / --host           监听地址 (默认 0.0.0.0)
    MM_PORT      / --port           监听端口 (默认 8000)
    MM_MAX_CONCURRENCY / --max-concurrency   并发上限 (默认 8)
    MM_REQUEST_TIMEOUT / --request-timeout   单请求超时秒数 (默认 120)
    MM_MAX_NEW_TOKENS / --max-new-tokens     默认生成长度 (默认 512)
    MM_TEMPERATURE / --temperature   采样温度 (默认 0.0 = 贪心)
    MM_TOP_P     / --top-p          核采样 (默认 0.9)
    MM_DEVICE    / --device         本地推理设备 (cuda / cpu)
    MM_DTYPE     / --dtype          本地推理精度 (bfloat16 / float16 / float32)
    MM_API_KEY   / --api-key        设置后要求 `Authorization: Bearer <key>`
    MM_VLLM_MODE / --vllm-mode      subprocess | engine | proxy
    MM_VLLM_BASE_URL / --vllm-base-url   proxy 模式的后端地址
    MM_VLLM_MODEL  / --vllm-model       请求里使用的模型名 (proxy 模式)
    MM_VLLM_ARGS   / --vllm-args        透传给 `vllm serve` 的额外参数 (字符串)

运行
----
    python src/deploy/serve.py --model-dir runs/exp1 --backend transformers --port 8000
    python src/deploy/serve.py --model-dir out/hf_llava --backend vllm --vllm-mode subprocess
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# 仓库根目录也要在 sys.path 上, 才能 `from src.mmllm... import ...`
# (直接 `python src/deploy/serve.py` 时, sys.path[0] 是 src/deploy 而不是仓库根)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---- 复用 vllm_multimodal 里的媒体预处理 / 占位符契约 (同一份实现, 避免两处漂移) ---- #
try:
    from vllm_multimodal import (  # type: ignore
        AUDIO_MARKER,
        IMAGE_MARKER,
        audio_chunk_count,
        chunk_audio,
        count_markers,
        default_cfg,
        expected_token_math,
        image_token_count,
        load_mm_config_class,
        log_mel,
        preprocess_audio_chunk,
        preprocess_image,
    )
except ImportError:  # pragma: no cover - 以包方式导入时的兜底
    from .vllm_multimodal import (  # type: ignore
        AUDIO_MARKER,
        IMAGE_MARKER,
        audio_chunk_count,
        chunk_audio,
        count_markers,
        default_cfg,
        expected_token_math,
        image_token_count,
        load_mm_config_class,
        log_mel,
        preprocess_audio_chunk,
        preprocess_image,
    )

# ---- 可选依赖 (缺一个都能优雅降级) ---- #
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    Request = Any  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore

    def JSONResponse(*a, **k):  # type: ignore
        raise RuntimeError("fastapi 未安装")

    def PlainTextResponse(*a, **k):  # type: ignore
        raise RuntimeError("fastapi 未安装")

    def StreamingResponse(*a, **k):  # type: ignore
        raise RuntimeError("fastapi 未安装")


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 50 * 1024 * 1024


# ======================================================================================= #
# 1. 配置
# ======================================================================================= #

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """运行期配置。命令行 > 环境变量 > 默认值。"""
    model_dir: str = ""
    backend: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrency: int = 8
    request_timeout: float = 120.0
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    device: str = "cuda"
    dtype: str = "bfloat16"
    api_key: str = ""
    vllm_mode: str = "subprocess"
    vllm_base_url: str = "http://127.0.0.1:8001"
    vllm_model: str = "mmllm"
    vllm_args: str = ""
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int = 8192
    enable_prefix_caching: bool = True
    log_level: str = "info"
    stream_chunk_ms: int = 40

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.model_dir = os.environ.get("MM_MODEL_DIR", s.model_dir)
        s.backend = os.environ.get("MM_BACKEND", s.backend)
        s.host = os.environ.get("MM_HOST", s.host)
        s.port = int(os.environ.get("MM_PORT", s.port))
        s.max_concurrency = int(os.environ.get("MM_MAX_CONCURRENCY", s.max_concurrency))
        s.request_timeout = float(os.environ.get("MM_REQUEST_TIMEOUT", s.request_timeout))
        s.max_new_tokens = int(os.environ.get("MM_MAX_NEW_TOKENS", s.max_new_tokens))
        s.temperature = float(os.environ.get("MM_TEMPERATURE", s.temperature))
        s.top_p = float(os.environ.get("MM_TOP_P", s.top_p))
        s.device = os.environ.get("MM_DEVICE", s.device)
        s.dtype = os.environ.get("MM_DTYPE", s.dtype)
        s.api_key = os.environ.get("MM_API_KEY", s.api_key)
        s.vllm_mode = os.environ.get("MM_VLLM_MODE", s.vllm_mode)
        s.vllm_base_url = os.environ.get("MM_VLLM_BASE_URL", s.vllm_base_url)
        s.vllm_model = os.environ.get("MM_VLLM_MODEL", s.vllm_model)
        s.vllm_args = os.environ.get("MM_VLLM_ARGS", s.vllm_args)
        s.log_level = os.environ.get("MM_LOG_LEVEL", s.log_level)
        return s

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Settings":
        s = cls.from_env()
        for key in ("model_dir", "backend", "host", "port", "max_concurrency",
                    "request_timeout", "max_new_tokens", "temperature", "top_p",
                    "device", "dtype", "api_key", "vllm_mode", "vllm_base_url",
                    "vllm_model", "vllm_args", "log_level"):
            val = getattr(args, key, None)
            if val is not None:
                setattr(s, key, val)
        return s


# ======================================================================================= #
# 2. 指标
# ======================================================================================= #

_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)


class Metrics:
    """极简 Prometheus 指标注册表 (不引入 prometheus_client 依赖)。"""

    def __init__(self, max_samples: int = 4096):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._hist: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "sum": 0.0, "buckets": {b: 0 for b in _BUCKETS},
                     "samples": deque(maxlen=max_samples)})

    @staticmethod
    def _key(name: str, labels: Optional[Dict[str, Any]]) -> str:
        if not labels:
            return name
        parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{parts}}}"

    def incr(self, name: str, value: float = 1.0, **labels: Any) -> None:
        self._counters[self._key(name, labels)] += value

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        self._gauges[self._key(name, labels)] = value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        h = self._hist[self._key(name, labels)]
        h["count"] += 1
        h["sum"] += value
        h["samples"].append(value)
        for b in _BUCKETS:
            if value <= b:
                h["buckets"][b] += 1

    def quantile(self, name: str, q: float) -> float:
        for h in self._hist.values():
            if h["count"]:
                s = sorted(h["samples"])
                idx = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
                return s[idx]
        return 0.0

    def render(self) -> str:
        lines: List[str] = []
        for k, v in sorted(self._counters.items()):
            lines.append(f"# TYPE {k} counter")
            lines.append(f"{k} {v}")
        for k, v in sorted(self._gauges.items()):
            lines.append(f"# TYPE {k} gauge")
            lines.append(f"{k} {v}")
        for k, h in sorted(self._hist.items()):
            lines.append(f"# TYPE {k} histogram")
            for b in _BUCKETS:
                lines.append(f'{k}_bucket{{le="{b}"}} {h["buckets"][b]}')
            lines.append(f'{k}_bucket{{le="+Inf"}} {h["count"]}')
            lines.append(f"{k}_sum {h['sum']}")
            lines.append(f"{k}_count {h['count']}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()


def gpu_stats() -> Dict[str, Any]:
    """采集 GPU 显存/利用率 (优先 nvidia-smi, 回落到 torch)。"""
    out: Dict[str, Any] = {"available": False, "memory_used_bytes": 0.0,
                            "memory_total_bytes": 0.0, "utilization_percent": 0.0,
                            "devices": 0}
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            r = subprocess.run(
                [exe, "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            rows = [l for l in r.stdout.strip().splitlines() if l.strip()]
            if rows:
                used = total = util = 0.0
                for row in rows:
                    parts = [p.strip() for p in row.split(",")]
                    used += float(parts[0]) * 1024 ** 2
                    total += float(parts[1]) * 1024 ** 2
                    util = max(util, float(parts[2]))
                out.update(available=True, memory_used_bytes=used,
                           memory_total_bytes=total, utilization_percent=util,
                           devices=len(rows))
                return out
        except Exception:
            pass
    if torch is not None and getattr(torch, "cuda", None) is not None:
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                out.update(available=True, memory_used_bytes=total - free,
                           memory_total_bytes=total,
                           devices=torch.cuda.device_count())
                return out
        except Exception:
            pass
    return out


# ======================================================================================= #
# 3. 媒体 IO: URL / base64 / 本地文件 -> 图像 / 波形
# ======================================================================================= #

def _decode_data_url(url: str) -> bytes:
    """`data:<mime>;base64,<payload>` -> 原始字节。"""
    if not url.startswith("data:"):
        raise ValueError("不是合法的 data URL")
    header, _, payload = url.partition(",")
    if not payload:
        raise ValueError("data URL 缺少 payload")
    if "base64" in header:
        try:
            return base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"base64 解码失败: {e}") from e
    from urllib.parse import unquote_to_bytes

    return unquote_to_bytes(payload)


async def _fetch_url(url: str, max_bytes: int, timeout: float = 20.0) -> bytes:
    """下载 http(s) 资源 (带大小上限)。"""
    import httpx

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise ValueError(f"资源超过大小上限 {max_bytes} 字节")
            return bytes(buf)


async def load_image_bytes(source: str) -> bytes:
    """把图片来源 (data URL / http URL / 本地路径) 统一成字节。"""
    if source.startswith("data:"):
        return _decode_data_url(source)
    if source.startswith(("http://", "https://")):
        return await _fetch_url(source, MAX_IMAGE_BYTES)
    p = Path(source)
    if p.is_file():
        if p.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("图片文件过大")
        return p.read_bytes()
    # 也允许直接给裸 base64
    try:
        return base64.b64decode(source, validate=True)
    except Exception as e:
        raise ValueError(f"无法解析图片来源: {source[:64]}...") from e


async def load_audio_bytes(source: str) -> bytes:
    if source.startswith("data:"):
        return _decode_data_url(source)
    if source.startswith(("http://", "https://")):
        return await _fetch_url(source, MAX_AUDIO_BYTES)
    p = Path(source)
    if p.is_file():
        if p.stat().st_size > MAX_AUDIO_BYTES:
            raise ValueError("音频文件过大")
        return p.read_bytes()
    try:
        return base64.b64decode(source, validate=True)
    except Exception as e:
        raise ValueError(f"无法解析音频来源: {source[:64]}...") from e


def decode_image(data: bytes):
    """字节 -> PIL.Image。"""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def decode_audio(data: bytes) -> Tuple[Any, int]:
    """字节 -> (波形 np.float32 单声道, 采样率)。

    依次尝试 soundfile -> librosa -> 标准库 wave。
    """
    import io

    try:
        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)
        return arr, int(sr)
    except Exception:
        pass
    try:
        import librosa

        arr, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
        return arr.astype("float32"), int(sr)
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
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
        if n_ch > 1:
            arr = arr.reshape(-1, n_ch).mean(axis=1)
        return arr, int(sr)
    except Exception as e:
        raise ValueError(f"无法解码音频 (需要 soundfile / librosa / 标准 wav): {e}") from e


def resample_to_16k(waveform, sr: int, target_sr: int = 16000):
    """线性重采样到 16kHz (够用; 要求高保真时改用 librosa.resample)。"""
    if sr == target_sr:
        return waveform
    n_out = int(round(len(waveform) * target_sr / float(sr)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(waveform), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, waveform).astype(np.float32)


# ======================================================================================= #
# 4. 占位符静态展开
# ======================================================================================= #

def expand_prompt_text(prompt: str, cfg, n_images: int, n_audio_chunks: int) -> str:
    """文本级展开: 把每个 `<image>` 重复成 N 个, 每个 `<audio>` 重复成 M 个。

    供"后端不做自动展开"的场景使用 (例如直接对着裸 vLLM / 自建服务发纯文本)。
    """
    n_img_tok = image_token_count(cfg)
    n_aud_tok = cfg.audio.out_tokens_per_chunk * max(n_audio_chunks, 0)
    out = prompt.replace(IMAGE_MARKER, IMAGE_MARKER * n_img_tok)
    out = out.replace(AUDIO_MARKER, AUDIO_MARKER * max(n_aud_tok, 0))
    return out


def expand_prompt_tokens(prompt: str, tokenizer, cfg, n_images: int, n_audio_chunks: int) -> List[int]:
    """token 级展开 (本地 transformers 后端用)。

    每个 `<image>` 标记 -> `cfg.vision.out_tokens` 个 image_token_id;
    每个 `<audio>` 标记 -> `n_audio_chunks * cfg.audio.out_tokens_per_chunk` 个 audio_token_id。
    """
    n_img_tok = image_token_count(cfg)
    n_aud_tok = int(cfg.audio.out_tokens_per_chunk) * max(n_audio_chunks, 0)

    ids: List[int] = []
    parts = re.split(f"({re.escape(IMAGE_MARKER)}|{re.escape(AUDIO_MARKER)})", prompt)
    for part in parts:
        if not part:
            continue
        if part == IMAGE_MARKER:
            ids.extend([int(cfg.image_token_id)] * n_img_tok)
        elif part == AUDIO_MARKER:
            ids.extend([int(cfg.audio_token_id)] * n_aud_tok)
        else:
            ids.extend(tokenizer.encode(part, add_special_tokens=False))
    if getattr(cfg.llm, "add_bos", False) and tokenizer is not None:
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None:
            ids = [int(bos)] + ids
    return ids


def count_audio_chunks(durations: Sequence[float], cfg) -> int:
    """总 chunk 数 = 各段音频按 30s 向上取整后求和。"""
    return sum(max(1, int(math.ceil(d / float(cfg.audio.chunk_seconds or 30))))
               for d in durations)


# ======================================================================================= #
# 5. tokenizer
# ======================================================================================= #

class ByteFallbackTokenizer:
    """没有 HF tokenizer 时的兜底: 逐字节编码 (仅用于本地联调, 不是真 BPE)。"""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        out = bytearray()
        for i in ids:
            if skip_special_tokens and i >= 32001:
                continue
            out.append(int(i) % 256)
        return out.decode("utf-8", errors="replace")

    @property
    def bos_token_id(self) -> Optional[int]:
        return 1

    @property
    def eos_token_id(self) -> Optional[int]:
        return 2


def load_tokenizer(model_dir: str, cfg):
    """按优先级加载 tokenizer。

    1) 模型目录下的本地 tokenizer (local_files_only, 不会触网, 失败也很快)
    2) `cfg.llm.tokenizer` 指向的 hub id —— **默认关闭**, 因为无外网的机器上
       HF 会重试 5 次、每次 10s 超时, 白白拖慢启动; 需要时用 `MM_ALLOW_HUB=1` 打开
    3) 字节兜底 tokenizer (仅本地联调)
    """
    allow_hub = os.environ.get("MM_ALLOW_HUB", "0").strip().lower() in ("1", "true", "yes", "on")
    try:
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            print(f"[serve] tokenizer 来自本地目录 {model_dir}")
            return tok
        except Exception as e:
            print(f"[serve] 本地目录无 tokenizer ({type(e).__name__})")

        if allow_hub:
            # 限制 hub 超时, 避免无外网时长时间阻塞启动
            os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
            try:
                tok = AutoTokenizer.from_pretrained(cfg.llm.tokenizer)
                print(f"[serve] tokenizer 来自 hub {cfg.llm.tokenizer}")
                return tok
            except Exception as e:
                print(f"[serve] hub tokenizer 不可用: {e}")
        else:
            print(f"[serve] 跳过 hub tokenizer ({cfg.llm.tokenizer}); "
                  f"需要联网下载请设 MM_ALLOW_HUB=1")
    except ImportError:
        print("[serve] 未安装 transformers, 使用字节兜底 tokenizer")
    return ByteFallbackTokenizer(cfg.llm.vocab_size)


# ======================================================================================= #
# 5.1 模型目录识别 (mmllm 原生 / HF Llava 导出目录)
# ======================================================================================= #

@dataclass
class ModelSpec:
    """模型目录的元信息。

    两种目录:
      * `mmllm`    : 原生 checkpoint (config.json 是 MMConfig) -> 图像+音频都支持
      * `hf-llava` : export_hf.py 导出的目录 -> 只有图像, 音频需走降级方案
    """
    cfg: Any
    kind: str = "mmllm"
    image_tokens: int = 256
    audio_tokens_per_chunk: int = 25
    note: str = ""


def load_model_spec(model_dir: str) -> ModelSpec:
    """从模型目录的 config.json 推断配置与占位符契约。"""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        print(f"[serve] 未找到 {cfg_path}, 使用默认 preset(base)")
        cfg = default_cfg()
        return ModelSpec(cfg=cfg, image_tokens=image_token_count(cfg),
                         audio_tokens_per_chunk=int(cfg.audio.out_tokens_per_chunk),
                         note="未找到 config.json, 使用默认配置")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ---- HF Llava 目录 ---- #
    if raw.get("model_type") == "llava" or "vision_config" in raw:
        vis = raw.get("vision_config", {}) or {}
        grid = (vis.get("image_size", 336) // max(vis.get("patch_size", 14), 1)) ** 2
        image_tokens = int(raw.get("image_seq_length") or grid)
        print(f"[serve] 识别为 HF Llava 目录: image_seq_length={image_tokens} "
              f"(音频塔不在 HF Llava 内, 音频需走降级方案)")
        cfg = default_cfg()
        return ModelSpec(cfg=cfg, kind="hf-llava", image_tokens=image_tokens,
                         audio_tokens_per_chunk=0,
                         note="HF Llava 目录: 仅图像; 音频请用 transformers 后端或 Whisper 前置方案")

    # ---- mmllm 原生目录 ---- #
    try:
        MMConfig = load_mm_config_class()
        cfg = MMConfig.from_dict(raw)
        return ModelSpec(cfg=cfg, kind="mmllm",
                         image_tokens=image_token_count(cfg),
                         audio_tokens_per_chunk=int(cfg.audio.out_tokens_per_chunk))
    except Exception as e:
        print(f"[serve] config.json 解析失败 ({e}), 回落到默认配置")
        cfg = default_cfg()
        return ModelSpec(cfg=cfg, image_tokens=image_token_count(cfg),
                         audio_tokens_per_chunk=int(cfg.audio.out_tokens_per_chunk),
                         note="config.json 解析失败, 使用默认配置")


# ======================================================================================= #
# 6. 后端
# ======================================================================================= #

@dataclass
class GenRequest:
    """统一的后端输入。"""
    prompt: str
    images: List[Any] = field(default_factory=list)      # PIL.Image
    audios: List[Tuple[Any, int]] = field(default_factory=list)   # (waveform, sr)
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    stream: bool = False


@dataclass
class GenResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft: float = 0.0
    total_time: float = 0.0
    tpot: float = 0.0
    finish_reason: str = "stop"
    extra: Dict[str, Any] = field(default_factory=dict)


class Backend:
    """后端接口。"""

    name = "base"

    async def generate(self, req: GenRequest) -> GenResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class TransformersBackend(Backend):
    """本地 transformers 推理 (直接用 `MultimodalLM.from_pretrained`)。"""

    name = "transformers"

    def __init__(self, model_dir: str, cfg, settings: Settings):
        if torch is None:
            raise RuntimeError("未安装 torch, 无法使用 transformers 后端")
        from src.mmllm.modeling import MultimodalLM  # type: ignore

        if not (Path(model_dir) / "model.safetensors").is_file():
            raise RuntimeError(
                f"{model_dir} 不是本项目的原生 checkpoint (缺少 model.safetensors); "
                f"transformers 后端只支持 mmllm 原生权重, HF Llava 目录请用 --backend vllm")
        self.cfg = cfg
        self.settings = settings
        self.device = settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[serve] 未检测到 CUDA, 回落到 CPU")
            self.device = "cpu"
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(settings.dtype, torch.float32)
        if self.device == "cpu":
            dtype = torch.float32
        print(f"[serve] 加载模型: {model_dir} (device={self.device}, dtype={dtype})")
        self.model = MultimodalLM.from_pretrained(model_dir)
        self.model.to(device=self.device, dtype=dtype)
        self.model.eval()
        self.tokenizer = load_tokenizer(model_dir, cfg)

    # -------- 单步采样 (与 modeling.generate 保持一致) -------- #
    def _sample_next(self, logits, temperature: float, top_p: float):
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = sp.cumsum(dim=-1)
        mask = cum - sp > top_p
        sp[mask] = 0.0
        sp = sp / sp.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sp.view(-1, sp.size(-1)), 1)
        return si.view(logits.shape[0], -1).gather(1, choice)

    async def generate(self, req: GenRequest) -> GenResult:
        cfg = self.cfg
        n_chunks = count_audio_chunks([len(a) / float(sr or 16000) for a, sr in req.audios], cfg)

        # ---- 占位符静态展开 ---- #
        ids = expand_prompt_tokens(req.prompt, self.tokenizer, cfg, len(req.images), n_chunks)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        pixel_values = None
        if req.images:
            pixel_values = torch.stack([preprocess_image(im, cfg.vision.image_size)
                                        for im in req.images]).to(self.device)

        input_features = None
        if req.audios:
            feats = []
            for wave, sr in req.audios:
                wave16 = resample_to_16k(wave, sr, cfg.audio.sample_rate)
                for ch in chunk_audio(wave16, cfg):
                    feats.append(preprocess_audio_chunk(ch, cfg))
            input_features = torch.stack(feats).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(
                input_ids, pixel_values=pixel_values, input_features=input_features)
            # ---- prefill ---- #
            logits, presents = self.model.language_model(
                inputs_embeds=inputs_embeds, use_cache=True)
            ttft = time.perf_counter() - t0

            out_ids: List[int] = []
            tpot_samples: List[float] = []
            eos = getattr(self.tokenizer, "eos_token_id", None)
            for _ in range(req.max_new_tokens):
                next_token = self._sample_next(logits[:, -1, :], req.temperature, req.top_p)
                token_id = int(next_token[0].item())
                out_ids.append(token_id)
                if eos is not None and token_id == int(eos):
                    break
                step_embeds = self.model.language_model.embed_tokens(next_token)
                t_step = time.perf_counter()
                logits, presents = self.model.language_model(
                    inputs_embeds=step_embeds, past_key_values=presents, use_cache=True)
                tpot_samples.append(time.perf_counter() - t_step)

        total = time.perf_counter() - t0
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        n_out = len(out_ids)
        tpot = (sum(tpot_samples) / len(tpot_samples)) if tpot_samples else 0.0
        return GenResult(text=text, prompt_tokens=len(ids), completion_tokens=n_out,
                         ttft=ttft, total_time=total, tpot=tpot,
                         finish_reason="stop" if (eos is None or n_out < req.max_new_tokens)
                         else "length")


class VLLMProxyBackend(Backend):
    """转发到 OpenAI 兼容服务 (vLLM / SGLang / 网关)。"""

    name = "vllm-proxy"

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def generate(self, req: GenRequest) -> GenResult:
        content: List[Dict[str, Any]] = [{"type": "text", "text": req.prompt}]
        for img in req.images:
            buf = _pil_to_png_bytes(img)
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," +
                                          base64.b64encode(buf).decode("ascii")}})
        for wave, sr in req.audios:
            wav = _wave_to_wav_bytes(wave, sr)
            content.append({"type": "input_audio",
                            "input_audio": {"data": base64.b64encode(wav).decode("ascii"),
                                            "format": "wav"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
        }
        t0 = time.perf_counter()
        r = await self._client.post(f"{self.base_url}/v1/chat/completions",
                                    json=payload, headers=self._headers())
        r.raise_for_status()
        data = r.json()
        total = time.perf_counter() - t0
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        n_out = int(usage.get("completion_tokens", 0) or 0)
        ttft = float(data.get("_mm_ttft", 0.0) or 0.0) or total
        tpot = (total - ttft) / max(n_out - 1, 1) if n_out > 1 else 0.0
        return GenResult(text=text,
                         prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                         completion_tokens=n_out, ttft=ttft, total_time=total, tpot=tpot)


class VLLMSubprocessBackend(VLLMProxyBackend):
    """拉起 `vllm serve` 子进程, 再按 OpenAI 协议转发。"""

    name = "vllm-subprocess"

    def __init__(self, model_dir: str, settings: Settings, port: Optional[int] = None):
        self.settings = settings
        self.model_dir = model_dir
        self.port = port or (settings.port + 1)
        self.proc: Optional[subprocess.Popen] = None
        super().__init__(f"http://127.0.0.1:{self.port}", settings.vllm_model, settings.api_key)

    def start(self) -> None:
        if shutil.which("vllm") is None:
            raise RuntimeError("未找到 vllm 可执行文件; 请先 `pip install vllm` 或改用 --backend transformers")
        cmd = ["vllm", "serve", self.model_dir,
               "--port", str(self.port),
               "--host", "127.0.0.1",
               "--gpu-memory-utilization", str(self.settings.vllm_gpu_memory_utilization),
               "--max-model-len", str(self.settings.vllm_max_model_len)]
        if self.settings.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.settings.vllm_args:
            cmd += self.settings.vllm_args.split()
        print(f"[serve] 启动 vLLM: {' '.join(cmd)}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    async def wait_ready(self, timeout: float = 600.0) -> bool:
        import httpx

        deadline = time.time() + timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() < deadline:
                if self.proc is not None and self.proc.poll() is not None:
                    print(f"[serve] vLLM 子进程提前退出 (code={self.proc.returncode})")
                    return False
                try:
                    r = await client.get(f"{self.base_url}/health")
                    if r.status_code == 200:
                        print("[serve] vLLM 已就绪")
                        return True
                except Exception:
                    pass
                await asyncio.sleep(2.0)
        return False

    async def aclose(self) -> None:
        await super().aclose()
        if self.proc is not None and self.proc.poll() is None:
            print("[serve] 正在关闭 vLLM 子进程 ...")
            try:
                self.proc.terminate()
                self.proc.wait(timeout=30)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


class VLLMEngineBackend(Backend):
    """进程内 vLLM 引擎 (AsyncLLMEngine / AsyncLLM)。"""

    name = "vllm-engine"

    def __init__(self, model_dir: str, settings: Settings):
        if torch is None:
            raise RuntimeError("未安装 torch")
        self.model_dir = model_dir
        self.settings = settings
        self.engine = None
        self._api = "unknown"
        try:
            # vLLM >= 0.10
            try:
                from vllm.v1.engine.async_llm import AsyncLLM  # type: ignore
                from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore

                self.engine = AsyncLLM.from_engine_args(
                    AsyncEngineArgs(model=model_dir,
                                    gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
                                    max_model_len=settings.vllm_max_model_len))
                self._api = "AsyncLLM"
            except Exception:
                # vLLM <= 0.9
                from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
                from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore

                self.engine = AsyncLLMEngine.from_engine_args(
                    AsyncEngineArgs(model=model_dir,
                                    gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
                                    max_model_len=settings.vllm_max_model_len))
                self._api = "AsyncLLMEngine"
        except Exception as e:
            raise RuntimeError(f"初始化 vLLM 引擎失败: {e}") from e
        print(f"[serve] vLLM 引擎就绪 ({self._api})")

    async def aclose(self) -> None:
        if self.engine is not None:
            for meth in ("shutdown", "close", "shutdown_async"):
                fn = getattr(self.engine, meth, None)
                if callable(fn):
                    try:
                        res = fn()
                        if asyncio.iscoroutine(res):
                            await res
                        return
                    except Exception:
                        pass

    async def generate(self, req: GenRequest) -> GenResult:
        from vllm.inputs.data import TextPrompt  # type: ignore

        mm_data: Dict[str, Any] = {}
        if req.images:
            mm_data["image"] = list(req.images)
        if req.audios:
            mm_data["audio"] = [a for a, _ in req.audios]

        prompt = TextPrompt(prompt=req.prompt, multi_modal_data=mm_data or None)
        sampling = _make_sampling_params(req)
        t0 = time.perf_counter()
        final: Optional[str] = None
        n_out = 0
        async for out in self.engine.generate(prompt, sampling, request_id=uuid.uuid4().hex):
            text = out.outputs[0].text if out.outputs else ""
            n_out = len(out.outputs[0].token_ids) if out.outputs else 0
            if text:
                final = text
        total = time.perf_counter() - t0
        return GenResult(text=final or "", completion_tokens=n_out,
                         ttft=total, total_time=total, tpot=0.0)


def _make_sampling_params(req: GenRequest):
    from vllm import SamplingParams  # type: ignore

    return SamplingParams(max_tokens=req.max_new_tokens, temperature=req.temperature,
                          top_p=req.top_p)


def _pil_to_png_bytes(img) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _wave_to_wav_bytes(waveform, sr: int) -> bytes:
    """波形 -> wav 字节 (16bit PCM)。"""
    import io
    import wave

    data = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ======================================================================================= #
# 7. 请求体
# ======================================================================================= #

if FASTAPI_AVAILABLE:
    class GenerateBody(BaseModel):
        """裸接口请求体。"""
        prompt: str = Field(..., description="提示词; 用 <image> / <audio> 标记多模态位置")
        images: List[str] = Field(default_factory=list, description="图片 URL / data URL / base64")
        audios: List[str] = Field(default_factory=list, description="音频 URL / data URL / base64")
        max_new_tokens: Optional[int] = Field(default=None, description="默认取 MM_MAX_NEW_TOKENS")
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        stream: bool = False
else:  # pragma: no cover
    class GenerateBody:  # type: ignore
        pass


# ======================================================================================= #
# 8. 应用
# ======================================================================================= #

def _parse_chat_messages(messages: Sequence[Dict[str, Any]]) -> Tuple[str, List[str], List[str]]:
    """把 OpenAI messages 拆成 (文本 prompt, 图片列表, 音频列表)。"""
    text_parts: List[str] = []
    images: List[str] = []
    audios: List[str] = []

    for msg in messages or []:
        content = msg.get("content")
        role = msg.get("role", "user")
        if isinstance(content, str):
            text_parts.append(f"{content}")
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text"):
                text_parts.append(str(part.get("text", "")))
            elif ptype == "image_url":
                iu = part.get("image_url") or {}
                url = iu.get("url") if isinstance(iu, dict) else iu
                if url:
                    images.append(str(url))
            elif ptype in ("audio", "input_audio"):
                a = part.get("audio") or part.get("input_audio") or {}
                if isinstance(a, dict):
                    url = a.get("url")
                    if url:
                        audios.append(str(url))
                    elif a.get("data"):
                        fmt = a.get("format", "wav")
                        audios.append(f"data:audio/{fmt};base64,{a['data']}")
            elif ptype == "video":
                raise HTTPException(status_code=400, detail="暂不支持 video 输入")
        if role == "assistant":
            text_parts.append("")

    prompt = "\n".join(t for t in text_parts if t is not None)
    return prompt, images, audios


async def _prepare_inputs(images: Sequence[str], audios: Sequence[str], cfg
                           ) -> Tuple[List[Any], List[Tuple[Any, int]], int]:
    """下载 + 解码 + 预处理, 返回 (图像列表, 音频列表, 音频 chunk 总数)。"""
    imgs = []
    for src in images:
        data = await load_image_bytes(src)
        imgs.append(decode_image(data))

    auds = []
    for src in audios:
        data = await load_audio_bytes(src)
        wave, sr = decode_audio(data)
        auds.append((wave, sr))

    n_chunks = count_audio_chunks([len(w) / float(sr or 16000) for w, sr in auds], cfg)
    return imgs, auds, n_chunks


def create_app(settings: Settings, cfg=None, backend: Optional[Backend] = None,
               spec: Optional[ModelSpec] = None):
    """构造 FastAPI 应用。"""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("未安装 fastapi / uvicorn, 无法启动服务; "
                           "请 `pip install fastapi uvicorn` 或使用 --backend 检查依赖")

    if cfg is None:
        cfg = default_cfg()
    if spec is None:
        spec = ModelSpec(cfg=cfg, image_tokens=image_token_count(cfg),
                         audio_tokens_per_chunk=int(cfg.audio.out_tokens_per_chunk))
    state: Dict[str, Any] = {"settings": settings, "cfg": cfg, "backend": backend,
                             "spec": spec, "sem": asyncio.Semaphore(settings.max_concurrency)}

    @asynccontextmanager
    async def lifespan(_app):
        """优雅关闭: 释放 httpx 连接 / 关掉 vLLM 子进程。"""
        yield
        print("[serve] 正在优雅关闭 ...")
        if backend is not None:
            try:
                await backend.aclose()
            except Exception as e:
                print(f"[serve] 关闭后端时出错: {e}")

    app = FastAPI(title="mmllm 多模态服务", version="1.0.0",
                  description="文本 / 图像 / 音频多模态大模型推理服务 (OpenAI 兼容)",
                  lifespan=lifespan)

    # ---------------- 鉴权 ---------------- #
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if settings.api_key:
            header = request.headers.get("authorization", "")
            token = header[7:] if header.lower().startswith("bearer ") else ""
            if token != settings.api_key:
                return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return await call_next(request)

    # ---------------- 健康检查 / 指标 ---------------- #
    @app.get("/healthz")
    async def healthz():
        b = state["backend"]
        ready = b is not None
        return {
            "status": "ok" if ready else "starting",
            "ready": ready,
            "backend": getattr(b, "name", "none"),
            "model_dir": settings.model_dir,
            "model": settings.vllm_model if settings.backend == "vllm" else "mmllm",
            "device": settings.device if settings.backend != "vllm" else "gpu",
            "model_kind": spec.kind,
            "image_tokens_per_image": spec.image_tokens,
            "audio_tokens_per_chunk": spec.audio_tokens_per_chunk,
            "note": spec.note,
        }

    @app.get("/metrics")
    async def metrics():
        g = gpu_stats()
        METRICS.gauge("mm_gpu_memory_used_bytes", g["memory_used_bytes"])
        METRICS.gauge("mm_gpu_memory_total_bytes", g["memory_total_bytes"])
        METRICS.gauge("mm_gpu_utilization_percent", g["utilization_percent"])
        METRICS.gauge("mm_gpu_devices", g["devices"])
        METRICS.gauge("mm_inflight_requests", state["sem"]._value)  # noqa: SLF001
        return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")

    # ---------------- 裸接口 ---------------- #
    @app.post("/generate")
    async def generate(body: GenerateBody):
        b = state["backend"]
        if b is None:
            raise HTTPException(status_code=503, detail="后端尚未就绪")
        t_start = time.perf_counter()
        try:
            imgs, auds, n_chunks = await _prepare_inputs(body.images, body.audios, cfg)
            req = GenRequest(
                prompt=body.prompt,
                images=imgs,
                audios=auds,
                max_new_tokens=body.max_new_tokens or settings.max_new_tokens,
                temperature=body.temperature if body.temperature is not None else settings.temperature,
                top_p=body.top_p if body.top_p is not None else settings.top_p,
            )
            async with state["sem"]:
                res = await asyncio.wait_for(b.generate(req), timeout=settings.request_timeout)
        except asyncio.TimeoutError:
            METRICS.incr("mm_requests_total", endpoint="generate", status="timeout")
            METRICS.incr("mm_errors_total", kind="timeout")
            raise HTTPException(status_code=504, detail="请求超时")
        except HTTPException:
            raise
        except Exception as e:
            METRICS.incr("mm_requests_total", endpoint="generate", status="error")
            METRICS.incr("mm_errors_total", kind=type(e).__name__)
            raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")

        METRICS.incr("mm_requests_total", endpoint="generate", status="ok")
        METRICS.observe("mm_request_duration_seconds", time.perf_counter() - t_start)
        METRICS.observe("mm_ttft_seconds", res.ttft)
        METRICS.observe("mm_tpot_seconds", res.tpot)
        METRICS.incr("mm_prompt_tokens_total", res.prompt_tokens)
        METRICS.incr("mm_output_tokens_total", res.completion_tokens)
        return {
            "text": res.text,
            "usage": {"prompt_tokens": res.prompt_tokens,
                      "completion_tokens": res.completion_tokens,
                      "total_tokens": res.prompt_tokens + res.completion_tokens},
            "metrics": {"ttft": res.ttft, "tpot": res.tpot, "total": res.total_time},
            "multimodal": {"images": len(imgs), "audio_chunks": n_chunks,
                           "image_tokens_each": image_token_count(cfg),
                           "audio_tokens_each_chunk": int(cfg.audio.out_tokens_per_chunk)},
            "finish_reason": res.finish_reason,
        }

    # ---------------- OpenAI 兼容 ---------------- #
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        b = state["backend"]
        if b is None:
            raise HTTPException(status_code=503, detail="后端尚未就绪")
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="请求体不是合法 JSON")

        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages 不能为空")
        prompt, image_srcs, audio_srcs = _parse_chat_messages(messages)
        if not prompt.strip():
            prompt = "请描述输入的内容。"

        max_new_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                             or settings.max_new_tokens)
        temperature = float(body.get("temperature", settings.temperature))
        top_p = float(body.get("top_p", settings.top_p))
        stream = bool(body.get("stream", False))
        model_name = body.get("model") or settings.vllm_model

        try:
            imgs, auds, n_chunks = await _prepare_inputs(image_srcs, audio_srcs, cfg)
        except Exception as e:
            METRICS.incr("mm_requests_total", endpoint="chat", status="error")
            raise HTTPException(status_code=400, detail=f"多模态输入处理失败: {e}")

        req = GenRequest(prompt=prompt, images=imgs, audios=auds,
                         max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, stream=stream)

        if stream:
            return StreamingResponse(_sse_stream(state, req, model_name),
                                     media_type="text/event-stream")

        t_start = time.perf_counter()
        try:
            async with state["sem"]:
                res = await asyncio.wait_for(b.generate(req), timeout=settings.request_timeout)
        except asyncio.TimeoutError:
            METRICS.incr("mm_requests_total", endpoint="chat", status="timeout")
            METRICS.incr("mm_errors_total", kind="timeout")
            raise HTTPException(status_code=504, detail="请求超时")
        except HTTPException:
            raise
        except Exception as e:
            METRICS.incr("mm_requests_total", endpoint="chat", status="error")
            METRICS.incr("mm_errors_total", kind=type(e).__name__)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

        METRICS.incr("mm_requests_total", endpoint="chat", status="ok")
        METRICS.observe("mm_request_duration_seconds", time.perf_counter() - t_start)
        METRICS.observe("mm_ttft_seconds", res.ttft)
        METRICS.observe("mm_tpot_seconds", res.tpot)
        METRICS.incr("mm_prompt_tokens_total", res.prompt_tokens)
        METRICS.incr("mm_output_tokens_total", res.completion_tokens)
        return _chat_response(res, model_name)

    return app


def _chat_response(res: GenResult, model_name: str) -> Dict[str, Any]:
    """OpenAI 兼容的 chat.completion 响应体。"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": res.text},
            "finish_reason": res.finish_reason,
        }],
        "usage": {
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens,
        },
        "mm_metrics": {"ttft": res.ttft, "tpot": res.tpot, "total": res.total_time},
    }


async def _sse_stream(state: Dict[str, Any], req: GenRequest, model_name: str):
    """流式输出 (SSE)。"""
    import json as _json

    b = state["backend"]
    settings: Settings = state["settings"]
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> str:
        payload = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"

    t0 = time.perf_counter()
    res: Optional[GenResult] = None
    try:
        async with state["sem"]:
            res = await asyncio.wait_for(b.generate(req), timeout=settings.request_timeout)
        # 一次性把结果切块吐出去 (后端返回的是整段文本)
        text = res.text or ""
        step = max(1, len(text) // 16 or 1)
        for i in range(0, len(text), step):
            yield chunk({"content": text[i:i + step]})
            await asyncio.sleep(settings.stream_chunk_ms / 1000.0)
        yield chunk({}, finish=res.finish_reason)
        yield "data: [DONE]\n\n"
    except Exception as e:  # pragma: no cover
        yield chunk({"content": f"[error] {e}"}, finish="error")
        yield "data: [DONE]\n\n"
    finally:
        METRICS.incr("mm_requests_total", endpoint="chat", status="ok")
        METRICS.observe("mm_request_duration_seconds", time.perf_counter() - t0)
        METRICS.observe("mm_ttft_seconds", res.ttft if res is not None else 0.0)


# ======================================================================================= #
# 9. 启动
# ======================================================================================= #

def build_backend(settings: Settings, cfg) -> Backend:
    """按配置构造后端。"""
    backend = settings.backend
    if backend == "auto":
        backend = "vllm" if shutil.which("vllm") is not None else "transformers"

    if backend == "transformers":
        return TransformersBackend(settings.model_dir, cfg, settings)

    if backend == "vllm":
        if settings.vllm_mode == "proxy":
            return VLLMProxyBackend(settings.vllm_base_url, settings.vllm_model,
                                    settings.api_key)
        if settings.vllm_mode == "engine":
            return VLLMEngineBackend(settings.model_dir, settings)
        proc = VLLMSubprocessBackend(settings.model_dir, settings)
        proc.start()
        return proc

    raise ValueError(f"未知后端: {settings.backend}")


async def _preload_subprocess(backend: Backend) -> None:
    if isinstance(backend, VLLMSubprocessBackend):
        ok = await backend.wait_ready()
        if not ok:
            raise RuntimeError("vLLM 子进程未能在超时时间内就绪")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="mmllm 多模态推理服务")
    ap.add_argument("--model-dir", default=None, help="checkpoint 目录")
    ap.add_argument("--backend", default=None, choices=["transformers", "vllm", "auto"])
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--request-timeout", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--vllm-mode", default=None, choices=["subprocess", "engine", "proxy"])
    ap.add_argument("--vllm-base-url", default=None)
    ap.add_argument("--vllm-model", default=None)
    ap.add_argument("--vllm-args", default=None)
    ap.add_argument("--log-level", default=None)
    ap.add_argument("--check", action="store_true", help="只检查依赖与配置, 不启动服务")
    args = ap.parse_args(argv)

    settings = Settings.from_args(args)
    if not settings.model_dir:
        print("[serve] 缺少 --model-dir (或环境变量 MM_MODEL_DIR)")
        return 2

    cfg = default_cfg()
    spec = load_model_spec(settings.model_dir)
    cfg = spec.cfg
    print(f"[serve] 配置: backend={settings.backend}, model_dir={settings.model_dir}, "
          f"并发={settings.max_concurrency}, 超时={settings.request_timeout}s")
    print(f"[serve] 模型类型: {spec.kind}; 占位符契约: 每图 {spec.image_tokens} 个 image token, "
          f"每 30s 音频 {spec.audio_tokens_per_chunk} 个 audio token")
    if spec.note:
        print(f"[serve] 提示: {spec.note}")

    if args.check:
        print(f"[serve] fastapi 可用: {FASTAPI_AVAILABLE}; "
              f"torch: {torch is not None}; "
              f"vllm 可执行文件: {shutil.which('vllm')}")
        return 0

    if not FASTAPI_AVAILABLE:
        print("[serve] 未安装 fastapi/uvicorn, 无法启动服务。请执行: "
              "pip install fastapi uvicorn")
        return 2

    try:
        import uvicorn
    except ImportError:
        print("[serve] 未安装 uvicorn, 无法启动服务。请执行: pip install uvicorn")
        return 2

    backend = build_backend(settings, cfg)
    app = create_app(settings, cfg, backend, spec)

    if isinstance(backend, VLLMSubprocessBackend):
        try:
            asyncio.run(_preload_subprocess(backend))
        except Exception as e:
            print(f"[serve] {e}")
            return 3

    print(f"[serve] 监听 http://{settings.host}:{settings.port}  "
          f"(文档: /docs)")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
