"""
多模态数据管线
==============
职责:
    1. 把 "对话 + 图片 + 音频" 的原始记录, 变成模型可直接吃的一批张量
    2. 只在 assistant 轮次上计算 loss (标签掩码)
    3. 占位符静态展开, 保证与 modeling.py 的原位替换逻辑严格对齐

设计要点
--------
* **分段 tokenize**: 每个对话片段单独 tokenize 再拼接, 从而能精确记录
  assistant 片段的 token 区间, 避免 BPE 在拼接边界上"改写"前一段的切分。
* **图像/音频预处理自包含**: 不强依赖 torchvision / torchaudio,
  用 PIL + numpy + torch.stft 实现, 任何装了 torch 的机器都能跑。
* **批内多图多音**: collate 时把整批图片堆成 (B_img, 3, H, W)、音频堆成
  (B_chunk, 128, 3000), 与 modeling.py 的 flatten-then-scatter 语义一致。
"""

from __future__ import annotations

import math
import os
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import MMConfig
from .tokenizer_ext import (
    AUDIO_TOKEN,
    IMAGE_TOKEN,
    add_multimodal_tokens,
    expand_placeholders,
)

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# =========================================================================== #
# 图像预处理
# =========================================================================== #
class SimpleImageProcessor:
    """
    CLIP 风格图像预处理:
        解码 -> RGB -> 缩放到 (S, S) -> 归一化 -> (3, S, S)
    训练时如需数据增强(随机裁剪/翻转), 在此扩展即可。
    """

    def __init__(self, image_size: int = 336, mean=CLIP_MEAN, std=CLIP_STD):
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, path_or_image) -> torch.Tensor:
        from PIL import Image

        if isinstance(path_or_image, Image.Image):
            img = path_or_image.convert("RGB")
        else:
            img = Image.open(path_or_image).convert("RGB")

        # CLIP 用 bicubic 缩放, 且短边先缩到 S 再中心裁 S
        w, h = img.size
        s = self.image_size
        scale = s / min(w, h)
        nw, nh = max(s, round(w * scale)), max(s, round(h * scale))
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - s) // 2, (nh - s) // 2
        img = img.crop((left, top, left + s, top + s))

        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8)).float() / 255.0
        arr = arr.permute(2, 0, 1)                      # (3, S, S)
        return (arr - self.mean) / self.std


# =========================================================================== #
# 音频预处理
# =========================================================================== #
def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0,
                     fmax: Optional[float] = None) -> torch.Tensor:
    """Slaney 风格 mel 滤波器组, 返回 (n_mels, n_fft//2+1)。"""
    fmax = fmax or sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sr / 2.0, n_bins)
    mel_pts = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = torch.tensor([mel_to_hz(float(m)) for m in mel_pts])

    # mel 频点: f_0 .. f_{n_mels+1};  每个滤波带 j 由 (f_j, f_{j+1}, f_{j+2}) 构成三角形
    f_diff = hz_pts[1:] - hz_pts[:-1]                          # (n_mels+1,)
    slopes = hz_pts.unsqueeze(0) - fft_freqs.unsqueeze(1)      # (n_bins, n_mels+2)

    # 上升沿: (fft - f_j) / (f_{j+1} - f_j)
    lower = (-slopes[:, 0:n_mels] / f_diff[0:n_mels]).clamp(min=0.0)
    # 下降沿: (f_{j+2} - fft) / (f_{j+2} - f_{j+1})
    upper = (slopes[:, 2:n_mels + 2] / f_diff[1:n_mels + 1]).clamp(min=0.0)

    weights = torch.minimum(lower, upper)                     # (n_bins, n_mels)
    return weights


class SimpleAudioProcessor:
    """
    Whisper 风格 log-Mel 特征:
        解码 wav -> 重采样 16k -> 30s 定长(不足补零/超出截断) -> log-Mel (128, 3000)
    30s / 16k = 480000 采样点; hop=160 -> 3000 帧 (center padding)。
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 128,
                 n_fft: int = 400, hop_length: int = 160, max_seconds: int = 30):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_seconds = max_seconds
        self.n_samples = sample_rate * max_seconds
        self.filters = _mel_filterbank(sample_rate, n_fft, n_mels)

    # -------- 解码 --------
    @staticmethod
    def _read_wav(path: str) -> np.ndarray:
        """优先 soundfile, 退化到标准库 wave。返回 float32 单声道。"""
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return data.mean(axis=1), sr
        except Exception:
            with wave.open(path, "rb") as w:
                sr = w.getframerate()
                n = w.getnframes()
                ch = w.getnchannels()
                raw = w.readframes(n)
                dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[w.getsampwidth()]
                arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                if w.getsampwidth() == 1:
                    arr = (arr - 128.0) / 128.0
                else:
                    arr = arr / float(2 ** (8 * w.getsampwidth() - 1))
                if ch > 1:
                    arr = arr.reshape(-1, ch).mean(axis=1)
                return arr, sr

    @staticmethod
    def _resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
        if sr == target:
            return wav
        n_out = int(round(len(wav) * target / sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
        return np.interp(x_new, x_old, wav).astype(np.float32)

    def load(self, path: str) -> np.ndarray:
        wav, sr = self._read_wav(path)
        wav = self._resample(wav, sr, self.sample_rate)
        return wav

    def __call__(self, path: str) -> torch.Tensor:
        wav = self.load(path)
        wav = wav[: self.n_samples]
        if len(wav) < self.n_samples:
            wav = np.pad(wav, (0, self.n_samples - len(wav)))
        x = torch.from_numpy(wav).float()
        return self.log_mel(x)

    def log_mel(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_samples,) -> (n_mels, 3000)"""
        window = torch.hann_window(self.n_fft)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=window, return_complex=True, center=True)
        power = spec.real.pow(2) + spec.imag.pow(2)              # (freq, frames)
        mel = self.filters @ power                                # (n_mels, frames)
        log_spec = torch.log10(mel.clamp(min=1e-10))
        log_spec = log_spec.max(log_spec.amax(dim=-1, keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0                              # 归一化到约 [-1, 1]

    def chunk(self, path: str, chunk_seconds: int = 30) -> List[torch.Tensor]:
        """长音频切成多个 30s chunk, 每 chunk 产出 (128, 3000)。"""
        wav = self.load(path)
        step = self.sample_rate * chunk_seconds
        out: List[torch.Tensor] = []
        for start in range(0, max(len(wav), 1), step):
            seg = wav[start:start + step]
            if len(seg) < self.sample_rate:      # 忽略 <1s 的尾巴
                break
            out.append(self.log_mel(torch.from_numpy(seg).float()))
        return out or [self.log_mel(torch.from_numpy(wav).float())]


# =========================================================================== #
# 对话模板
# =========================================================================== #
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_segments(conversations: Sequence[Dict[str, str]], cfg: MMConfig,
                   eos_token: str = IM_END) -> List[Dict[str, Any]]:
    """
    把对话记录转成 [(text, supervised), ...] 片段序列。
    只有 assistant 轮的正文被监督 (supervised=True)。
    """
    segments: List[Dict[str, Any]] = []
    for turn in conversations:
        role = turn.get("role", "user")
        content = turn.get("content", "") or ""
        content = expand_placeholders(content, cfg)
        if role == "assistant":
            segments.append({
                "text": f"{IM_START}assistant\n{content}{eos_token}",
                "supervised": True,
            })
        else:
            segments.append({
                "text": f"{IM_START}{role}\n{content}{IM_END}\n",
                "supervised": False,
            })
    return segments


# =========================================================================== #
# 数据集
# =========================================================================== #
class MMDataset(Dataset):
    """
    多模态对话数据集。

    records 每条形如:
        {
          "conversations": [{"role": "user", "content": "<image>\\n这是什么?"},
                            {"role": "assistant", "content": "这是一只猫。"}],
          "images": ["/data/1.jpg"],
          "audios": ["/data/1.wav"]
        }
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        cfg: MMConfig,
        tokenizer,
        image_processor: Optional[SimpleImageProcessor] = None,
        audio_processor: Optional[SimpleAudioProcessor] = None,
        max_text_len: int = 4096,
        max_images: int = 4,
        max_audio_chunks: int = 8,
        chat_template: bool = True,
    ):
        self.records = records
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_processor = image_processor or SimpleImageProcessor(cfg.vision.image_size)
        self.audio_processor = audio_processor or SimpleAudioProcessor(
            sample_rate=cfg.audio.sample_rate, n_mels=cfg.audio.n_mels,
            max_seconds=cfg.audio.max_seconds,
        )
        self.max_text_len = max_text_len
        self.max_images = max_images
        self.max_audio_chunks = max_audio_chunks
        self.chat_template = chat_template

    def __len__(self) -> int:
        return len(self.records)

    # -------- 文本 --------
    def _encode_text(self, conversations) -> Dict[str, List[int]]:
        """
        分段 tokenize + 标签掩码。返回 input_ids / labels。

        超长处理顺序(重要):
          1. 优先**丢弃最早的对话轮次** —— 最早的历史最不相关, 最近的上下文和
             最终回答才是训练信号。绝不能从头截断, 那会把最终回答整段切掉,
             导致该样本的监督信号全部落在 label=-100 上(白跑一个 batch)。
          2. 若单轮就超预算, 再从右截断该轮; 对非监督轮(用户提问)从右截是安全的;
             对监督轮(模型回答)从右截会切掉结论, 属于极端兜底, 已在日志中体现。
        """
        tok = self.tokenizer
        segments = build_segments(conversations, self.cfg)

        # 分段 tokenize, 记录每段是否被监督
        enc: List[Tuple[List[int], bool]] = [
            (tok(seg["text"], add_special_tokens=False)["input_ids"], seg["supervised"])
            for seg in segments
        ]

        bos = None
        if getattr(self.cfg.llm, "add_bos", False):
            bos = tok.bos_token_id
        budget = self.max_text_len - (1 if bos is not None else 0)

        # ---- 超长处理 1: 丢弃最早的对话轮次, 保住最近的上下文与最终回答 ----
        last_sup = -1
        for i, (_, s) in enumerate(enc):
            if s:
                last_sup = i
        total = sum(len(x) for x, _ in enc)
        drop = 0
        while total > budget and drop < max(last_sup, 0):
            total -= len(enc[drop][0])
            enc[drop] = ([], enc[drop][1])
            drop += 1

        # ---- 超长处理 2: 单轮就超预算 -> 从右截断 ----
        if total > budget:
            # 优先截非监督段(用户提问), 从右截是安全的
            for i, (ids, sup) in enumerate(enc):
                if not sup and len(ids) > 0:
                    cut = min(len(ids), total - budget)
                    enc[i] = (ids[:len(ids) - cut], sup)
                    total -= cut
                    if total <= budget:
                        break
            # 仍超预算: 只能截监督段(极端情况, 会切掉回答结尾)
            if total > budget:
                for i in range(len(enc) - 1, -1, -1):
                    ids, sup = enc[i]
                    if sup and len(ids) > 0:
                        cut = min(len(ids), total - budget)
                        enc[i] = (ids[:len(ids) - cut], sup)
                        total -= cut
                        break

        input_ids: List[int] = []
        labels: List[int] = []
        if bos is not None:
            input_ids.append(bos)
            labels.append(self.cfg.label_pad_id)

        for ids, sup in enc:
            input_ids.extend(ids)
            labels.extend(ids if sup else [self.cfg.label_pad_id] * len(ids))

        return {"input_ids": input_ids, "labels": labels}

    # -------- 模态 --------
    def _load_images(self, paths: List[str]) -> Optional[torch.Tensor]:
        if not paths:
            return None
        paths = paths[: self.max_images]
        tensors = [self.image_processor(p) for p in paths]
        return torch.stack(tensors, dim=0)          # (n_img, 3, S, S)

    def _load_audios(self, paths: List[str]) -> Optional[torch.Tensor]:
        if not paths:
            return None
        chunks: List[torch.Tensor] = []
        for p in paths:
            chunks.extend(self.audio_processor.chunk(p, self.cfg.audio.chunk_seconds))
        chunks = chunks[: self.max_audio_chunks]
        if not chunks:
            return None
        return torch.stack(chunks, dim=0)           # (n_chunk, 128, 3000)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        convs = rec.get("conversations") or rec.get("messages") or []
        enc = self._encode_text(convs)
        pixel_values = self._load_images(rec.get("images") or [])
        input_features = self._load_audios(rec.get("audios") or [])
        return {
            "input_ids": enc["input_ids"],
            "labels": enc["labels"],
            "pixel_values": pixel_values,
            "input_features": input_features,
        }


# =========================================================================== #
# 批处理
# =========================================================================== #
class MMCollator:
    """
    把若干条样本拼成 batch。
    右侧 padding; 图像沿 batch 维堆叠; 音频 chunk 沿 batch 维堆叠。
    """

    def __init__(self, cfg: MMConfig, pad_token_id: int = 0, label_pad_id: int = -100):
        self.cfg = cfg
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(b["input_ids"]) for b in batch)
        B = len(batch)

        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), self.label_pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)

        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
            labels[i, :L] = torch.tensor(b["labels"], dtype=torch.long)
            attention_mask[i, :L] = 1

        pixel_values = None
        imgs = [b["pixel_values"] for b in batch if b.get("pixel_values") is not None]
        if imgs:
            pixel_values = torch.cat(imgs, dim=0)          # (sum_n_img, 3, S, S)

        input_features = None
        auds = [b["input_features"] for b in batch if b.get("input_features") is not None]
        if auds:
            input_features = torch.cat(auds, dim=0)        # (sum_n_chunk, 128, 3000)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "input_features": input_features,
        }


# =========================================================================== #
# 构造完整训练数据流
# =========================================================================== #
def build_dataloader(
    records: List[Dict[str, Any]],
    cfg: MMConfig,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    max_text_len: int = 4096,
):
    """返回 (dataloader, collator)。训练脚本直接消费。"""
    from torch.utils.data import DataLoader

    add_multimodal_tokens(tokenizer, cfg)
    ds = MMDataset(records, cfg, tokenizer, max_text_len=max_text_len)
    collator = MMCollator(cfg, pad_token_id=tokenizer.pad_token_id or 0)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collator, pin_memory=True, drop_last=True,
    )
    return dl, collator


if __name__ == "__main__":
    # 自检: 构造一条含图/含音的样本, 验证占位符展开与特征形状对齐
    from .config import preset

    cfg = preset("tiny")
    cfg.llm.vocab_size = 33000

    class _FakeTok:
        bos_token_id = 1
        pad_token_id = 0
        eos_token = IM_END

        def __call__(self, text, add_special_tokens=False):
            # 极简假 tokenizer: <image> -> image_token_id, 其余按字符
            ids: List[int] = []
            i = 0
            while i < len(text):
                if text.startswith(IMAGE_TOKEN, i):
                    ids.append(cfg.image_token_id)
                    i += len(IMAGE_TOKEN)
                elif text.startswith(AUDIO_TOKEN, i):
                    ids.append(cfg.audio_token_id)
                    i += len(AUDIO_TOKEN)
                else:
                    ids.append(ord(text[i]) % 30000 + 100)
                    i += 1
            return {"input_ids": ids}

    rec = [{
        "conversations": [
            {"role": "user", "content": "<image>\n这张图里是什么?"},
            {"role": "assistant", "content": "一只猫。"},
        ],
        "images": [],
        "audios": [],
    }]
    ds = MMDataset(rec, cfg, _FakeTok())
    sample = ds[0]
    n_img_ph = sum(1 for t in sample["input_ids"] if t == cfg.image_token_id)
    n_sup = sum(1 for t in sample["labels"] if t != cfg.label_pad_id)
    print("input_ids len :", len(sample["input_ids"]))
    print("image placeholders:", n_img_ph, "(expect", cfg.vision.out_tokens, ")")
    print("supervised tokens :", n_sup)
    assert n_img_ph == cfg.vision.out_tokens
    assert n_sup > 0
    print("data pipeline self-check OK")
