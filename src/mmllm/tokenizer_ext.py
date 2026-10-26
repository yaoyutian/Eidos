"""
多模态 tokenizer 扩展与占位符静态展开
======================================
核心概念: **占位符静态展开 (Static Placeholder Expansion)**

    对话文本中的 <image> / <audio> 是"标记", 在送入模型前会被展开成
    固定数量的连续特殊 token:

        "<image>"  ->  [32001] * 256     (每张图 256 个视觉 token)
        "<audio>"  ->  [32002] * 25      (每 30s 音频 25 个音频 token)

为什么必须静态展开?
  1. 形状静态: 推理时 batch shape 恒定, 可用 CUDA Graph / 连续批处理
  2. 掩码对齐: 模型靠 `input_ids == image_token_id` 的掩码把特征"原位替换"回去,
     占位符数量必须精确等于特征数量, 否则位置对不上
  3. 调度友好: 长度可提前算出, 便于连续批处理与显存预估
"""

from __future__ import annotations

from typing import List, Optional

from .config import MMConfig

IMAGE_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"
VIDEO_TOKEN = "<video>"


def add_multimodal_tokens(tokenizer, cfg: MMConfig) -> int:
    """
    给 tokenizer 增加特殊 token, 并把它们的 id 写回 cfg。
    返回新增 token 数量。
    注意: 增加 token 后必须 resize token embedding, 否则 id 越界。
    """
    special = {"<image>": cfg.image_token_id,
               "<audio>": cfg.audio_token_id,
               "<video>": cfg.video_token_id}
    new_tokens = [t for t in special if t not in tokenizer.get_vocab()]
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}
    ) if new_tokens else 0

    # 回填真实 id (add_special_tokens 可能因已存在而返回不同 id)
    for tok, fallback in special.items():
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tok == "<image>":
            cfg.image_token_id = tid if tid is not None and tid >= 0 else fallback
        elif tok == "<audio>":
            cfg.audio_token_id = tid if tid is not None and tid >= 0 else fallback
        elif tok == "<video>":
            cfg.video_token_id = tid if tid is not None and tid >= 0 else fallback
    return added


def count_placeholders(text: str) -> tuple[int, int]:
    """返回 (图片占位符个数, 音频占位符个数)"""
    return text.count(IMAGE_TOKEN), text.count(AUDIO_TOKEN)


def expand_placeholders(text: str, cfg: MMConfig) -> str:
    """
    把 <image> / <audio> 展开成连续重复的占位标记串。
    实际转 token id 在 dataset 中完成(这里保持字符串形式便于可读/可存盘)。
    """
    text = text.replace(IMAGE_TOKEN, IMAGE_TOKEN * cfg.vision.out_tokens)
    text = text.replace(AUDIO_TOKEN, AUDIO_TOKEN * cfg.audio.out_tokens_per_chunk)
    return text


def expand_ids(ids: List[int], cfg: MMConfig) -> List[int]:
    """在 token id 层面做等价展开(用于已经 tokenize 过的场景)。"""
    out: List[int] = []
    for t in ids:
        if t == cfg.image_token_id:
            out.extend([cfg.image_token_id] * cfg.vision.out_tokens)
        elif t == cfg.audio_token_id:
            out.extend([cfg.audio_token_id] * cfg.audio.out_tokens_per_chunk)
        else:
            out.append(t)
    return out


def estimate_multimodal_tokens(
    n_images: int, n_audio_chunks: int, cfg: MMConfig
) -> int:
    """预估多模态部分占用的上下文 token 数 —— 用于长度预算与显存估算。"""
    return n_images * cfg.vision.out_tokens + n_audio_chunks * cfg.audio.out_tokens_per_chunk
