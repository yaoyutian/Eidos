"""
多模态大模型配置契约 (Config Contract)
========================================
所有模块共享的唯一事实来源。训练、导出、服务端都从这里读取，
避免"训练时的结构"和"部署时的结构"漂移。

设计要点
--------
1. 三塔 + 单脑:
   VisionTower / AudioTower / TextEmbedding -> Projector -> LanguageModel
2. 每个 Projector 输出固定数量的 token (Nv / Na),
   这样占位符数量在 tokenizer 阶段就可静态展开,推理期无需动态 shape。
3. 所有子模块可独立冻结, 用于分阶段训练。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

PoolMode = Literal["mlp", "perceiver"]
NormMode = Literal["rms", "layer"]
AttnImpl = Literal["sdpa", "flash2"]


# --------------------------------------------------------------------------- #
# 视觉塔
# --------------------------------------------------------------------------- #
@dataclass
class VisionConfig:
    """视觉编码器 (ViT / CLIP-ViT)"""

    # 权重来源: HF hub id 或本地路径。部署时导出为 safetensors 随主权重一起分发
    backbone: str = "openai/clip-vit-large-patch14-336"

    hidden_size: int = 1024          # ViT-L/14-336 的宽度
    num_layers: int = 24
    num_heads: int = 16
    image_size: int = 336            # 输入分辨率
    patch_size: int = 14             # -> 24 x 24 = 576 个 patch

    # 投影后送入 LLM 的视觉 token 数
    # 256 是精度/延迟的平衡点; 576 = 不压缩; 64 = 极低延迟
    out_tokens: int = 256

    # 压缩方式:
    #   mlp      -> 直接 MLP 投影, token 数不变 (需靠 pixel-shuffle 之类手段减量)
    #   perceiver-> Perceiver Resampler, 用可学习 latent 跨注意力压缩到 out_tokens
    pool_mode: PoolMode = "perceiver"

    # anyres: 高分辨率切图 + 全局图多尺度输入
    anyres: bool = False
    anyres_max_tiles: int = 4        # 每图最多切几块

    freeze: bool = True              # 是否冻结
    dtype: str = "bfloat16"

    @property
    def num_patches(self) -> int:
        g = self.image_size // self.patch_size
        return g * g


# --------------------------------------------------------------------------- #
# 音频塔
# --------------------------------------------------------------------------- #
@dataclass
class AudioConfig:
    """音频编码器 (Whisper encoder 为主)"""

    backbone: str = "openai/whisper-large-v3"

    hidden_size: int = 1280           # whisper-large encoder 宽度
    num_layers: int = 32
    num_heads: int = 20

    sample_rate: int = 16000
    n_mels: int = 128
    max_seconds: int = 30            # 单 chunk 上限, 与 Whisper 训练窗口一致
    chunk_seconds: int = 30          # 切块粒度; 30s -> 1 chunk

    # 每个 chunk 压缩后送入 LLM 的 token 数
    # Whisper 30s encoder 输出 1500 帧, 压到 25 => 60x 压缩
    out_tokens_per_chunk: int = 25

    freeze: bool = True
    dtype: str = "bfloat16"

    @property
    def encoder_frames(self) -> int:
        """Whisper conv 前端下采样 2x + 30s 固定 3000 mel 帧 -> 1500"""
        return 1500


# --------------------------------------------------------------------------- #
# 语言塔 (decoder-only backbone)
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    """自回归解码器 Transformer"""

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8             # GQA: KV 头数 < Q 头数, 显著省 KV cache
    head_dim: int = 128
    intermediate_size: int = 14336    # SwiGLU
    max_position_embeddings: int = 32768

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    norm_mode: NormMode = "rms"

    # 文本 tokenizer: byte-level BPE, 保证任意字节可编码 (图像/音频不经过它)
    tokenizer: str = "meta-llama/Llama-2-7b-hf"
    add_bos: bool = True

    tie_word_embeddings: bool = False
    attn_impl: AttnImpl = "sdpa"
    dtype: str = "bfloat16"


# --------------------------------------------------------------------------- #
# 多模态总配置
# --------------------------------------------------------------------------- #
@dataclass
class MMConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    # 特殊占位 token id (需与 tokenizer 扩展保持一致)
    image_token_id: int = 32001       # <image>
    audio_token_id: int = 32002       # <audio>
    video_token_id: int = 32003       # <video>

    # 投影器
    projector_hidden_mult: int = 4    # MLP 投影器中间层倍数
    projector_dropout: float = 0.0
    projector_norm_out: bool = True   # 对投影输出做 LayerNorm, 与文本 embedding 对齐尺度
    projector_scale: float = 1.0      # 视觉/音频特征的额外缩放 (训练初期稳定)

    # 训练相关
    label_pad_id: int = -100
    tie_vision_proj: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MMConfig":
        d = dict(d)
        llm = LLMConfig(**d.pop("llm", {}))
        vision = VisionConfig(**d.pop("vision", {}))
        audio = AudioConfig(**d.pop("audio", {}))
        return cls(llm=llm, vision=vision, audio=audio, **d)

    # ---------------- 便捷派生属性 ---------------- #
    @property
    def d_model(self) -> int:
        return self.llm.hidden_size

    @property
    def n_visual_tokens(self) -> int:
        return self.vision.out_tokens

    @property
    def n_audio_tokens_per_chunk(self) -> int:
        return self.audio.out_tokens_per_chunk


# --------------------------------------------------------------------------- #
# 常见规模预设
# --------------------------------------------------------------------------- #
def preset(name: str) -> MMConfig:
    """按规模快速取一组自洽的超参, 避免手填出错。"""
    name = name.lower()

    if name in ("tiny", "0.5b"):
        return MMConfig(
            llm=LLMConfig(vocab_size=32000, hidden_size=1024, num_layers=24,
                          num_heads=16, num_kv_heads=4, head_dim=64,
                          intermediate_size=2816, max_position_embeddings=8192),
            vision=VisionConfig(backbone="openai/clip-vit-base-patch16-224",
                                hidden_size=768, num_layers=12, num_heads=12,
                                image_size=224, patch_size=16, out_tokens=64),
            audio=AudioConfig(backbone="openai/whisper-small",
                              hidden_size=768, num_layers=12, num_heads=12,
                              out_tokens_per_chunk=12),
        )

    if name in ("small", "3b"):
        return MMConfig(
            llm=LLMConfig(vocab_size=32000, hidden_size=2560, num_layers=32,
                          num_heads=32, num_kv_heads=8, head_dim=80,
                          intermediate_size=8960, max_position_embeddings=32768),
            vision=VisionConfig(out_tokens=256),
            audio=AudioConfig(out_tokens_per_chunk=25),
        )

    if name in ("base", "7b"):
        return MMConfig(
            llm=LLMConfig(vocab_size=32000, hidden_size=4096, num_layers=32,
                          num_heads=32, num_kv_heads=8, head_dim=128,
                          intermediate_size=14336, max_position_embeddings=32768),
            vision=VisionConfig(out_tokens=256),
            audio=AudioConfig(out_tokens_per_chunk=25),
        )

    if name in ("large", "70b"):
        return MMConfig(
            llm=LLMConfig(vocab_size=32000, hidden_size=8192, num_layers=80,
                          num_heads=64, num_kv_heads=8, head_dim=128,
                          intermediate_size=28672, max_position_embeddings=32768),
            vision=VisionConfig(out_tokens=256),
            audio=AudioConfig(out_tokens_per_chunk=25),
        )

    raise ValueError(f"unknown preset: {name}")
