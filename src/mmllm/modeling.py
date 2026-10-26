"""
多模态大模型网络结构参考实现 (From-Scratch Reference Implementation)
===================================================================
一个自包含的 PyTorch 实现, 完整定义三模态 (文本 / 图像 / 音频) 的网络结构。

    文本  -> TokenEmbedding --------------------------------\
    图像  -> ViTTower -> PixelShuffle/Perceiver -> Projector -> 拼接成一条序列 -> Decoder
    音频  -> ConvFrontend+Transformer -> TemporalPool -> Projector ->/              -> LM Head

核心设计 (为什么这么设计)
--------------------------
1. **单序列融合 (early fusion / prefix concat)**
   视觉与音频特征经投影后直接作为"虚拟 token"拼进文本序列, 与 LLaVA / Fuyu 同源。
   好处: 只有一条 KV cache、只有一条推理路径, 可直接吃 vLLM/SGLang 的标准多模态接口,
   无需 cross-attention 旁路。代价: 视觉/音频 token 会占用上下文长度, 需要压缩策略。

2. **固定 token 预算**
   每张图压缩为固定 Nv 个 token, 每 30s 音频压缩为固定 Na 个 token。
   这样占位符数量在拼 prompt 阶段即可静态展开, 推理时 shape 静态, 可开 CUDA Graph。

3. **子塔解耦冻结**
   三个塔可以独立 freeze/unfreeze, 支撑"对齐 -> 指令微调 -> 全模态 SFT"的分阶段训练。

4. **统一原语**
   ViT / 音频塔 / LLM 复用同一套 Attention + Norm + MLP 基元, 只靠配置区分
   (causal? GQA? SwiGLU 还是 GELU-MLP?)。结构清晰, 便于逐层讲解和扩展。

作者注: 本文件为结构参考实现, 精度对齐 HF 官方权重需做 key 映射(见 convert_hf_llm_state_dict)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MMConfig, preset


# =========================================================================== #
# 基础算子
# =========================================================================== #
class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm —— LLaMA / Qwen 标配, 无均值中心化, 更省算力。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class LayerNorm(nn.Module):
    """标准 LayerNorm —— ViT / Whisper 风格塔使用。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, H, L, head_dim)"""
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """RoPE —— 用旋转把相对位置编码进 Q/K, 外推能力优于绝对位置编码。"""

    def __init__(self, dim: int, base: float = 10000.0, max_pos: int = 32768):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_pos)

    def _build_cache(self, max_pos: int) -> None:
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)          # (T, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype) -> Tuple[Tensor, Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(max(seq_len, self.cos_cached.shape[0] * 2))
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]     # (1,1,L,dim)


class Attention(nn.Module):
    """
    统一多头注意力。
      - causal=True  : LLM 自回归解码 (加因果掩码)
      - causal=False : ViT / 音频塔的双向编码
      - num_kv_heads < num_heads : GQA, 省 KV cache 与显存带宽
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        causal: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 32768,
        attn_impl: str = "sdpa",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.causal = causal
        self.use_rope = use_rope
        self.attn_impl = attn_impl
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, rope_theta, max_position_embeddings)

    def _shape(self, t: Tensor, heads: int) -> Tensor:
        B, L, _ = t.shape
        return t.view(B, L, heads, self.head_dim).transpose(1, 2)      # (B,H,L,hd)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ):
        B, L, _ = x.shape
        q = self._shape(self.q_proj(x), self.num_heads)
        k = self._shape(self.k_proj(x), self.num_kv_heads)
        v = self._shape(self.v_proj(x), self.num_kv_heads)
        q_len = q.shape[2]

        # ---- RoPE: 必须先旋转"本次新算出的" q/k, 再与缓存拼接 ----
        # 顺序不能反。缓存里的 k 已经带旋转信息; 若先拼接再整体旋转,
        # 会用错误的位置对已缓存的 key 重复施加旋转, 导致增量解码结果
        # 与"一次性前向"不一致(贪心采样下会肉眼可见地跑偏)。
        if self.use_rope:
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            cos, sin = self.rope(offset + q_len, x.device, x.dtype)
            if offset > 0:
                cos = cos[:, :, offset:offset + q_len, :]
                sin = sin[:, :, offset:offset + q_len, :]
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        present = (k, v) if use_cache else None

        # GQA: KV 头复制到 Q 头数
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if self.attn_impl == "flash2":
            from flash_attn import flash_attn_func
            out = flash_attn_func(q, k, v, causal=self.causal)
            out = out.reshape(B, q_len, -1)
        else:
            is_causal = self.causal and attn_mask is None and q_len > 1
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale
            )
            out = out.transpose(1, 2).reshape(B, q_len, -1)

        return self.o_proj(out), present


class SwiGLUMLP(nn.Module):
    """SwiGLU —— Llama/Qwen 风格: 3 个矩阵, 门控增强表达力。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GELUMLP(nn.Module):
    """经典 2 矩阵前馈 —— ViT / Whisper 风格。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def make_norm(dim: int, mode: str, eps: float) -> nn.Module:
    return RMSNorm(dim, eps) if mode == "rms" else LayerNorm(dim, eps)


def make_mlp(hidden_size: int, intermediate_size: int, kind: str) -> nn.Module:
    return SwiGLUMLP(hidden_size, intermediate_size) if kind == "swiglu" else GELUMLP(hidden_size, intermediate_size)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block —— 三塔共用的最小单元。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        causal: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 32768,
        norm_mode: str = "rms",
        mlp_kind: str = "swiglu",
        attn_impl: str = "sdpa",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        inter = intermediate_size or hidden_size * 4
        self.input_norm = make_norm(hidden_size, norm_mode, norm_eps)
        self.attn = Attention(
            hidden_size, num_heads, num_kv_heads, head_dim, causal, use_rope,
            rope_theta, max_position_embeddings, attn_impl,
        )
        self.post_attn_norm = make_norm(hidden_size, norm_mode, norm_eps)
        self.mlp = make_mlp(hidden_size, inter, mlp_kind)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False):
        h, present = self.attn(self.input_norm(x), attn_mask, past_kv, use_cache)
        x = x + h
        x = x + self.mlp(self.post_attn_norm(x))
        return x, present


# =========================================================================== #
# 语言塔: decoder-only 自回归 Transformer
# =========================================================================== #
class LanguageModel(nn.Module):
    """Decoder-only LLM 骨架: Embedding -> N x Block -> FinalNorm -> LMHead"""

    def __init__(self, cfg: MMConfig):
        super().__init__()
        c = cfg.llm
        self.cfg = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=c.hidden_size,
                num_heads=c.num_heads,
                num_kv_heads=c.num_kv_heads,
                head_dim=c.head_dim,
                intermediate_size=c.intermediate_size,
                causal=True,
                use_rope=True,
                rope_theta=c.rope_theta,
                max_position_embeddings=c.max_position_embeddings,
                norm_mode=c.norm_mode,
                mlp_kind="swiglu",
                attn_impl=c.attn_impl,
                norm_eps=c.rms_norm_eps,
            )
            for _ in range(c.num_layers)
        ])
        self.final_norm = make_norm(c.hidden_size, c.norm_mode, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ):
        B, L = (input_ids.shape if input_ids is not None else inputs_embeds.shape[:2])
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

        # 已有 KV cache 时, 本次只前向 L 个新 token, 但要 attend 全部 offset+L 个位置
        offset = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total = offset + L

        # --- 注意力掩码 ---
        # 快路径: 无 padding、无 cache -> 直接用 SDPA 的 is_causal(不物化 L x L 掩码)
        # 慢路径: 有 padding 或有 cache -> 显式构造 (B,1,L,total) 的 bool 掩码
        #         SDPA 的 bool 掩码语义为 True=参与注意, False=屏蔽
        # 注意掩码的 query 维只取本次实际前向的 L 行(offset 之前的行是缓存, 不参与)
        attn_mask = None
        if attention_mask is not None or (offset > 0 and L > 1):
            if L > 1:
                causal = torch.ones(total, total, dtype=torch.bool, device=h.device).triu(1)
                blocked = causal[offset:offset + L, :][None, None]      # (1,1,L,total)
            else:
                # 单 token 解码: 因果约束自动满足, 只需处理 padding
                blocked = torch.zeros(1, 1, 1, total, dtype=torch.bool, device=h.device)
            if attention_mask is not None:
                pad = attention_mask
                if pad.shape[1] < total:
                    # 只给了本次新增部分的 mask, 前面的缓存部分视为有效
                    pad = torch.cat(
                        [torch.ones(B, total - pad.shape[1], dtype=pad.dtype, device=pad.device), pad],
                        dim=1,
                    )
                blocked = blocked | (~pad.bool())[:, None, None, :]
            attn_mask = ~blocked            # True = 允许注意

        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            h, present = layer(h, attn_mask, pkv, use_cache)
            if use_cache:
                presents.append(present)

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits, presents


# =========================================================================== #
# 视觉塔: ViT
# =========================================================================== #
class VisionTower(nn.Module):
    """
    ViT 图像编码器。
      输入 (B, 3, H, W) -> patch embedding -> +pos -> N x (双向) Block -> (B, P, D)
    默认接 CLIP-ViT-L/14-336 的权重 (通过 key 映射), 从而继承图文对齐先验。
    """

    def __init__(self, cfg: MMConfig):
        super().__init__()
        c = cfg.vision
        self.cfg = c
        self.image_size = c.image_size
        self.patch_size = c.patch_size
        self.hidden_size = c.hidden_size
        self.num_patches = c.num_patches

        self.patch_embed = nn.Conv2d(3, c.hidden_size, kernel_size=c.patch_size, stride=c.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, c.num_patches, c.hidden_size))
        self.pre_layrnorm = LayerNorm(c.hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=c.hidden_size,
                num_heads=c.num_heads,
                intermediate_size=c.hidden_size * 4,
                causal=False,
                use_rope=False,
                norm_mode="layer",
                mlp_kind="gelu",
                attn_impl="sdpa",
            )
            for _ in range(c.num_layers)
        ])
        self.post_layernorm = LayerNorm(c.hidden_size)
        self.apply(LanguageModel._init_weights)
        nn.init.normal_(self.pos_embed, std=0.01)

    def forward(self, pixel_values: Tensor) -> Tensor:
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)   # (B, P, D)
        x = self.pre_layrnorm(x + self.pos_embed)
        for blk in self.blocks:
            x, _ = blk(x)
        return self.post_layernorm(x)                                    # (B, P, D)


# =========================================================================== #
# 音频塔: Whisper 式卷积前端 + 双向 Transformer
# =========================================================================== #
class AudioTower(nn.Module):
    """
    Whisper 风格音频编码器 (只取 encoder, 不含 decoder)。
      输入 (B, n_mels, 3000) -> Conv1d(核3/步2) x2 -> (B, D, 1500) -> +pos
      -> N x 双向 Block -> (B, 1500, D)
    """

    def __init__(self, cfg: MMConfig):
        super().__init__()
        c = cfg.audio
        self.cfg = c
        self.hidden_size = c.hidden_size
        self.max_frames = c.encoder_frames

        self.conv1 = nn.Conv1d(c.n_mels, c.hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(c.hidden_size, c.hidden_size, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding",
                             torch.zeros(1, self.max_frames, c.hidden_size), persistent=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=c.hidden_size,
                num_heads=c.num_heads,
                intermediate_size=c.hidden_size * 4,
                causal=False,
                use_rope=False,
                norm_mode="layer",
                mlp_kind="gelu",
                attn_impl="sdpa",
            )
            for _ in range(c.num_layers)
        ])
        self.layer_norm = LayerNorm(c.hidden_size)
        self.apply(LanguageModel._init_weights)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, input_features: Tensor) -> Tensor:
        """input_features: (B, n_mels, 3000) 已做 log-Mel 归一化"""
        x = F.gelu(self.conv1(input_features))     # (B, D, 3000)
        x = F.gelu(self.conv2(x))                  # (B, D, 1500)
        x = x.transpose(1, 2)                      # (B, 1500, D)
        x = x + self.positional_embedding[:, : x.shape[1], :].to(x.dtype)
        for blk in self.blocks:
            x, _ = blk(x)
        return self.layer_norm(x)                 # (B, 1500, D)


# =========================================================================== #
# 投影器: 模态空间 -> LLM token 空间的桥
# =========================================================================== #
def _choose_grid(n_tokens: int, gh: int, gw: int) -> Tuple[int, int]:
    """
    在保持长宽比的前提下, 找出 (th, tw) 使 th*tw == n_tokens。
    用于把 ViT 的 patch 网格重采样到目标 token 数。
    """
    best = None
    for th in range(1, int(math.sqrt(n_tokens)) + 1):
        if n_tokens % th != 0:
            continue
        tw = n_tokens // th
        score = abs(th / tw - gh / gw)
        if best is None or score < best[0]:
            best = (score, th, tw)
    if best is None:
        return (1, n_tokens)
    return (best[1], best[2])


class MLPProjector(nn.Module):
    """
    视觉投影器: 把 ViT 输出的 N 个 patch token 压缩成**恰好 out_tokens 个** token,
    再投影到 LLM 的 token 空间。

        Linear(D_in -> D*mult) -> GELU -> Linear(D*mult -> D) -> LayerNorm

    压缩方式(当给出 out_tokens 与 grid_in 时):
        把 patch 网格 (gh, gw) 用 2D 自适应平均池化重采样到 (th, tw),
        其中 th*tw == out_tokens, 且 th/tw 尽量接近 gh/gw(保持长宽比)。
        576(24x24) -> 256(16x16) -> 144(12x12) -> 64(8x8) 都能精确命中。

    为什么不用固定 2x2 的 PixelShuffle:
        PixelShuffle 只能把 token 数降为 1/4, 无法落到任意目标值。
        而"配置里的 out_tokens 必须等于模型实际输出的 token 数"是
        **占位符契约成立的前提** —— 否则 <image> 占位符数量与特征数量对不上,
        特征会静默错位到别的图片上。宁可让配置说话, 也不要让实现偷偷改口径。
    """

    def __init__(self, in_dim: int, out_dim: int, out_tokens: Optional[int] = None,
                 grid_in: Optional[Tuple[int, int]] = None, mult: int = 4,
                 dropout: float = 0.0, norm_out: bool = True):
        super().__init__()
        self.out_tokens = out_tokens
        self.grid_in = grid_in
        self.grid_out = None
        if out_tokens is not None and grid_in is not None:
            gh, gw = grid_in
            self.grid_out = _choose_grid(out_tokens, gh, gw)
        self.fc1 = nn.Linear(in_dim, out_dim * mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim * mult, out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = LayerNorm(out_dim) if norm_out else nn.Identity()
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.grid_out is not None:
            B, N, C = x.shape
            gh, gw = self.grid_in
            th, tw = self.grid_out
            assert gh * gw == N, f"patch 网格 {gh}x{gw} 与 token 数 {N} 不符"
            x = x.transpose(1, 2).reshape(B, C, gh, gw)
            x = F.adaptive_avg_pool2d(x, (th, tw))      # (B, C, th, tw)
            x = x.flatten(2).transpose(1, 2)             # (B, th*tw, C) == (B, out_tokens, C)
        return self.norm(self.fc2(self.drop(self.act(self.fc1(x)))))


class PerceiverResampler(nn.Module):
    """
    Flamingo 式 Perceiver Resampler: 用 N 个可学习 latent 向量跨注意力汇聚 M 个输入 token。
    输出固定 N 个 token, 且天然对输入长度不敏感 —— 是压缩视觉/音频 token 的主力方案。
    """

    def __init__(self, in_dim: int, out_dim: int, num_latents: int = 256,
                 num_heads: int = 16, depth: int = 2, mlp_mult: int = 4):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, out_dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(out_dim, num_heads, batch_first=True) for _ in range(depth)
        ])
        self.norm_cross = nn.ModuleList([LayerNorm(out_dim) for _ in range(depth)])
        self.norm_ffn = nn.ModuleList([LayerNorm(out_dim) for _ in range(depth)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(out_dim, out_dim * mlp_mult), nn.GELU(),
                          nn.Linear(out_dim * mlp_mult, out_dim))
            for _ in range(depth)
        ])
        self.out_norm = LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.in_proj(x)                                     # (B, M, D)
        lat = self.latents.expand(B, -1, -1).to(x.dtype)        # (B, N, D)
        for cross, nc, nf, ffn in zip(self.cross_attn, self.norm_cross, self.norm_ffn, self.ffn):
            q = nc(lat)
            out, _ = cross(q, x, x, need_weights=False)          # latent 读取输入
            lat = lat + out
            lat = lat + ffn(nf(lat))
        return self.out_norm(lat)


class AudioProjector(nn.Module):
    """
    音频投影器: 时域压缩 + MLP。
      (B, 1500, D_audio) -> Conv1d(stride) -> (B, T', D_audio) -> MLP -> (B, Na, D_llm)
    压缩倍数需让 Na 与 30s 音频的语义节奏匹配 (Whisper 约 2 token/秒 -> 30s ≈ 60;
    这里取 25, 约 0.83 token/秒, 更省上下文)。
    """

    def __init__(self, in_dim: int, out_dim: int, out_tokens: int, mult: int = 4,
                 dropout: float = 0.0, norm_out: bool = True):
        super().__init__()
        self.out_tokens = out_tokens
        self.temporal = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=8, stride=4, padding=2), nn.GELU(),
            nn.Conv1d(in_dim, in_dim, kernel_size=5, stride=5, padding=2), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(out_tokens)
        self.fc1 = nn.Linear(in_dim, out_dim * mult)
        self.fc2 = nn.Linear(out_dim * mult, out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = LayerNorm(out_dim) if norm_out else nn.Identity()
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        h = self.temporal(x.transpose(1, 2))          # (B, D, T')
        h = self.pool(h)                               # (B, D, Na)
        h = h.transpose(1, 2)                         # (B, Na, D)
        return self.norm(self.fc2(self.drop(F.gelu(self.fc1(h)))))


# =========================================================================== #
# 主模型: 多模态大模型
# =========================================================================== #
@dataclass
class MMOutput:
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None


class MultimodalLM(nn.Module):
    """
    文本 + 图像 + 音频 统一自回归模型。

    forward 流程:
      1. 文本 -> embed_tokens
      2. 图像 -> VisionTower -> (VisionProjector) -> (B, Nv, D)
      3. 音频 -> AudioTower  -> (AudioProjector)  -> (B, Na, D)
      4. 按占位符掩码把 2/3 的特征"原位替换"进第 1 步的 embedding
      5. LanguageModel(inputs_embeds) -> logits -> loss
    """

    def __init__(self, cfg: MMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.language_model = LanguageModel(cfg)
        self.vision_tower = VisionTower(cfg)
        self.audio_tower = AudioTower(cfg)

        gh = gw = cfg.vision.image_size // cfg.vision.patch_size
        if cfg.vision.pool_mode == "perceiver":
            self.vision_resampler = PerceiverResampler(
                in_dim=cfg.vision.hidden_size, out_dim=d,
                num_latents=cfg.vision.out_tokens, num_heads=cfg.llm.num_heads,
            )
            # 特征已被压成 out_tokens 个, 投影器只做线性映射
            self.vision_projector = MLPProjector(
                in_dim=d, out_dim=d, out_tokens=None, grid_in=None,
                mult=cfg.projector_hidden_mult, dropout=cfg.projector_dropout,
                norm_out=cfg.projector_norm_out,
            )
        else:
            # mlp 模式: 由投影器把 patch 网格重采样到恰好 out_tokens 个 token
            self.vision_resampler = None
            self.vision_projector = MLPProjector(
                in_dim=cfg.vision.hidden_size, out_dim=d,
                out_tokens=cfg.vision.out_tokens, grid_in=(gh, gw),
                mult=cfg.projector_hidden_mult, dropout=cfg.projector_dropout,
                norm_out=cfg.projector_norm_out,
            )

        self.audio_projector = AudioProjector(
            in_dim=cfg.audio.hidden_size, out_dim=d,
            out_tokens=cfg.audio.out_tokens_per_chunk,
            mult=cfg.projector_hidden_mult, dropout=cfg.projector_dropout,
            norm_out=cfg.projector_norm_out,
        )

    # ---------------- 冻结控制 ---------------- #
    def set_trainable(self, llm: bool = True, vision: bool = False, audio: bool = False,
                      projector: bool = True):
        """分阶段训练的核心开关。Stage1 只训 projector; Stage2 解冻 llm。"""
        for p in self.language_model.parameters():
            p.requires_grad_(llm)
        for p in self.vision_tower.parameters():
            p.requires_grad_(vision)
        for p in self.audio_tower.parameters():
            p.requires_grad_(audio)
        for m in (self.vision_projector, self.audio_projector):
            for p in m.parameters():
                p.requires_grad_(projector)
        if self.vision_resampler is not None:
            for p in self.vision_resampler.parameters():
                p.requires_grad_(projector)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---------------- 模态编码 ---------------- #
    def encode_images(self, pixel_values: Tensor) -> Tensor:
        """-> (B, Nv, D)"""
        feats = self.vision_tower(pixel_values)                 # (B, P, D_vis)
        if self.vision_resampler is not None:
            feats = self.vision_resampler(feats)                # (B, Nv, D)
        return self.vision_projector(feats) * self.cfg.projector_scale

    def encode_audio(self, input_features: Tensor) -> Tensor:
        """-> (B, Na, D)"""
        feats = self.audio_tower(input_features)                # (B, 1500, D_aud)
        return self.audio_projector(feats) * self.cfg.projector_scale

    # ---------------- 序列组装 ---------------- #
    @staticmethod
    def _scatter(embeddings: Tensor, mask: Tensor, features: Tensor) -> Tensor:
        """
        把模态特征按占位符掩码"原位替换"进文本 embedding。

            embeddings : (B, L, D)
            mask       : (B, L) bool, True 的位置是占位符
            features   : (N_items, K, D)
                         N_items = 整个 batch 的模态项总数(所有样本沿 batch 维堆叠)
                         K       = 每项的 token 数(视觉=out_tokens, 音频=每 chunk token 数)

        为什么不能简单按 batch 维对齐:
            同一个 batch 里, 不同样本的图片/音频数量不同(有的 0 张, 有的 2 张)。
            若假设 features 第 b 行属于样本 b, 就会张冠李戴。
            因此这里按"每样本的模态项数"算出每个样本在 features 中的起始行(偏移量),
            再用扁平索引精确取回对应特征。差分且可导, 梯度直接回传到对应模态塔。
        """
        B, L, D = embeddings.shape
        N_items, K, _ = features.shape
        flat = features.reshape(N_items * K, D)

        per_sample_counts = mask.sum(dim=1)                       # (B,) 每样本占位符数
        if not bool(torch.all(per_sample_counts % K == 0)):
            bad = int(torch.nonzero(per_sample_counts % K != 0)[0, 0])
            raise ValueError(
                f"样本 {bad} 的占位符数 {int(per_sample_counts[bad])} 不是 {K} 的整数倍; "
                f"每个模态项必须恰好展开 {K} 个占位符。"
            )
        item_counts = per_sample_counts // K                      # (B,) 每样本模态项数
        offsets = torch.cumsum(item_counts, dim=0) - item_counts  # exclusive cumsum

        cnt = mask.long().cumsum(dim=1)                           # (B,L) 含当前位的计数
        within = (cnt - 1) % K                                   # 项内第几个 token
        item_idx = (cnt - 1) // K                                # 样本内第几个模态项
        global_row = offsets.unsqueeze(1) + item_idx              # (B,L) 全局模态项号
        flat_idx = (global_row * K + within).clamp(min=0, max=N_items * K - 1)

        gathered = flat.gather(0, flat_idx.reshape(-1, 1).expand(-1, D)).view(B, L, D)
        return torch.where(mask.unsqueeze(-1), gathered, embeddings)

    def prepare_inputs_embeds(
        self,
        input_ids: Tensor,
        pixel_values: Optional[Tensor] = None,
        input_features: Optional[Tensor] = None,
    ) -> Tensor:
        """把多模态特征融合进 token embedding。推理与训练共用这一条路径。"""
        embeds = self.language_model.embed_tokens(input_ids)
        cfg = self.cfg

        if pixel_values is not None and pixel_values.numel() > 0:
            vis = self.encode_images(pixel_values.to(embeds.dtype))
            mask = input_ids == cfg.image_token_id
            n_placeholder = int(mask.sum().item())
            if n_placeholder != vis.shape[0] * vis.shape[1]:
                raise ValueError(
                    f"图像占位符数量({n_placeholder}) 与视觉特征数量"
                    f"({vis.shape[0] * vis.shape[1]}) 不匹配; "
                    f"每图应展开 {cfg.vision.out_tokens} 个 <image> 占位符。"
                )
            embeds = self._scatter(embeds, mask, vis)

        if input_features is not None and input_features.numel() > 0:
            aud = self.encode_audio(input_features.to(embeds.dtype))
            mask = input_ids == cfg.audio_token_id
            n_placeholder = int(mask.sum().item())
            if n_placeholder != aud.shape[0] * aud.shape[1]:
                raise ValueError(
                    f"音频占位符数量({n_placeholder}) 与音频特征数量"
                    f"({aud.shape[0] * aud.shape[1]}) 不匹配; "
                    f"每 chunk 应展开 {cfg.audio.out_tokens_per_chunk} 个 <audio> 占位符。"
                )
            embeds = self._scatter(embeds, mask, aud)

        return embeds

    # ---------------- 训练 ---------------- #
    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        input_features: Optional[Tensor] = None,
    ) -> MMOutput:
        inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values, input_features)
        logits, _ = self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.cfg.label_pad_id,
            )
        return MMOutput(loss=loss, logits=logits)

    # ---------------- 推理 ---------------- #
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        pixel_values: Optional[Tensor] = None,
        input_features: Optional[Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ):
        """
        自回归解码, 支持 KV cache。
        注意: 多模态前置特征只在第一步注入, 后续步只喂新 token。
        """
        self.eval()
        inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values, input_features)
        B = input_ids.shape[0]
        out_ids: List[Tensor] = []

        # --- prefill ---
        logits, presents = self.language_model(
            inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True
        )

        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits.float() / temperature, dim=-1)
                sp, si = torch.sort(probs, descending=True, dim=-1)
                cum = sp.cumsum(dim=-1)
                mask = cum - sp > top_p
                sp[mask] = 0.0
                sp = sp / sp.sum(dim=-1, keepdim=True)
                choice = torch.multinomial(sp.view(-1, sp.size(-1)), 1)
                next_token = si.view(B, -1).gather(1, choice)

            out_ids.append(next_token)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # --- decode: 只喂新 token 的 embedding ---
            step_embeds = self.language_model.embed_tokens(next_token)
            logits, presents = self.language_model(
                inputs_embeds=step_embeds, past_key_values=presents, use_cache=True
            )

        return torch.cat(out_ids, dim=1) if out_ids else input_ids.new_zeros((B, 0))

    # ---------------- 保存 / 加载 ---------------- #
    def save_pretrained(self, out_dir: str):
        import json, os
        from safetensors.torch import save_file
        os.makedirs(out_dir, exist_ok=True)
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, os.path.join(out_dir, "model.safetensors"))
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, out_dir: str, map_location="cpu") -> "MultimodalLM":
        import json, os
        from safetensors.torch import load_file
        with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
            cfg = MMConfig.from_dict(json.load(f))
        model = cls(cfg)
        state = load_file(os.path.join(out_dir, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[load] missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}")
        return model


# =========================================================================== #
# HF 权重互转
# =========================================================================== #
def convert_hf_llm_state_dict(hf_sd: dict) -> dict:
    """
    把 HuggingFace Llama/Qwen 的 state_dict 映射到本实现的 LanguageModel 命名。
    映射规则:
        model.embed_tokens.weight            -> embed_tokens.weight
        model.layers.{i}.self_attn.q_proj.*  -> layers.{i}.attn.q_proj.*
        model.layers.{i}.mlp.gate_proj.*     -> layers.{i}.mlp.gate_proj.*
        model.layers.{i}.input_layernorm.*   -> layers.{i}.input_norm.*
        model.layers.{i}.post_attention_layernorm.* -> layers.{i}.post_attn_norm.*
        model.norm.weight                     -> final_norm.weight
        lm_head.weight                       -> lm_head.weight
    """
    out = {}
    for k, v in hf_sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("layers."):
            parts = k.split(".")
            i = parts[1]
            rest = ".".join(parts[2:])
            rest = rest.replace("self_attn.", "attn.")
            rest = rest.replace("input_layernorm.", "input_norm.")
            rest = rest.replace("post_attention_layernorm.", "post_attn_norm.")
            k = f"layers.{i}.{rest}"
        elif k == "norm.weight":
            k = "final_norm.weight"
        out[k] = v
    return out


# =========================================================================== #
# 形状 / 梯度 自检
# =========================================================================== #
def _selfcheck():
    import torch
    torch.manual_seed(0)

    cfg = preset("tiny")
    cfg.llm.vocab_size = 33000
    cfg.image_token_id = 32001
    cfg.audio_token_id = 32002
    model = MultimodalLM(cfg)

    NV = cfg.vision.out_tokens              # tiny: 64
    NA = cfg.audio.out_tokens_per_chunk     # tiny: 12
    B = 2
    # row0: [文本][64 视觉][文本][12 音频][文本]
    # row1: [文本][64 视觉][文本]
    row0 = ([100] * 5 + [cfg.image_token_id] * NV + [101] * 5
            + [cfg.audio_token_id] * NA + [102] * 5)
    row1 = ([100] * 5 + [cfg.image_token_id] * NV + [101] * 10)
    L = max(len(row0), len(row1))
    ids = torch.full((B, L), 0, dtype=torch.long)
    ids[0, :len(row0)] = torch.tensor(row0)
    ids[1, :len(row1)] = torch.tensor(row1)

    px = torch.randn(B, 3, cfg.vision.image_size, cfg.vision.image_size)
    af = torch.randn(1, cfg.audio.n_mels, 3000)          # 1 个音频 chunk
    labels = ids.clone()
    labels[labels == cfg.image_token_id] = cfg.label_pad_id
    labels[labels == cfg.audio_token_id] = cfg.label_pad_id

    print("=" * 66)
    print(f"配置: LLM {cfg.llm.num_layers}L/{cfg.llm.hidden_size}d "
          f"| 视觉 {cfg.vision.num_patches} patch -> {NV} token "
          f"| 音频 {cfg.audio.encoder_frames} 帧 -> {NA} token")
    print("=" * 66)

    # --- 前向 ---
    model.set_trainable(llm=True, vision=True, audio=True, projector=True)
    out = model(input_ids=ids, labels=labels, pixel_values=px, input_features=af)
    print(f"input_ids      {tuple(ids.shape)}")
    print(f"logits         {tuple(out.logits.shape)}")
    print(f"loss           {float(out.loss.detach()):.4f}")

    # --- 特征形状核对 ---
    vis = model.encode_images(px)
    aud = model.encode_audio(af)
    print(f"视觉特征       {tuple(vis.shape)}   (B_img, Nv, D)")
    print(f"音频特征       {tuple(aud.shape)}   (B_chunk, Na, D)")
    assert vis.shape[1] == NV and aud.shape[1] == NA

    # --- 占位符 / 特征数量一致性 ---
    n_img_ph = int((ids == cfg.image_token_id).sum())
    n_aud_ph = int((ids == cfg.audio_token_id).sum())
    assert n_img_ph == B * NV, f"图像占位符 {n_img_ph} != {B * NV}"
    assert n_aud_ph == 1 * NA, f"音频占位符 {n_aud_ph} != {NA}"
    print(f"占位符校验     图像 {n_img_ph} / 视觉特征 {vis.shape[0] * vis.shape[1]}  ✓")
    print(f"               音频 {n_aud_ph} / 音频特征 {aud.shape[0] * aud.shape[1]}  ✓")

    # --- 反向: 三塔梯度必须都能回传 ---
    out.loss.backward()
    grads = {}
    for name, mod in (("vision_tower", model.vision_tower),
                      ("audio_tower", model.audio_tower),
                      ("vision_projector", model.vision_projector),
                      ("audio_projector", model.audio_projector),
                      ("language_model", model.language_model)):
        g = sum(float(p.grad.abs().sum()) for p in mod.parameters()
                if p.grad is not None)
        grads[name] = g
        print(f"梯度 {name:<20} |grad| = {g:.4f}")
    for k, v in grads.items():
        assert v > 0, f"{k} 没有收到梯度"

    # --- 冻结开关校验 ---
    model.set_trainable(llm=False, vision=False, audio=False, projector=True)
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print("-" * 66)
    print(f"Stage1(只训投影器) 可训练 {n_tr/1e6:.2f}M / 总计 {n_all/1e6:.2f}M "
          f"({100*n_tr/n_all:.2f}%)")

    # --- 增量解码一致性: 带 KV cache 的 generate 必须等于逐 token 全量前向 ---
    # 这条断言专门盯 RoPE 与因果掩码在增量路径上的实现, 是最容易悄悄跑偏的地方。
    model.eval()
    prompt = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
    with_cache = model.generate(prompt, max_new_tokens=12, temperature=0.0)

    manual = []
    cur = prompt.clone()
    for _ in range(12):
        with torch.no_grad():
            lg, _ = model.language_model(input_ids=cur)
        nxt = lg[:, -1, :].argmax(dim=-1, keepdim=True)
        manual.append(nxt)
        cur = torch.cat([cur, nxt], dim=1)
    manual = torch.cat(manual, dim=1)

    same = bool(torch.equal(with_cache, manual))
    print(f"增量解码一致性   cache={with_cache.tolist()} 无cache={manual.tolist()}")
    print(f"                 {'✓ 一致' if same else '✗ 不一致 —— KV cache 路径有 bug'}")
    assert same, "带 KV cache 的 generate 与全量前向结果不一致"

    print("全部断言通过 ✓")
    return model


if __name__ == "__main__":
    _selfcheck()
