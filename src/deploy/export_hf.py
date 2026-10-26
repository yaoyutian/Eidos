#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_hf.py — 把本项目权重导出成 HuggingFace `LlavaForConditionalGeneration` 目录
==============================================================================

为什么要导出到 HF 目录
----------------------
vLLM / SGLang / HF 生态默认只认 HF 目录结构。把 `model.safetensors + config.json`
转换成一个标准 HF 目录, 就能直接 `vllm serve <hf_dir>`, 复用现成的
paged attention / continuous batching / prefix caching 等工程能力 (部署路径 A)。

产出目录结构
------------
    <dst>/
        config.json                        # LlavaConfig 格式 (vision_config / text_config / image_token_index ...)
        generation_config.json             # 采样默认值 (可选但推荐)
        preprocessor_config.json           # CLIPImageProcessor 配置
        processor_config.json              # LlavaProcessor 配置 (image_seq_length ...)
        model-00001-of-00002.safetensors   # 分片权重
        model-00002-of-00002.safetensors
        model.safetensors.index.json       # 分片索引 + weight_map
        mmllm_extra.safetensors            # HF Llava 无对应结构的权重 (音频塔 / Perceiver / 投影器 ...)
        mmllm_extra.json                   # 上述权重的说明
        export_report.json                 # 本次导出的完整体检报告

字段重命名映射 (双向)
---------------------
    本项目 (mmllm)                                      HuggingFace (Llava)
    ------------------------------------------------------------------------------------------
    language_model.embed_tokens.weight                 -> language_model.model.embed_tokens.weight
    language_model.layers.{i}.input_norm.weight        -> language_model.model.layers.{i}.input_layernorm.weight
    language_model.layers.{i}.post_attn_norm.weight   -> language_model.model.layers.{i}.post_attention_layernorm.weight
    language_model.layers.{i}.attn.q_proj.weight       -> language_model.model.layers.{i}.self_attn.q_proj.weight
    language_model.layers.{i}.mlp.gate_proj.weight    -> language_model.model.layers.{i}.mlp.gate_proj.weight
    language_model.final_norm.weight                  -> language_model.model.norm.weight
    language_model.lm_head.weight                     -> language_model.lm_head.weight
    ------------------------------------------------------------------------------------------
    vision_tower.patch_embed.weight                    -> vision_tower.vision_model.embeddings.patch_embedding.weight
    vision_tower.pos_embed                            -> vision_tower.vision_model.embeddings.position_embedding.weight
    vision_tower.pre_layrnorm.weight                  -> vision_tower.vision_model.pre_layrnorm.weight
    vision_tower.blocks.{i}.input_norm.weight         -> vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight
    vision_tower.blocks.{i}.post_attn_norm.weight    -> vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight
    vision_tower.blocks.{i}.attn.k_proj.weight        -> vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight
    vision_tower.blocks.{i}.mlp.fc1.weight            -> vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight
    vision_tower.post_layernorm.weight                -> vision_tower.vision_model.post_layernorm.weight
    ------------------------------------------------------------------------------------------
    vision_projector.fc1.*                            -> multi_modal_projector.linear_1.*
    vision_projector.fc2.*                            -> multi_modal_projector.linear_2.*
    ------------------------------------------------------------------------------------------
    vision_resampler.*                                -> (无对应结构) -> mmllm_extra.safetensors
    audio_tower.* / audio_projector.*                 -> (无对应结构) -> mmllm_extra.safetensors

重要提醒 (务必读 README 的 "路径 A / 路径 B 取舍" 一节)
------------------------------------------------------
HF 的 `LlavaForConditionalGeneration` 把视觉 token 数**硬绑定在 CLIP patch 网格上**
(= (image_size // patch_size) ** 2), 而本项目的视觉 token 数由 `pool_mode` 决定:

    * `pool_mode="perceiver"` : 用 Perceiver Resampler 压到 `out_tokens`(默认 256) -> HF 无对应结构
    * `pool_mode="mlp"`       : 投影器里写死了 PixelShuffle, token 数 = 网格 / 4 -> HF 无对应结构

也就是说, **当前的 `modeling.py` 没有一种配置能被 HF Llava 1:1 表达**。本脚本因此不会
静默导出"看起来成功、实际数值错误"的权重: 它会输出一份 `export_report.json`, 逐条列出
blocker / warning, 并给出可执行的补救动作。

命令行
------
    python src/deploy/export_hf.py --src runs/exp1 --dst out/hf_llava
    python src/deploy/export_hf.py --src runs/exp1 --dst out/hf_llava --shards 2 --verify
    python src/deploy/export_hf.py --src runs/exp1 --dst out/hf_llava --strict   # 有 blocker 就直接失败
    python src/deploy/export_hf.py --src runs/exp1 --dst out/hf_llava --with-remote-code  # 附带本项目 modeling 代码
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------- #
# 允许在无 torch 的环境下做"纯文本"级别的检查 (map_weight_name 不依赖 torch)
# --------------------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"


def load_mm_config_class():
    """拿到 `MMConfig` 类。

    优先 `from mmllm.config import MMConfig`; 若 `mmllm/__init__.py` 因为缺 torch 而导入失败,
    退化成"按文件路径直接加载 config.py", 从而在只装了 numpy 的机器上也能跑本脚本。
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


# ======================================================================================= #
# 1. 命名映射规则 (本项目 <-> HuggingFace)
# ======================================================================================= #

# (正则, 替换模板); 顺序敏感, 先匹配到的先生效
_LLM_RULES: List[Tuple[str, str]] = [
    (r"^language_model\.embed_tokens\.(.+)$", r"language_model.model.embed_tokens.\1"),
    (r"^language_model\.layers\.(\d+)\.input_norm\.(.+)$",
     r"language_model.model.layers.\1.input_layernorm.\2"),
    (r"^language_model\.layers\.(\d+)\.post_attn_norm\.(.+)$",
     r"language_model.model.layers.\1.post_attention_layernorm.\2"),
    (r"^language_model\.layers\.(\d+)\.attn\.(.+)$",
     r"language_model.model.layers.\1.self_attn.\2"),
    (r"^language_model\.layers\.(\d+)\.mlp\.(.+)$",
     r"language_model.model.layers.\1.mlp.\2"),
    (r"^language_model\.final_norm\.(.+)$", r"language_model.model.norm.\1"),
    (r"^language_model\.lm_head\.(.+)$", r"language_model.lm_head.\1"),
]

_VISION_RULES: List[Tuple[str, str]] = [
    (r"^vision_tower\.patch_embed\.(.+)$",
     r"vision_tower.vision_model.embeddings.patch_embedding.\1"),
    (r"^vision_tower\.pos_embed$",
     r"vision_tower.vision_model.embeddings.position_embedding.weight"),
    (r"^vision_tower\.pre_layrnorm\.(.+)$", r"vision_tower.vision_model.pre_layrnorm.\1"),
    (r"^vision_tower\.blocks\.(\d+)\.input_norm\.(.+)$",
     r"vision_tower.vision_model.encoder.layers.\1.layer_norm1.\2"),
    (r"^vision_tower\.blocks\.(\d+)\.post_attn_norm\.(.+)$",
     r"vision_tower.vision_model.encoder.layers.\1.layer_norm2.\2"),
    (r"^vision_tower\.blocks\.(\d+)\.attn\.(.+)$",
     r"vision_tower.vision_model.encoder.layers.\1.self_attn.\2"),
    (r"^vision_tower\.blocks\.(\d+)\.mlp\.(.+)$",
     r"vision_tower.vision_model.encoder.layers.\1.mlp.\2"),
    (r"^vision_tower\.post_layernorm\.(.+)$", r"vision_tower.vision_model.post_layernorm.\1"),
]

# HF 侧允许存在的 bias, 但本项目的 Attention 是 bias=False, 导出时补零
_BIAS_SUFFIXES = (".bias",)


def _compile(rules: Sequence[Tuple[str, str]]) -> List[Tuple[re.Pattern, str]]:
    return [(re.compile(pat), rep) for pat, rep in rules]


_LLM_COMPILED = _compile(_LLM_RULES)
_VISION_COMPILED = _compile(_VISION_RULES)


def map_weight_name(src_key: str) -> Optional[str]:
    """本项目权重名 -> HF 权重名; 无法映射时返回 None。

    只做"名字"层面的映射; 形状层面的差异 (CLIP 的 CLS token / 缺失的 bias) 由
    `export_state_dict()` 里的补丁函数处理。
    """
    for pat, rep in _LLM_COMPILED:
        m = pat.match(src_key)
        if m:
            return pat.sub(rep, src_key)
    for pat, rep in _VISION_COMPILED:
        m = pat.match(src_key)
        if m:
            return pat.sub(rep, src_key)
    return None


def map_hf_to_project(hf_key: str) -> Optional[str]:
    """HF 权重名 -> 本项目权重名 (反向映射, 与 `map_weight_name` 对称)。

    用于把 HF 官方权重 (例如 openai/clip-vit-large-patch14-336) 灌回本项目,
    或者在训练脚本里做双向核对。
    """
    if hf_key.startswith("language_model.model."):
        rest = hf_key[len("language_model.model."):]
        rest = rest.replace("input_layernorm.", "input_norm.")
        rest = rest.replace("post_attention_layernorm.", "post_attn_norm.")
        rest = rest.replace("self_attn.", "attn.")
        # language_model.model.norm.* -> language_model.final_norm.*
        if rest.startswith("norm."):
            rest = "final_norm." + rest[len("norm."):]
        return f"language_model.{rest}"
    if hf_key.startswith("language_model.lm_head."):
        return hf_key
    if hf_key.startswith("vision_tower.vision_model."):
        rest = hf_key[len("vision_tower.vision_model."):]
        rest = rest.replace("embeddings.patch_embedding.", "patch_embed.")
        rest = rest.replace("embeddings.position_embedding.weight", "pos_embed")
        rest = rest.replace("pre_layrnorm.", "pre_layrnorm.")
        rest = rest.replace("encoder.layers.", "blocks.")
        rest = rest.replace(".layer_norm1.", ".input_norm.")
        rest = rest.replace(".layer_norm2.", ".post_attn_norm.")
        rest = rest.replace(".self_attn.", ".attn.")
        return f"vision_tower.{rest}"
    return None


# ======================================================================================= #
# 2. 路径 A 就绪性体检
# ======================================================================================= #

@dataclass
class Issue:
    """一条体检结论。"""
    code: str
    severity: str          # "blocker" | "warning" | "info"
    title: str
    detail: str
    remedy: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "remedy": self.remedy,
        }


def check_path_a_readiness(cfg) -> List[Issue]:
    """检查当前 checkpoint 能否被 HF `LlavaForConditionalGeneration` 1:1 表达。

    返回 Issue 列表 (severity=blocker 的会让 `--strict` 直接失败)。
    """
    issues: List[Issue] = []
    vis = cfg.vision

    # R1: Perceiver Resampler
    if vis.pool_mode == "perceiver":
        issues.append(Issue(
            "R1", "blocker", "Perceiver Resampler 无 HF 对应结构",
            "HF Llava 的 vision_tower -> multi_modal_projector 之间没有可学习 latent 查询模块, "
            "无法承载 vision_resampler.*。",
            "要么改用路径 B (自定义 vLLM 模型) 保持数值一致; "
            "要么用 pool_mode='mlp' 重训一版 CLIP 网格对齐的视觉塔。",
        ))

    # R2: PixelShuffle
    if vis.pool_mode != "perceiver":
        issues.append(Issue(
            "R2", "blocker", "投影器内写死 PixelShuffle, token 数与 HF 不一致",
            "modeling.py 在 pool_mode != 'perceiver' 时固定 pixel_shuffle=True, "
            "视觉 token 数 = patch 网格 / 4; 而 HF Llava 恒等于 patch 网格。",
            "路径 A 需要一个 'CLIP 网格对齐' 的视觉塔 (grid == out_tokens, 无 pixel shuffle); "
            "当前 modeling.py 无法表达该配置, 请走路径 B 或改造 modeling.py。",
        ))

    # R3: projector_hidden_mult
    if cfg.projector_hidden_mult != 1:
        issues.append(Issue(
            "R3", "blocker", "投影器隐藏维度倍数与 HF 不一致",
            f"projector_hidden_mult={cfg.projector_hidden_mult}; "
            "HF 的 multi_modal_projector 是固定的 2 层 MLP (vision_hidden -> text_hidden -> text_hidden)。",
            "把 projector_hidden_mult 设为 1 后重训/重导。",
        ))

    # R4: 投影器输出 LayerNorm
    if cfg.projector_norm_out:
        issues.append(Issue(
            "R4", "warning", "投影器末端 LayerNorm 无 HF 对应结构",
            "vision_projector.norm 是对投影输出做的 LayerNorm, HF Llava 的 multi_modal_projector 没有这一步。",
            "训练时设 projector_norm_out=False, 或接受导出后视觉特征的尺度差异。",
        ))

    # R5: CLIP CLS token
    grid = (vis.image_size // vis.patch_size) ** 2
    issues.append(Issue(
        "R5", "warning", "CLIP 多一个 CLS token, 位置编码形状不同",
        f"HF CLIP 的 position_embedding 形状为 ({grid + 1}, {vis.hidden_size}) (含 CLS), "
        f"本项目为 (1, {grid}, {vis.hidden_size})。导出时会补一行零 CLS。",
        "HF 侧用 image_feature_select_strategy='default' 丢掉 CLS 输出, 但 CLS 仍参与第 0 层注意力, "
        "会造成极小的数值差异, 上线前请做逐层对比。",
    ))

    # R6: Attention bias
    issues.append(Issue(
        "R6", "info", "注意力 bias 缺失需补零",
        "本项目 Attention 为 bias=False, HF CLIP 的 q/k/v/out_proj 带 bias。",
        "导出时自动补零 bias, 以保证与本项目前向完全一致。",
    ))

    # R7: 音频塔
    issues.append(Issue(
        "R7", "blocker", "音频塔无 HF Llava 对应结构",
        "HF Llava 只支持图像; audio_tower.* / audio_projector.* 无处安放。",
        "权重写入 mmllm_extra.safetensors; 音频请走路径 B 或本项目的 serve.py "
        "(transformers 后端可完整支持音频)。",
    ))

    # R8: 词表与特殊 token
    max_special = max(cfg.image_token_id, cfg.audio_token_id, cfg.video_token_id)
    if max_special >= cfg.llm.vocab_size:
        issues.append(Issue(
            "R8", "blocker", "特殊 token id 超出词表范围",
            f"max(special_token_id)={max_special} >= vocab_size={cfg.llm.vocab_size}, "
            "embedding 表放不下这些 id。",
            "训练时把 llm.vocab_size 设为 >= 32004 (或更大), 并同步扩展 tokenizer。",
        ))

    return issues


def hf_image_token_count(cfg) -> int:
    """HF Llava 侧一张图会产生多少个 image token (CLIP 网格, 丢掉 CLS)。"""
    vis = cfg.vision
    return (vis.image_size // vis.patch_size) ** 2


def project_image_token_count(cfg) -> int:
    """本项目侧一张图会产生多少个 image token。"""
    vis = cfg.vision
    if vis.pool_mode == "perceiver":
        return vis.out_tokens
    # modeling.py: 非 perceiver 模式强制 pixel_shuffle=True -> 网格 / 4
    return max(((vis.image_size // vis.patch_size) ** 2) // 4, 1)


# ======================================================================================= #
# 3. 权重转换
# ======================================================================================= #

def _infer_bias_shape(weight):
    """由同名 weight 推出 bias 的形状。"""
    shape = tuple(weight.shape)
    if len(shape) == 2:      # nn.Linear
        return (shape[0],)
    if len(shape) == 4:      # nn.Conv2d
        return (shape[0],)
    return (shape[0],)


def _fix_position_embedding(dst_key: str, tensor):
    """把 (1, grid, hidden) 的位置编码补成 HF 期望的 (grid+1, hidden)。"""
    if dst_key.endswith("embeddings.position_embedding.weight"):
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.dim() == 2:
            import torch

            cls_row = torch.zeros((1, tensor.shape[-1]), dtype=tensor.dtype)
            tensor = torch.cat([cls_row, tensor], dim=0)
    return tensor


def export_state_dict(state: dict, cfg, force: bool = False):
    """把本项目 state_dict 拆成 (hf_state, extra_state, issues)。

    hf_state : 能 1:1 放进 HF Llava 的张量
    extra_state: HF 没有对应结构的张量 (写进 mmllm_extra.safetensors)
    """
    import torch

    hf_state: Dict[str, object] = {}
    extra_state: Dict[str, object] = {}
    issues: List[Issue] = []
    unmapped: List[str] = []

    projector_compatible = (
        cfg.vision.pool_mode != "perceiver"
        and cfg.projector_hidden_mult == 1
    )

    for key, tensor in state.items():
        # ---- 投影器 ---- #
        if key.startswith("vision_projector."):
            if projector_compatible:
                sub = key[len("vision_projector."):]
                if sub.startswith("fc1."):
                    dst = f"multi_modal_projector.linear_1.{sub[len('fc1.'):]}"
                elif sub.startswith("fc2."):
                    dst = f"multi_modal_projector.linear_2.{sub[len('fc2.'):]}"
                elif sub.startswith("norm."):
                    # 末端 LayerNorm: HF 无对应结构, 放进 extra 并提示
                    extra_state[f"vision_projector.{sub}"] = tensor
                    issues.append(Issue(
                        "R4", "warning", "投影器末端 LayerNorm 未写入 HF 权重",
                        "vision_projector.norm.* 在 HF multi_modal_projector 中没有对应参数, "
                        "已转移到 mmllm_extra.safetensors。",
                        "训练时设 projector_norm_out=False 可消除该差异。",
                    ))
                    continue
                else:
                    unmapped.append(key)
                    continue
                hf_state[dst] = tensor
                continue
            # projector 不兼容: 权重一律转入 extra, 保证导出不丢张量
            extra_state[key] = tensor
            if force:
                issues.append(Issue(
                    "R3", "blocker", "投影器结构与 HF 不兼容, 已转入 extra",
                    f"{key} 无法映射到 multi_modal_projector.* (pool_mode={cfg.vision.pool_mode}, "
                    f"projector_hidden_mult={cfg.projector_hidden_mult})。",
                    "见 README 的 '路径 A 就绪配置' 一节。",
                ))
            else:
                issues.append(Issue(
                    "R3", "blocker", "投影器结构与 HF 不兼容",
                    f"{key} 无法映射到 multi_modal_projector.*。",
                    "已转入 mmllm_extra.safetensors; 要得到可用的 HF 权重请见 README 的 "
                    "'路径 A 就绪配置' 一节, 或改用路径 B。",
                ))
            unmapped.append(key)
            continue

        # ---- Perceiver / 音频塔: HF 没有对应结构 ---- #
        if key.startswith("vision_resampler.") or key.startswith("audio_tower.") \
                or key.startswith("audio_projector."):
            extra_state[key] = tensor
            continue

        # ---- 常规映射 ---- #
        dst = map_weight_name(key)
        if dst is None:
            unmapped.append(key)
            extra_state[key] = tensor
            continue

        tensor = _fix_position_embedding(dst, tensor)
        hf_state[dst] = tensor

    # ---- 补齐 HF 期望但本项目没有的 bias (CLIP 的 attn bias) ---- #
    added_bias: List[str] = []
    for dst in list(hf_state.keys()):
        if not dst.endswith(".bias"):
            continue
        weight_key = dst[: -len(".bias")] + ".weight"
        if weight_key in hf_state:
            continue
        w = hf_state[weight_key]
        hf_state[dst] = torch.zeros(_infer_bias_shape(w), dtype=w.dtype)
        added_bias.append(dst)

    if added_bias:
        issues.append(Issue(
            "R6", "info", f"已补齐 {len(added_bias)} 个零 bias",
            ", ".join(added_bias[:4]) + (" ..." if len(added_bias) > 4 else ""),
            "本项目 Attention 无 bias, 补零保证前向与本项目完全一致。",
        ))

    if unmapped:
        issues.append(Issue(
            "R9", "warning", f"{len(unmapped)} 个张量无法映射",
            ", ".join(unmapped[:4]) + (" ..." if len(unmapped) > 4 else ""),
            "这些张量已保存在 mmllm_extra.safetensors 中。",
        ))

    return hf_state, extra_state, issues


# ======================================================================================= #
# 4. config.json (LlavaConfig 格式)
# ======================================================================================= #

def build_hf_config(cfg, issues: List[Issue], token_count: int) -> dict:
    """构造 HF `LlavaConfig` 字典。"""
    llm, vis = cfg.llm, cfg.vision

    vision_config = {
        "model_type": "clip_vision_model",
        "hidden_size": vis.hidden_size,
        "intermediate_size": vis.hidden_size * 4,
        "num_hidden_layers": vis.num_layers,
        "num_attention_heads": vis.num_heads,
        "num_channels": 3,
        "image_size": vis.image_size,
        "patch_size": vis.patch_size,
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "hidden_act": "quick_gelu",
        "projection_dim": llm.hidden_size,
        "initializer_range": 0.02,
    }

    text_config = {
        "model_type": "llama",
        "hidden_size": llm.hidden_size,
        "intermediate_size": llm.intermediate_size,
        "num_hidden_layers": llm.num_layers,
        "num_attention_heads": llm.num_heads,
        "num_key_value_heads": llm.num_kv_heads,
        "head_dim": llm.head_dim,
        "max_position_embeddings": llm.max_position_embeddings,
        "rms_norm_eps": llm.rms_norm_eps,
        "vocab_size": llm.vocab_size,
        "rope_theta": llm.rope_theta,
        "tie_word_embeddings": llm.tie_word_embeddings,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": llm.dtype,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }

    return {
        "architectures": ["LlavaForConditionalGeneration"],
        "model_type": "llava",
        "image_token_index": cfg.image_token_id,
        "image_seq_length": token_count,
        "projector_hidden_act": "gelu",
        "vision_feature_select_strategy": "default",
        "vision_config": vision_config,
        "text_config": text_config,
        # ---- 本项目私有信息 (PretrainedConfig 会原样保留) ---- #
        "_mmllm_export": {
            "source": "mmllm",
            "project_image_tokens": project_image_token_count(cfg),
            "hf_image_tokens": token_count,
            "audio_token_id": cfg.audio_token_id,
            "video_token_id": cfg.video_token_id,
            "audio_out_tokens_per_chunk": cfg.audio.out_tokens_per_chunk,
            "notes": [i.to_dict() for i in issues],
        },
    }


def build_processor_config(cfg, token_count: int) -> Tuple[dict, dict]:
    """构造 preprocessor_config.json / processor_config.json。"""
    vis = cfg.vision
    pre = {
        "image_processor_type": "CLIPImageProcessor",
        "processor_class": "LlavaProcessor",
        "do_resize": True,
        "size": {"shortest_edge": vis.image_size},
        "do_center_crop": True,
        "crop_size": {"height": vis.image_size, "width": vis.image_size},
        "do_rescale": True,
        "rescale_factor": 0.00392156862745098,
        "do_normalize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "do_convert_rgb": True,
        "resample": 3,
    }
    proc = {
        "processor_class": "LlavaProcessor",
        "image_seq_length": token_count,
        "patch_size": vis.patch_size,
        "vision_feature_select_strategy": "default",
        "num_additional_image_tokens": 0,
    }
    return pre, proc


# ======================================================================================= #
# 5. 分片写出
# ======================================================================================= #

_SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024 ** 2,
    "GB": 1024 ** 3,
    "TB": 1024 ** 4,
}


def parse_size(text: str) -> int:
    """'5GB' / '500MB' / '1024' -> 字节数。"""
    s = str(text).strip().upper()
    for unit in ("TB", "GB", "MB", "KB", "B"):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * _SIZE_UNITS[unit])
    return int(s)


def split_shards(state: Dict[str, object], max_bytes: int, force_shards: Optional[int] = None
                 ) -> List[Tuple[str, Dict[str, object]]]:
    """把 state 切成若干分片, 返回 [(文件名, {name: tensor})]。"""
    import torch

    items = list(state.items())
    if force_shards and force_shards > 0:
        n = min(force_shards, max(len(items), 1))
        per = (len(items) + n - 1) // n
        chunks = [items[i * per:(i + 1) * per] for i in range(n)]
        chunks = [c for c in chunks if c]
    else:
        chunks, cur, cur_size = [], [], 0
        for k, v in items:
            nbytes = v.numel() * v.element_size()
            if cur and cur_size + nbytes > max_bytes:
                chunks.append(cur)
                cur, cur_size = [], 0
            cur.append((k, v))
            cur_size += nbytes
        if cur:
            chunks.append(cur)

    total = len(chunks)
    out = []
    for i, chunk in enumerate(chunks, start=1):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        out.append((name, dict(chunk)))
    return out


def write_shards(state: Dict[str, object], dst: Path, max_bytes: int,
                 force_shards: Optional[int] = None) -> Tuple[List[str], Dict[str, str], int]:
    """写分片 + 生成 index.json; 返回 (分片文件名, weight_map, 总字节数)。"""
    from safetensors.torch import save_file

    shards = split_shards(state, max_bytes, force_shards)
    weight_map: Dict[str, str] = {}
    total_size = 0
    names: List[str] = []
    for name, tensors in shards:
        path = dst / name
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path),
                  metadata={"format": "pt"})
        for k, v in tensors.items():
            weight_map[k] = name
            total_size += v.numel() * v.element_size()
        names.append(name)
        print(f"  [shard] {name}  {len(tensors)} tensors  "
              f"{path.stat().st_size / 1024 ** 2:.1f} MiB")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(dst / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"  [index] model.safetensors.index.json  ({len(weight_map)} tensors, "
          f"total_size={total_size / 1024 ** 2:.1f} MiB)")
    return names, weight_map, total_size


# ======================================================================================= #
# 6. tokenizer 处理
# ======================================================================================= #

_TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "vocab.txt", "spiece.model", "tokenizer.model",
]


def copy_tokenizer(src_tokenizer: Optional[str], dst: Path, cfg, extend_special: bool = True
                   ) -> Dict[str, object]:
    """把 tokenizer 拷进导出目录, 并尽量让 `<image>` 的 id 等于 cfg.image_token_id。"""
    info: Dict[str, object] = {"requested": src_tokenizer, "copied": [], "extended": False}
    if not src_tokenizer:
        info["note"] = "未提供 --tokenizer; 导出目录缺少 tokenizer, vLLM/HF 侧需自行补齐"
        return info

    src_path = Path(src_tokenizer)
    if not src_path.exists():
        info["error"] = f"tokenizer 路径不存在: {src_tokenizer}"
        return info

    if src_path.is_file():       # 单文件 (例如 tokenizer.json)
        shutil.copy2(src_path, dst / src_path.name)
        info["copied"].append(src_path.name)
        return info

    for name in _TOKENIZER_FILES:
        p = src_path / name
        if p.is_file():
            shutil.copy2(p, dst / name)
            info["copied"].append(name)

    # ---- 扩展特殊 token ---- #
    if extend_special:
        try:
            from transformers import AutoTokenizer

            # local_files_only: 避免在无外网的机器上重试 hub 导致长时间阻塞
            tok = AutoTokenizer.from_pretrained(str(src_path), local_files_only=True)
            need = max(cfg.image_token_id, cfg.audio_token_id, cfg.video_token_id)
            base = len(tok)
            added = []
            if base > need:
                info["warning"] = (f"tokenizer 已有 {base} 个 token, 无法把 <image> 放到 id={need}; "
                                   f"请把 cfg.image_token_id 改成 >= {base} (推荐 {base})")
            else:
                filler = tok.add_tokens([f"<extra_{i}>" for i in range(need - base)])
                added += [f"<extra_{i}>" for i in range(filler)]
                n = tok.add_tokens(["<image>", "<audio>", "<video>"])
                added += ["<image>", "<audio>", "<video>"][:n]
                tok.save_pretrained(str(dst))
                info["extended"] = True
                info["added_tokens"] = added
                info["image_token_id"] = tok.convert_tokens_to_ids("<image>")
                info["audio_token_id"] = tok.convert_tokens_to_ids("<audio>")
        except Exception as e:  # transformers 缺失或 tokenizer 不支持扩展
            info["warning"] = f"特殊 token 扩展失败: {e}"
    return info


# ======================================================================================= #
# 7. verify
# ======================================================================================= #

def verify_export(dst: Path, sample: int = 12) -> bool:
    """加载导出目录, 打印张量名 / shape 抽样, 并校验 index 覆盖完整性。"""
    from safetensors import safe_open

    print(f"\n[verify] 目录: {dst}")
    index_path = dst / "model.safetensors.index.json"
    if not index_path.is_file():
        print("[verify] 缺少 model.safetensors.index.json -> FAIL")
        return False
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index.get("weight_map", {})
    print(f"[verify] index 内张量数: {len(weight_map)}")

    # 1) 抽样子集
    shard_names = sorted(set(weight_map.values()))
    shown = 0
    all_keys: List[str] = []
    for name in shard_names:
        path = dst / name
        if not path.is_file():
            print(f"[verify] 分片缺失: {name} -> FAIL")
            return False
        with safe_open(str(path), framework="pt") as sf:
            keys = list(sf.keys())
            all_keys.extend(keys)
            for k in keys:
                if shown < sample:
                    shape = sf.get_slice(k).get_shape()
                    dtype = sf.get_slice(k).get_dtype()
                    print(f"    - {k}  {tuple(shape)}  {dtype}")
                    shown += 1
    print(f"[verify] 分片数: {len(shard_names)}, 实际张量数: {len(all_keys)}")

    # 2) index 覆盖检查
    missing = sorted(set(all_keys) - set(weight_map))
    extra = sorted(set(weight_map) - set(all_keys))
    if missing:
        print(f"[verify] index 缺少 {len(missing)} 个张量, 例如 {missing[:3]} -> FAIL")
        return False
    if extra:
        print(f"[verify] index 多出 {len(extra)} 个张量, 例如 {extra[:3]} -> FAIL")
        return False

    # 3) 反向映射覆盖率
    back_ok = 0
    back_bad: List[str] = []
    for k in all_keys:
        proj = map_hf_to_project(k)
        if proj is not None:
            back_ok += 1
    print(f"[verify] 反向映射覆盖率: {back_ok}/{len(all_keys)} "
          f"(仅统计可映射张量; multi_modal_projector / 音频塔 无对应项)")

    # 4) extra 文件
    extra_path = dst / "mmllm_extra.safetensors"
    if extra_path.is_file():
        with safe_open(str(extra_path), framework="pt") as sf:
            extra_keys = list(sf.keys())
        print(f"[verify] mmllm_extra.safetensors: {len(extra_keys)} 个张量 "
              f"(例如 {extra_keys[:3]})")

    print("[verify] 结果: PASS")
    return True


# ======================================================================================= #
# 8. 主流程
# ======================================================================================= #

def load_source(src: Path):
    """读取本项目 checkpoint: (cfg, state_dict)。"""
    cfg_path = src / "config.json"
    weights = src / "model.safetensors"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"缺少 {cfg_path}")
    if not weights.is_file():
        raise FileNotFoundError(f"缺少 {weights} (请先用 model.save_pretrained() 保存)")

    MMConfig = load_mm_config_class()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = MMConfig.from_dict(json.load(f))

    from safetensors.torch import load_file

    state = load_file(str(weights))
    return cfg, state


def copy_remote_code(dst: Path) -> Dict[str, object]:
    """把本项目的 modeling/config 拷进导出目录, 供 `trust_remote_code=True` 使用。

    这是"路径 A 的第三条路": HF 目录保持标准结构, 但模型代码来自本仓库,
    从而 100% 保持数值一致 (代价: 不能直接被 vLLM 的 Llava 加载器识别)。
    """
    info: Dict[str, object] = {"copied": []}
    modeling_src = _SRC_DIR / "mmllm" / "modeling.py"
    config_src = _SRC_DIR / "mmllm" / "config.py"
    if not modeling_src.is_file() or not config_src.is_file():
        info["error"] = "找不到 src/mmllm/modeling.py 或 config.py"
        return info

    modeling_txt = modeling_src.read_text(encoding="utf-8")
    # 相对导入 -> 扁平导入 (HF remote code 目录是平的)
    modeling_txt = modeling_txt.replace("from .config import MMConfig",
                                        "from configuration_mmllm import MMConfig")
    (dst / "modeling_mmllm.py").write_text(modeling_txt, encoding="utf-8")
    info["copied"].append("modeling_mmllm.py")
    shutil.copy2(config_src, dst / "configuration_mmllm.py")
    info["copied"].append("configuration_mmllm.py")
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="把 mmllm 权重导出为 HF LlavaForConditionalGeneration 目录",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", required=True, help="本项目 checkpoint 目录 (含 config.json + model.safetensors)")
    ap.add_argument("--dst", required=True, help="导出目录")
    ap.add_argument("--tokenizer", default=None, help="tokenizer 目录或 hub id (如 meta-llama/Llama-2-7b-hf)")
    ap.add_argument("--max-shard-size", default="5GB", help="单个分片最大字节数 (如 5GB / 500MB)")
    ap.add_argument("--shards", type=int, default=0, help="强制分片数 (0=按大小自动)")
    ap.add_argument("--dtype", default=None, help="转换后的 dtype (float16/bfloat16/float32), 默认保持原样")
    ap.add_argument("--force", action="store_true",
                    help="显式确认: 即使存在 blocker 也导出 (权重会转入 mmllm_extra.safetensors)")
    ap.add_argument("--strict", action="store_true", help="存在 blocker 时直接失败退出 (CI 用)")
    ap.add_argument("--verify", action="store_true", help="导出后打印张量名/shape 抽样并校验 index")
    ap.add_argument("--with-remote-code", action="store_true",
                    help="额外拷贝本项目 modeling/config, 支持 trust_remote_code=True")
    args = ap.parse_args(argv)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[export] 源目录: {src}")
    print(f"[export] 目标目录: {dst}")

    cfg, state = load_source(src)
    print(f"[export] 权重张量数: {len(state)}")

    # ---- 1. 就绪性体检 ---- #
    issues = check_path_a_readiness(cfg)
    blockers = [i for i in issues if i.severity == "blocker"]
    token_count = hf_image_token_count(cfg)
    print(f"\n[export] 视觉 token 数: 本项目={project_image_token_count(cfg)}, "
          f"HF Llava={token_count} (CLIP 网格)")
    print(f"[export] 路径 A 体检: {len(blockers)} blocker / "
          f"{len([i for i in issues if i.severity == 'warning'])} warning")
    for i in issues:
        mark = {"blocker": "[BLOCK]", "warning": "[WARN]", "info": "[INFO]"}[i.severity]
        print(f"  {mark} {i.code} {i.title}")
        print(f"         详情: {i.detail}")
        print(f"         建议: {i.remedy}")

    if blockers and args.strict:
        print("\n[export] --strict 生效, 存在 blocker, 终止导出。")
        return 2

    # ---- 2. 权重转换 ---- #
    hf_state, extra_state, conv_issues = export_state_dict(state, cfg, force=args.force)
    issues.extend(conv_issues)

    if args.dtype:
        import torch

        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[args.dtype]
        hf_state = {k: v.to(dt) for k, v in hf_state.items()}
        extra_state = {k: v.to(dt) for k, v in extra_state.items()}
        print(f"[export] dtype -> {args.dtype}")

    # ---- 3. 写 config ---- #
    hf_cfg = build_hf_config(cfg, issues, token_count)
    with open(dst / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f, indent=2, ensure_ascii=False)
    print(f"[export] 已写 config.json (LlavaConfig, image_token_index={cfg.image_token_id}, "
          f"image_seq_length={token_count})")

    pre, proc = build_processor_config(cfg, token_count)
    with open(dst / "preprocessor_config.json", "w", encoding="utf-8") as f:
        json.dump(pre, f, indent=2, ensure_ascii=False)
    with open(dst / "processor_config.json", "w", encoding="utf-8") as f:
        json.dump(proc, f, indent=2, ensure_ascii=False)

    gen_cfg = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
    }
    with open(dst / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2, ensure_ascii=False)

    if args.with_remote_code:
        info = copy_remote_code(dst)
        hf_cfg["auto_map"] = {
            "AutoConfig": "configuration_mmllm.MMConfig",
            "AutoModel": "modeling_mmllm.MultimodalLM",
            "AutoModelForCausalLM": "modeling_mmllm.MultimodalLM",
        }
        with open(dst / "config.json", "w", encoding="utf-8") as f:
            json.dump(hf_cfg, f, indent=2, ensure_ascii=False)
        print(f"[export] remote code: {info}")

    # ---- 4. 写权重 (分片) ---- #
    print("\n[export] 写权重:")
    shard_names, weight_map, total_size = write_shards(
        hf_state, dst, parse_size(args.max_shard_size), args.shards or None)

    if extra_state:
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in extra_state.items()},
                  str(dst / "mmllm_extra.safetensors"), metadata={"format": "pt"})
        with open(dst / "mmllm_extra.json", "w", encoding="utf-8") as f:
            json.dump({"tensors": list(extra_state.keys()),
                       "note": "HF Llava 无对应结构的权重; 供路径 B / 本项目 serve.py 使用"},
                      f, indent=2, ensure_ascii=False)
        print(f"  [extra] mmllm_extra.safetensors  {len(extra_state)} tensors")

    # ---- 5. tokenizer ---- #
    tok_info = copy_tokenizer(args.tokenizer or cfg.llm.tokenizer, dst, cfg)
    print(f"[export] tokenizer: {tok_info}")

    # ---- 6. 报告 ---- #
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(src),
        "dst": str(dst),
        "path_a_ready": len(blockers) == 0,
        "forced": bool(args.force and blockers),
        "image_tokens": {
            "project": project_image_token_count(cfg),
            "hf_llava": token_count,
            "match": project_image_token_count(cfg) == token_count,
        },
        "shards": shard_names,
        "total_size_bytes": total_size,
        "tensor_count": len(weight_map),
        "extra_tensor_count": len(extra_state),
        "tokenizer": tok_info,
        "issues": [i.to_dict() for i in issues],
    }
    with open(dst / "export_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[export] 完成: {len(weight_map)} 个张量 / {len(shard_names)} 个分片 / "
          f"{total_size / 1024 ** 2:.1f} MiB")
    if report["path_a_ready"]:
        print("[export] 路径 A 就绪: 是")
    else:
        print(f"[export] 路径 A 就绪: 否 -- 该 checkpoint 有 {len(blockers)} 个 blocker, "
              f"详见 export_report.json")
        print("[export]   -> 路径 B (自定义 vLLM 模型) 可保持数值完全一致; 见 README。")

    if args.verify:
        ok = verify_export(dst)
        if not ok:
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
