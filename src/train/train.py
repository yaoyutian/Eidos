"""
多模态大模型训练入口 (Stage1 对齐 / Stage2 指令微调 / Stage3 全模态 SFT)
======================================================================
本文件是"训练侧"的唯一入口, 不含任何网络结构实现 —— 结构一律来自
``src/mmllm/modeling.py`` 的 :class:`MultimodalLM`。

能力清单
--------
1. YAML 配置 (PyYAML) + argparse 覆盖, 不引入新依赖;
2. 参数分组学习率: projector / llm / vision / audio 四组独立 lr 与 weight decay,
   norm 与 bias 参数不做 weight decay;
3. 精度: bf16 autocast (torch.amp), GradScaler 仅在 fp16 时启用, fp32 可跑 CPU 冒烟;
4. 梯度检查点: 对 decoder layer 挂 ``torch.utils.checkpoint`` (use_reentrant=False,
   与 FSDP 兼容), 提供开关;
5. 分布式: ``torchrun`` 兼容, FSDP(auto_wrap 按 TransformerBlock) 或 DDP,
   同时支持单机单卡 (不初始化 process group);
6. 梯度累积 + ``no_sync`` 优化, 梯度裁剪 (max_norm=1.0);
7. 学习率调度: warmup + cosine decay;
8. 断点续训: model / optimizer / scheduler / scaler / step / epoch 全量保存与恢复;
9. 日志: loss / 各组 lr / tokens_per_sec / 显存峰值 (torch.cuda.max_memory_allocated);
10. checkpoint 走 ``model.save_pretrained(out_dir)``, 同时保存 optimizer 状态;
11. 权重加载: 可从 HF Llama/Qwen 初始化 LLM 主干
    (``convert_hf_llm_state_dict`` + ``language_model.`` 前缀补齐),
    ``--base-model`` 为空时即为"从零预训练"模式。

用法
----
    python -m src.train.train --config src/train/configs/stage1_align.yaml
    torchrun --nproc_per_node 8 -m src.train.train \
        --config src/train/configs/stage2_instruct.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# --------------------------------------------------------------------------- #
# 路径引导: 让 `python src/train/train.py` 与 `python -m src.train.train` 都能 import mmllm
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve()
_SRC_DIR = _HERE.parents[1]          # .../src
_REPO_ROOT = _SRC_DIR.parent         # 仓库根
for _p in (str(_REPO_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmllm.config import MMConfig, preset                              # noqa: E402
from mmllm.modeling import (                                            # noqa: E402
    MultimodalLM,
    TransformerBlock,
    convert_hf_llm_state_dict,
)

# 四个参数分组 (顺序即打印顺序)
PARAM_GROUPS: Tuple[str, ...] = ("projector", "llm", "vision", "audio")

# 各阶段"应该"解冻什么 —— 与 YAML 不一致时给出告警, 避免误配
STAGE_EXPECT_FREEZE: Dict[str, Dict[str, bool]] = {
    "stage1_align": {"llm": False, "vision": False, "audio": False, "projector": True},
    "stage2_instruct": {"llm": True, "vision": False, "audio": False, "projector": True},
    "stage3_omni_sft": {"llm": True, "vision": False, "audio": True, "projector": True},
}


# =========================================================================== #
# 1. 通用小工具
# =========================================================================== #
def log_info(msg: str, rank: int = 0) -> None:
    """仅 rank0 打印, 避免多卡刷屏。"""
    if rank == 0:
        print(f"[train] {msg}", flush=True)


def warn(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"[train][WARN] {msg}", flush=True)


def get_in(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """按 "a.b.c" 取嵌套配置。"""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_in(d: Dict[str, Any], path: str, value: Any) -> None:
    """按 "a.b.c" 写嵌套配置 (缺失的中间层自动建 dict)。"""
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 (other 覆盖 base)。"""
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")
    return data


def dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def set_seed(seed: int, rank: int = 0) -> None:
    """种子: 各 rank 偏移, 保证 dropout 不同; FSDP/DDP 会在初始化时同步参数。"""
    seed = int(seed) + int(rank)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy 非必需
        pass


def _parse_scalar(text: str) -> Any:
    """把命令行字符串解析成 bool/int/float/str。"""
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def apply_set_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """--set a.b.c=value 形式的覆盖。"""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set 需要 KEY=VALUE 形式, 收到: {item}")
        key, value = item.split("=", 1)
        val = _parse_scalar(value)
        if isinstance(val, str):
            # PyYAML 不会把 "1e-3" 解析成 float, 这里兜底
            try:
                val = float(val)
            except ValueError:
                pass
        set_in(cfg, key.strip(), val)
    return cfg


# =========================================================================== #
# 2. 分布式
# =========================================================================== #
@dataclass
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    enabled: bool = False
    backend: str = "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_dist(cfg: Dict[str, Any]) -> DistInfo:
    """
    单机单卡: 直接返回, **不初始化 process group**。
    torchrun (WORLD_SIZE>1): 初始化 nccl (无 GPU 时退化 gloo)。
    """
    train_cfg = get_in(cfg, "train", {}) or {}
    backend_pref = train_cfg.get("dist_backend", "auto")
    world = int(os.environ.get("WORLD_SIZE", "1"))

    if backend_pref == "none" or world <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DistInfo(device=device, enabled=False, backend="none")

    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    log_info(f"分布式已启动: backend={backend} world_size={world} rank={rank}", rank)
    return DistInfo(rank=rank, world_size=world, local_rank=local_rank,
                    device=device, enabled=True, backend=backend)


def cleanup_dist(dist_info: DistInfo) -> None:
    if not dist_info.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """取回被 DDP/FSDP 包装前的原始模块。"""
    return getattr(model, "module", model)


# =========================================================================== #
# 3. 配置 -> MMConfig
# =========================================================================== #
def _coerce(value: Any, field_type: Any) -> Any:
    """按 dataclass 字段类型做一次温和的类型转换。"""
    if field_type in (int, float, str, bool):
        try:
            if field_type is bool and isinstance(value, str):
                return value.lower() in ("1", "true", "yes", "y")
            return field_type(value)
        except (TypeError, ValueError):
            return value
    return value


def build_model_cfg(cfg: Dict[str, Any]) -> MMConfig:
    """preset + YAML 覆盖 -> MMConfig。"""
    model_cfg = get_in(cfg, "model", {}) or {}
    preset_name = model_cfg.get("preset", "tiny")
    cfg_obj = preset(preset_name)

    for sub, section in (("llm", "llm"), ("vision", "vision"), ("audio", "audio")):
        target = getattr(cfg_obj, sub)
        for key, value in (model_cfg.get(section) or {}).items():
            if not hasattr(target, key):
                warn(f"model.{section}.{key} 不是 {sub.upper()}Config 的字段, 已忽略")
                continue
            ftype = type(getattr(target, key))
            setattr(target, key, _coerce(value, ftype))
    return cfg_obj


def override_from_hf_config(cfg_obj: MMConfig, base_model: str,
                            trust_remote_code: bool = False) -> None:
    """
    用 HF config 的结构字段覆盖本项目 LLMConfig, 保证 HF 权重能对齐 shape。
    只覆盖"结构", 不覆盖训练相关字段。
    """
    from transformers import AutoConfig

    try:
        hf_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    except Exception as e:  # 离线/无该模型时不影响主流程
        warn(f"读取 HF config 失败 ({base_model}): {e}; 跳过结构对齐")
        return

    mapping = {
        "hidden_size": "hidden_size",
        "num_layers": "num_hidden_layers",
        "num_heads": "num_attention_heads",
        "num_kv_heads": "num_key_value_heads",
        "head_dim": "head_dim",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        "rope_theta": "rope_theta",
        "rms_norm_eps": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "tie_word_embeddings": "tie_word_embeddings",
    }
    for hf_key, our_key in mapping.items():
        val = getattr(hf_cfg, hf_key, None)
        if val is None:
            continue
        if our_key == "head_dim" and not hasattr(hf_cfg, "head_dim"):
            val = cfg_obj.llm.hidden_size // cfg_obj.llm.num_heads
        if hasattr(cfg_obj.llm, our_key):
            old = getattr(cfg_obj.llm, our_key)
            if old != val:
                log_info(f"结构对齐: llm.{our_key}: {old} -> {val} (来自 {base_model})")
            setattr(cfg_obj.llm, our_key, val)


def build_tokenizer(cfg_obj: MMConfig, cfg: Dict[str, Any]):
    """加载 tokenizer 并注册 <image>/<audio>/<video> 特殊 token。"""
    from transformers import AutoTokenizer

    from mmllm.tokenizer_ext import add_multimodal_tokens

    model_cfg = get_in(cfg, "model", {}) or {}
    name = model_cfg.get("tokenizer") or cfg_obj.llm.tokenizer
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    added = add_multimodal_tokens(tok, cfg_obj)
    # 词表必须覆盖新增的特殊 token, 否则 embedding 越界
    cfg_obj.llm.vocab_size = max(int(cfg_obj.llm.vocab_size), len(tok))
    log_info(f"tokenizer={name} 词表={len(tok)} (新增特殊 token {added} 个) "
             f"image_id={cfg_obj.image_token_id} audio_id={cfg_obj.audio_token_id}")
    return tok


# =========================================================================== #
# 4. 模型构建 / 权重加载
# =========================================================================== #
def load_hf_llm_weights(model: MultimodalLM, base_model: str,
                        trust_remote_code: bool = False) -> None:
    """
    从 HF Llama/Qwen 灌入 LLM 主干。

    ``convert_hf_llm_state_dict`` 产出的 key 不带 ``language_model.`` 前缀
    (embed_tokens.* / layers.* / final_norm.* / lm_head.*),
    这里自行补前缀后写入 ``model.language_model``。
    词表不一致时按行部分拷贝 (新增的特殊 token 行保持随机初始化)。
    """
    from transformers import AutoModelForCausalLM

    try:
        hf = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.float32, trust_remote_code=trust_remote_code)
    except TypeError:  # 老版本 transformers 只有 torch_dtype
        hf = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)

    hf_sd = hf.state_dict()
    converted = convert_hf_llm_state_dict(hf_sd)
    target = model.language_model.state_dict()

    loaded, skipped = 0, []
    new_state: Dict[str, torch.Tensor] = {}
    for key, val in converted.items():
        full_key = f"language_model.{key}"
        if full_key not in target:
            skipped.append(key)
            continue
        tgt = target[full_key]
        if tgt.shape == val.shape:
            new_state[full_key] = val.to(tgt.dtype)
            loaded += 1
        elif tgt.dim() == 2 and val.dim() == 2 and tgt.shape[1] == val.shape[1] \
                and tgt.shape[0] >= val.shape[0]:
            # 词表扩容场景: 只拷贝重叠的行, 新增行保持随机初始化
            buf = tgt.detach().clone()
            buf[: val.shape[0]] = val.to(tgt.dtype)
            new_state[full_key] = buf
            loaded += 1
        else:
            skipped.append(f"{key} {tuple(val.shape)}!={tuple(tgt.shape)}")

    missing, unexpected = model.language_model.load_state_dict(new_state, strict=False)
    log_info(f"已从 {base_model} 加载 {loaded} 个 LLM 张量; "
             f"跳过 {len(skipped)} 个 (形状/结构不匹配)")
    if skipped:
        warn(f"跳过示例: {skipped[:5]}")
    if missing:
        warn(f"LLM 缺失(保持随机初始化)示例: {list(missing)[:5]}")
    del hf


def build_model(cfg_obj: MMConfig, cfg: Dict[str, Any]) -> MultimodalLM:
    """构造 MultimodalLM; base_model 为空 -> 从零随机初始化。"""
    model_cfg = get_in(cfg, "model", {}) or {}
    base_model = (model_cfg.get("base_model") or "").strip()
    model = MultimodalLM(cfg_obj)
    if base_model:
        log_info(f"复用开源基座: {base_model}")
        load_hf_llm_weights(model, base_model,
                            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)))
    else:
        log_info("从零初始化 (base_model 为空): 全部参数随机初始化")
    return model


def apply_freeze(model: MultimodalLM, cfg: Dict[str, Any]) -> Dict[str, bool]:
    """按 YAML 的 freeze 策略调用 set_trainable。"""
    freeze = get_in(cfg, "train.freeze", {}) or {}
    llm = bool(freeze.get("llm", True))
    vision = bool(freeze.get("vision", False))
    audio = bool(freeze.get("audio", False))
    projector = bool(freeze.get("projector", True))
    model.set_trainable(llm=llm, vision=vision, audio=audio, projector=projector)
    log_info(f"冻结策略: llm={llm} vision={vision} audio={audio} projector={projector}")
    return {"llm": llm, "vision": vision, "audio": audio, "projector": projector}


def check_stage_consistency(cfg: Dict[str, Any], freeze: Dict[str, bool]) -> None:
    stage = cfg.get("stage") or get_in(cfg, "train.stage")
    if not stage or stage not in STAGE_EXPECT_FREEZE:
        return
    expect = STAGE_EXPECT_FREEZE[stage]
    for key, want in expect.items():
        if freeze.get(key) != want:
            warn(f"阶段 {stage} 期望 freeze.{key}={want}, 实际为 {freeze.get(key)}; "
                 f"请确认这是有意为之")


# =========================================================================== #
# 5. LoRA (peft 可选)
# =========================================================================== #
def maybe_apply_lora(model: MultimodalLM, cfg: Dict[str, Any]) -> Tuple[MultimodalLM, Dict[str, Any]]:
    """
    可选 LoRA:
      * peft 可用 -> 只给 LLM 挂 LoRA (projector / 音频塔仍全参训练);
      * peft 不可用 -> **退化路径**: 退回 LLM 全参微调, 并明确告警 + 落盘说明。
    """
    lora_cfg = get_in(cfg, "train.lora", {}) or {}
    info: Dict[str, Any] = {"enabled": False, "fallback": None}
    if not lora_cfg.get("enabled", False):
        return model, info

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        info["fallback"] = "peft 未安装"
        warn(f"lora.enabled=true 但 peft 不可用 ({e}); "
             f"退化路径: 改为 LLM 全参微调 (等价于 stage2/stage3 的默认行为), "
             f"显存与速度按全参估算")
        return model, info

    target_modules = lora_cfg.get("target_modules") or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(target_modules),
        bias="none",
        task_type=None,
    )
    # 只包 LLM; 包在 language_model 上, 保证 MultimodalLM 的 forward 路径不变
    model.language_model = get_peft_model(model.language_model, peft_cfg)
    # peft 会向基座 forward 传 output_attentions 等参数, 本项目 forward 不认 -> 打补丁
    patch_language_model_forward()
    info["enabled"] = True
    n_lora = sum(p.numel() for n, p in model.language_model.named_parameters() if p.requires_grad)
    log_info(f"LoRA 已启用: r={lora_cfg.get('r', 16)} 目标模块={target_modules} "
             f"可训练 LoRA 参数={n_lora / 1e6:.2f}M")
    return model, info


def patch_language_model_forward() -> None:
    """
    让 LanguageModel.forward 容忍 peft 传入的多余关键字
    (output_attentions / output_hidden_states / return_dict / task_ids)。
    只在运行时打补丁, 不修改 src/mmllm 下的文件。
    """
    import mmllm.modeling as modeling

    if getattr(modeling.LanguageModel, "_kwargs_patched", False):
        return
    orig_forward = modeling.LanguageModel.forward

    def tolerant_forward(self, *args, **kwargs):
        for junk in ("output_attentions", "output_hidden_states", "return_dict", "task_ids"):
            kwargs.pop(junk, None)
        return orig_forward(self, *args, **kwargs)

    modeling.LanguageModel.forward = tolerant_forward
    modeling.LanguageModel._kwargs_patched = True


# =========================================================================== #
# 6. 梯度检查点 / 精度
# =========================================================================== #
def enable_gradient_checkpointing(model: MultimodalLM) -> int:
    """
    对 decoder layer 启用 ``torch.utils.checkpoint``。
    使用 use_reentrant=False: 与 FSDP + 非重入 autograd 兼容, 且支持无梯度输入。
    """
    n = 0
    for module in model.modules():
        if not isinstance(module, TransformerBlock):
            continue
        if getattr(module, "_grad_ckpt", False):
            continue
        orig_forward = module.forward

        def ckpt_forward(self, x, attn_mask=None, past_kv=None, use_cache=False,
                         _orig=orig_forward):
            # 推理/带 KV cache 时不做 checkpoint (会破坏 KV 语义)
            if past_kv is not None or use_cache or not torch.is_grad_enabled():
                return _orig(x, attn_mask, past_kv, use_cache)
            out, present = torch.utils.checkpoint.checkpoint(
                _orig, x, attn_mask, None, False, use_reentrant=False
            )
            return out, present

        # 注意: forward 挂在实例上不会自动绑定 self, 必须用 MethodType 绑定
        module.forward = types.MethodType(ckpt_forward, module)
        module._grad_ckpt = True
        n += 1
    log_info(f"梯度检查点已启用, 覆盖 {n} 个 TransformerBlock")
    return n


def make_autocast(precision: str, device_type: str):
    """bf16 / fp16 autocast; fp32 或不支持的组合返回空上下文。"""
    if precision == "fp32":
        return contextlib.nullcontext(), None
    if device_type == "cpu" and precision == "fp16":
        warn("CPU 不支持 fp16 autocast, 自动退化为 bf16")
        precision = "bf16"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    device_type = "cuda" if device_type == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype), dtype


def make_scaler(precision: str, device_type: str):
    """GradScaler 只在 fp16 时启用 (bf16 不需要)。"""
    enabled = (precision == "fp16") and (device_type == "cuda")
    try:
        scaler = torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - 老版本 torch
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)
    return scaler


# =========================================================================== #
# 7. 参数分组 (核心: 四组独立 lr / weight decay)
# =========================================================================== #
def _group_of(name: str) -> str:
    """按参数名判定所属分组 (兼容 peft 包装后的名字)。"""
    low = name.lower()
    if "vision_tower" in low:
        return "vision"
    if "audio_tower" in low:
        return "audio"
    if "projector" in low or "resampler" in low:
        return "projector"
    if "language_model" in low or "base_model.model" in low:
        return "llm"
    return "other"


def _is_no_decay(name: str, keywords: Sequence[str]) -> bool:
    low = name.lower()
    return any(k.lower() in low for k in keywords)


def build_param_groups(model: nn.Module, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    依据 ``model.trainable_parameters()`` 的 id 把参数分到
    projector / llm / vision / audio 四组, 每组独立 lr 与 weight decay;
    norm / bias 类参数一律不做 weight decay。
    """
    raw = unwrap_model(model)
    trainable_ids = {id(p) for p in raw.trainable_parameters()}
    lr_cfg = get_in(cfg, "train.lr", {}) or {}
    wd_cfg = get_in(cfg, "train.weight_decay", {}) or {}
    no_decay_keywords = get_in(cfg, "train.no_decay_keywords",
                               ["norm", "bias", "ln_", "layernorm"]) or []
    default_wd = float(wd_cfg.get("default", 0.0))

    buckets: Dict[Tuple[str, bool], List[nn.Parameter]] = {}
    for name, param in raw.named_parameters():
        if id(param) not in trainable_ids:
            continue
        group = _group_of(name)
        if group not in PARAM_GROUPS:
            warn(f"参数 {name} 无法归类, 归入 other 组 (lr 使用 default)")
            group = "other"
        no_decay = _is_no_decay(name, no_decay_keywords)
        buckets.setdefault((group, no_decay), []).append(param)

    param_groups: List[Dict[str, Any]] = []
    for group in list(PARAM_GROUPS) + ["other"]:
        lr = float(lr_cfg.get(group, lr_cfg.get("default", 0.0)) or 0.0)
        if lr <= 0:
            # lr=0 视为"该组不训练"; 参数已由 set_trainable 冻结时天然不会出现在这里
            continue
        wd = float(wd_cfg.get(group, default_wd) or 0.0)
        for no_decay in (False, True):
            params = buckets.get((group, no_decay))
            if not params:
                continue
            n_elem = sum(p.numel() for p in params)
            param_groups.append({
                "params": params,
                "lr": lr,
                "weight_decay": 0.0 if no_decay else wd,
                "group_name": f"{group}{'_nodecay' if no_decay else ''}",
                "group_kind": group,
                "n_params": n_elem,
            })
            log_info(f"参数组 {group}{'(no_decay)' if no_decay else ''}: "
                     f"{len(params)} 个张量 / {n_elem / 1e6:.2f}M 参数, "
                     f"lr={lr:g} wd={0.0 if no_decay else wd:g}")
    if not param_groups:
        raise RuntimeError("没有任何可训练参数: 请检查 train.freeze / train.lr 配置")
    return param_groups


# =========================================================================== #
# 8. 调度器
# =========================================================================== #
def build_scheduler(optimizer, warmup_steps: int, total_steps: int,
                    min_lr_ratio: float = 0.1):
    """warmup + cosine decay。"""
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================================== #
# 9. 数据
# =========================================================================== #
def load_records(path: str) -> List[Dict[str, Any]]:
    """读取 jsonl / json 数据文件。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("data", data.get("records", []))
        records = list(data)
    else:  # jsonl
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log_info(f"载入数据 {p}: {len(records)} 条")
    return records


def build_loader(records: List[Dict[str, Any]], cfg_obj: MMConfig, tokenizer,
                 cfg: Dict[str, Any], dist_info: DistInfo):
    """构造 DataLoader (分布式下自动切分)。"""
    from mmllm.data import MMCollator, MMDataset

    data_cfg = get_in(cfg, "data", {}) or {}
    train_cfg = get_in(cfg, "train", {}) or {}
    max_text_len = int(train_cfg.get("max_seq_len") or data_cfg.get("max_text_len", 4096))
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    shuffle = bool(data_cfg.get("shuffle", True))

    # MMDataset 的参数在不同版本间可能叫 audio_processor / feature_extractor,
    # 这里按签名动态选择, 避免版本漂移导致崩溃
    sig = inspect.signature(MMDataset.__init__)
    kwargs: Dict[str, Any] = {}
    img_proc = None
    aud_proc = None
    try:
        from mmllm.data import SimpleAudioProcessor, SimpleImageProcessor

        img_proc = SimpleImageProcessor(cfg_obj.vision.image_size)
        aud_proc = SimpleAudioProcessor(sample_rate=cfg_obj.audio.sample_rate,
                                        n_mels=cfg_obj.audio.n_mels,
                                        max_seconds=cfg_obj.audio.max_seconds)
    except Exception as e:  # pragma: no cover - 内置处理器缺失时用 data.py 默认值
        warn(f"内置图像/音频处理器不可用 ({e}), 使用 data.py 默认实现")
    if "image_processor" in sig.parameters and img_proc is not None:
        kwargs["image_processor"] = img_proc
    if "audio_processor" in sig.parameters and aud_proc is not None:
        kwargs["audio_processor"] = aud_proc
    if "feature_extractor" in sig.parameters and aud_proc is not None:
        kwargs["feature_extractor"] = aud_proc
    if "max_text_len" in sig.parameters:
        kwargs["max_text_len"] = max_text_len
    if "max_images" in sig.parameters:
        kwargs["max_images"] = int(data_cfg.get("max_images", 4))
    if "max_audio_chunks" in sig.parameters:
        kwargs["max_audio_chunks"] = int(data_cfg.get("max_audio_chunks", 8))

    dataset = MMDataset(records, cfg_obj, tokenizer, **kwargs)

    coll_sig = inspect.signature(MMCollator.__init__)
    coll_kwargs: Dict[str, Any] = {"pad_token_id": tokenizer.pad_token_id or 0}
    if "label_pad_id" in coll_sig.parameters:
        coll_kwargs["label_pad_id"] = cfg_obj.label_pad_id
    collator = MMCollator(cfg_obj, **coll_kwargs)

    sampler = None
    if dist_info.enabled:
        sampler = DistributedSampler(dataset, shuffle=shuffle,
                                     drop_last=True, seed=int(cfg.get("seed", 42)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, collator


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# =========================================================================== #
# 10. 分布式包装 (FSDP / DDP)
# =========================================================================== #
def wrap_distributed(model: MultimodalLM, dist_info: DistInfo, cfg: Dict[str, Any]) -> nn.Module:
    """按配置选择 FSDP / DDP / 不包装。"""
    if not dist_info.enabled:
        return model
    train_cfg = get_in(cfg, "train", {}) or {}
    backend = train_cfg.get("dist_backend", "auto")
    if backend in ("auto", "none"):
        backend = "fsdp" if torch.cuda.is_available() else "ddp"

    if backend == "fsdp":
        return _wrap_fsdp(model, dist_info, cfg)
    if backend == "ddp":
        return _wrap_ddp(model, dist_info, cfg)
    raise ValueError(f"未知 dist_backend: {backend}")


def _wrap_fsdp(model: MultimodalLM, dist_info: DistInfo, cfg: Dict[str, Any]) -> nn.Module:
    from torch.distributed.fsdp import (
        CPUOffload,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    fsdp_cfg = get_in(cfg, "train.fsdp", {}) or {}
    precision = get_in(cfg, "train.precision", "bf16")

    param_dtype = torch.bfloat16 if precision == "bf16" else (
        torch.float16 if precision == "fp16" else torch.float32)

    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    strategy = strategy_map.get(str(fsdp_cfg.get("sharding_strategy", "full_shard")),
                                ShardingStrategy.FULL_SHARD)

    if torch.cuda.is_available():
        model.to(dist_info.device)

    policy = ModuleWrapPolicy({TransformerBlock})
    log_info(f"FSDP 包装: strategy={strategy} use_orig_params=True "
             f"param_dtype={param_dtype}")
    return FSDP(
        model,
        auto_wrap_policy=policy,
        # use_orig_params=True: 保留原始 Parameter 对象, 才能按 projector/llm 分组设不同 lr
        use_orig_params=True,
        sharding_strategy=strategy,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=param_dtype,
        ),
        cpu_offload=CPUOffload(offload_params=True) if fsdp_cfg.get("cpu_offload") else None,
        device_id=dist_info.device,
        sync_module_states=bool(fsdp_cfg.get("sync_module_states", True)),
        limit_all_gathers=True,
    )


def _wrap_ddp(model: MultimodalLM, dist_info: DistInfo, cfg: Dict[str, Any]) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP

    ddp_cfg = get_in(cfg, "train.ddp", {}) or {}
    find_unused = bool(ddp_cfg.get("find_unused_parameters", True))
    if torch.cuda.is_available():
        model.to(dist_info.device)
    log_info(f"DDP 包装: find_unused_parameters={find_unused}")
    return DDP(
        model,
        device_ids=[dist_info.local_rank] if torch.cuda.is_available() else None,
        output_device=dist_info.local_rank if torch.cuda.is_available() else None,
        broadcast_buffers=True,
        find_unused_parameters=find_unused,
    )


# =========================================================================== #
# 11. checkpoint 保存 / 恢复
# =========================================================================== #
def _save_pretrained_safe(model: MultimodalLM, out_dir: Path) -> None:
    """
    走 ``model.save_pretrained(out_dir)``。
    tie_word_embeddings=True 时 lm_head 与 embed_tokens 共享显存,
    safetensors 会拒绝共享张量 -> 保存前临时解绑, 保存后恢复。
    """
    lm = model.language_model
    tied = getattr(lm, "lm_head", None) is not None and \
        lm.lm_head.weight is lm.embed_tokens.weight
    if tied:
        lm.lm_head.weight = nn.Parameter(lm.lm_head.weight.detach().clone())
    try:
        model.save_pretrained(str(out_dir))
    finally:
        if tied:
            lm.lm_head.weight = lm.embed_tokens.weight


def save_checkpoint(out_dir: Path, model: nn.Module, optimizer, scheduler, scaler,
                    step: int, epoch: int, cfg: Dict[str, Any], dist_info: DistInfo,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """保存权重 + 训练状态 (只有 rank0 真正写盘)。"""
    if not dist_info.is_main:
        # 非主进程不写盘; 同步由调用方 (_save_step) 的 barrier 统一负责
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)

    # FSDP 需要先收集 full state dict 才能得到完整的 safetensors
    fsdp_state_ctx = contextlib.nullcontext()
    if type(model).__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        fsdp_state_ctx = FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )

    with fsdp_state_ctx:
        try:
            _save_pretrained_safe(raw, out_dir)
        except Exception as e:  # pragma: no cover - 兜底路径
            warn(f"save_pretrained 失败 ({e}); 退化为手工保存 safetensors")
            from safetensors.torch import save_file

            state = {k: v.detach().cpu().contiguous() for k, v in raw.state_dict().items()}
            save_file(state, str(out_dir / "model.safetensors"))
            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(raw.cfg.to_dict(), f, indent=2, ensure_ascii=False)

    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        },
    }
    if extra:
        state.update(extra)
    torch.save(state, out_dir / "trainer_state.pt")
    dump_yaml(cfg, out_dir / "train_config.yaml")
    log_info(f"checkpoint 已保存: {out_dir}")


def resolve_resume_path(path: str) -> Path:
    """支持三种写法: 具体 step 目录 / 含 latest 的目录 / 直接指向 latest。"""
    p = Path(path)
    if (p / "trainer_state.pt").exists():
        return p
    latest = p / "latest"
    if latest.is_dir():
        latest = latest.resolve()
    if latest.is_file():
        name = latest.read_text(encoding="utf-8").strip()
        cand = p / name
        if cand.exists():
            return cand
    if (p / "model.safetensors").exists():
        return p
    raise FileNotFoundError(f"无法在 {p} 找到可恢复的 checkpoint (需要 trainer_state.pt)")


def load_checkpoint(resume_path: Path, model: nn.Module, optimizer, scheduler, scaler,
                    dist_info: DistInfo) -> Tuple[int, int]:
    """恢复权重 + optimizer/scheduler/scaler/step/epoch。"""
    raw = unwrap_model(model)
    weights = resume_path / "model.safetensors"
    if weights.exists():
        from safetensors.torch import load_file

        state = load_file(str(weights))
        missing, unexpected = raw.load_state_dict(state, strict=False)
        log_info(f"恢复权重自 {weights}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        warn(f"{resume_path} 中没有 model.safetensors, 仅恢复优化器状态")

    trainer_state = resume_path / "trainer_state.pt"
    if trainer_state.exists():
        state = torch.load(str(trainer_state), map_location="cpu", weights_only=False)
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            warn(f"optimizer 状态恢复失败 (可能 world_size 变化): {e}")
        if scheduler is not None and state.get("scheduler"):
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                warn(f"scheduler 状态恢复失败: {e}")
        if scaler is not None and state.get("scaler"):
            try:
                scaler.load_state_dict(state["scaler"])
            except Exception as e:
                warn(f"scaler 状态恢复失败: {e}")
        rng = state.get("rng") or {}
        if "torch" in rng and torch.is_tensor(rng["torch"]):
            torch.set_rng_state(rng["torch"].cpu())
        if "python" in rng:
            random.setstate(rng["python"])
        step = int(state.get("step", 0))
        epoch = int(state.get("epoch", 0))
        log_info(f"恢复训练状态: step={step} epoch={epoch}")
        return step, epoch
    return 0, 0


def prune_checkpoints(save_dir: Path, keep: int) -> None:
    """只保留最近 keep 个 step_xxx 目录。"""
    if keep <= 0:
        return
    dirs = sorted([d for d in save_dir.glob("step_*") if d.is_dir()])
    for old in dirs[:-keep] if len(dirs) > keep else []:
        shutil.rmtree(old, ignore_errors=True)


# =========================================================================== #
# 12. 训练循环
# =========================================================================== #
class Meter:
    """滑动统计: loss / token 数 / 耗时。"""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.loss_cnt = 0
        self.tokens = 0
        self.start = time.time()

    def update(self, loss: float, tokens: int) -> None:
        self.loss_sum += loss
        self.loss_cnt += 1
        self.tokens += tokens

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.loss_cnt = 0
        self.tokens = 0
        self.start = time.time()

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(1, self.loss_cnt)

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.time() - self.start)


def _count_tokens(batch: Dict[str, Any]) -> int:
    mask = batch.get("attention_mask")
    if isinstance(mask, torch.Tensor):
        return int(mask.sum().item())
    return 0


def clip_grads(model: nn.Module, optimizer, max_norm: float) -> float:
    """梯度裁剪 (FSDP 用自带实现, 保证分片梯度的全局范数正确)。"""
    if max_norm is None or max_norm <= 0:
        return 0.0
    if type(model).__name__ == "FullyShardedDataParallel":
        return float(model.clip_grad_norm_(max_norm))
    params = [p for group in optimizer.param_groups for p in group["params"]]
    total = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total)


def lr_of_group(optimizer, group_name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("group_name") == group_name:
            return float(group["lr"])
    return 0.0


def _zero_grad_anchor(optimizer) -> Optional[torch.Tensor]:
    """
    构造一个 0 权重的标量锚点。

    场景: Stage1 (只训 projector) 遇到"纯文本 batch"时, loss 没有任何可训练参数参与,
          `loss.requires_grad == False` -> backward 直接报错; 即使跳过 backward,
          DDP/FSDP 的梯度同步语义也会被破坏 (本轮没有 backward 就没有 all-reduce)。
    处理: 给 loss 拼一个 0 权重的锚点, 让 backward 正常执行 (梯度为 0, 不改变训练效果),
          同时保证各 rank 的同步语义一致。
    """
    terms = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad and p.numel() > 0:
                terms.append(p.flatten()[0] * 0.0)
    if not terms:
        return None
    return terms[0] if len(terms) == 1 else torch.stack(terms).sum()


def train_sft(model, loader, optimizer, scheduler, scaler, cfg, dist_info,
              start_step: int, start_epoch: int, cfg_obj=None, tokenizer=None) -> int:
    """SFT 主循环 (Stage1 / Stage2 / Stage3 共用)。"""
    train_cfg = get_in(cfg, "train", {}) or {}
    epochs = int(train_cfg.get("epochs", 1))
    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    log_every = max(1, int(train_cfg.get("logging_steps", 10)))
    save_every = max(0, int(train_cfg.get("save_steps", 0)))
    eval_every = max(0, int(train_cfg.get("eval_steps", 0)))
    max_steps = int(train_cfg.get("max_steps", 0) or 0)
    save_dir = Path(train_cfg.get("save_dir", "runs/mmllm"))
    save_limit = int(train_cfg.get("save_total_limit", 3) or 0)
    precision = train_cfg.get("precision", "bf16")
    device_type = "cuda" if dist_info.device.type == "cuda" else "cpu"
    autocast_ctx, _ = make_autocast(precision, device_type)

    eval_loader = None
    eval_path = get_in(cfg, "data.eval_path")
    if eval_path and cfg_obj is not None and tokenizer is not None:
        eval_records = load_records(eval_path)
        eval_loader, _ = build_loader(eval_records, cfg_obj, tokenizer, cfg, dist_info)

    global_step = start_step
    meter = Meter()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for micro_step, batch in enumerate(loader):
            batch = to_device(batch, dist_info.device)
            is_accum_boundary = (micro_step + 1) % grad_accum == 0
            is_last_micro = (micro_step + 1) == len(loader)
            sync_now = is_accum_boundary or is_last_micro

            # 梯度累积的中间步用 no_sync 跳过梯度同步, 省带宽
            sync_ctx = contextlib.nullcontext()
            if dist_info.enabled and not sync_now and hasattr(model, "no_sync"):
                sync_ctx = model.no_sync()

            with sync_ctx:
                with autocast_ctx:
                    out = model(**batch)
                    loss = out.loss
                    if loss is None:
                        raise RuntimeError("model 返回的 loss 为 None, 请检查 labels 是否传入")
                    scaled = loss / grad_accum
                if not scaled.requires_grad:
                    # 该 batch 没有可训练参数参与 (如 stage1 的纯文本 batch)
                    anchor = _zero_grad_anchor(optimizer)
                    if anchor is None:
                        raise RuntimeError(
                            "loss 没有计算图且没有可训练参数, 请检查 freeze/lr 配置")
                    scaled = scaled + anchor
                scaler.scale(scaled).backward()

            meter.update(float(loss.detach().item()), _count_tokens(batch))

            if not sync_now:
                continue

            # ---- 梯度裁剪 + optimizer step ----
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = clip_grads(model, optimizer, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if global_step % log_every == 0:
                lr_info = " ".join(
                    f"{gp.get('group_name', '?')}={float(gp['lr']):.2e}"
                    for gp in optimizer.param_groups
                )
                mem = (torch.cuda.max_memory_allocated() / 1024 ** 3
                       if torch.cuda.is_available() else 0.0)
                tps = meter.tokens / meter.elapsed
                msg = (f"step {global_step}/{epochs}e | loss {meter.avg_loss:.4f} | "
                       f"lr {lr_info} | grad_norm {grad_norm:.3f} | "
                       f"tok/s {tps:.0f} | mem {mem:.1f}GB")
                log_info(msg, dist_info.rank)
                _append_log(save_dir, {"step": global_step, "loss": meter.avg_loss,
                                       "tokens_per_sec": tps, "grad_norm": grad_norm,
                                       "mem_gb": mem})
                meter.reset()

            if save_every and global_step % save_every == 0:
                _save_step(save_dir, model, optimizer, scheduler, scaler, global_step,
                           epoch, cfg, dist_info, save_limit)
            if eval_every and eval_loader is not None and global_step % eval_every == 0:
                eval_loss = evaluate(model, eval_loader, dist_info, autocast_ctx)
                log_info(f"eval @ step {global_step}: loss={eval_loss:.4f}", dist_info.rank)
                _append_log(save_dir, {"step": global_step, "eval_loss": eval_loss})

            if max_steps and global_step >= max_steps:
                log_info(f"达到 max_steps={max_steps}, 提前结束", dist_info.rank)
                _save_step(save_dir, model, optimizer, scheduler, scaler, global_step,
                           epoch, cfg, dist_info, save_limit)
                return global_step

    _save_step(save_dir, model, optimizer, scheduler, scaler, global_step,
               max(start_epoch, epochs - 1), cfg, dist_info, save_limit, force=False)
    return global_step


def _save_step(save_dir: Path, model, optimizer, scheduler, scaler, step, epoch,
               cfg, dist_info, save_limit: int, force: bool = True) -> None:
    import torch.distributed as dist

    step_dir = save_dir / f"step_{step:06d}"
    if not force and (step_dir / "trainer_state.pt").exists():
        # 收尾保存: 该 step 已经存过, 不重复写盘
        if dist_info.enabled:
            dist.barrier()
        return
    save_checkpoint(step_dir, model, optimizer, scheduler, scaler, step, epoch,
                    cfg, dist_info)
    if dist_info.is_main:
        (save_dir).mkdir(parents=True, exist_ok=True)
        (save_dir / "latest").write_text(step_dir.name, encoding="utf-8")
        prune_checkpoints(save_dir, save_limit)
    if dist_info.enabled:
        dist.barrier()


def _append_log(save_dir: Path, record: Dict[str, Any]) -> None:
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


@torch.no_grad()
def evaluate(model, loader, dist_info, autocast_ctx) -> float:
    """在 eval 集上算平均 loss (只做前向, 不建图)。"""
    if loader is None:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = to_device(batch, dist_info.device)
        with autocast_ctx:
            out = model(**batch)
        if out.loss is None:
            continue
        total += float(out.loss.item())
        count += 1
    model.train()
    if dist_info.enabled and count > 0:
        import torch.distributed as dist

        t = torch.tensor([total, float(count)], device=dist_info.device,
                         dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, count = float(t[0].item()), int(t[1].item())
    return total / max(1, count)


# =========================================================================== #
# 13. DPO (Stage3 可选)
# =========================================================================== #
def encode_text_only(conversations: List[Dict[str, str]], tokenizer, cfg_obj: MMConfig,
                     max_len: int) -> Dict[str, List[int]]:
    """
    DPO 走纯文本偏好对: 用与 MMDataset 一致的分段规则 tokenize,
    但只监督 assistant 轮次。图像/音频占位符在此被剔除 (DPO 不带模态特征)。
    """
    from mmllm.data import build_segments

    input_ids: List[int] = []
    labels: List[int] = []
    if getattr(cfg_obj.llm, "add_bos", False):
        bos = tokenizer.bos_token_id
        if bos is not None:
            input_ids.append(bos)
            labels.append(cfg_obj.label_pad_id)
    for seg in build_segments(conversations, cfg_obj):
        ids = tokenizer(seg["text"], add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        if seg["supervised"]:
            labels.extend(ids)
        else:
            labels.extend([cfg_obj.label_pad_id] * len(ids))
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def build_dpo_dataset(records: List[Dict[str, Any]], tokenizer, cfg_obj: MMConfig,
                      max_len: int = 4096) -> List[Dict[str, Any]]:
    """把 chosen/rejected 记录转成"偏好对"样本列表。"""
    pairs = []
    for rec in records:
        chosen = rec.get("chosen")
        rejected = rec.get("rejected")
        if not chosen or not rejected:
            continue
        c = encode_text_only(chosen, tokenizer, cfg_obj, max_len)
        r = encode_text_only(rejected, tokenizer, cfg_obj, max_len)
        pairs.append({"chosen": c, "rejected": r})
    return pairs


def dpo_collate(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    把 B 个偏好对拼成 2B 条序列: 前 B 条是 chosen, 后 B 条是 rejected,
    这样 pi[:B] 与 pi[B:] 一一对应, 配对关系不会错乱。
    """
    seqs = [s["chosen"] for s in samples] + [s["rejected"] for s in samples]
    max_len = max(len(s["input_ids"]) for s in seqs)
    pad_id = int(samples[0].get("pad_id", 0) or 0)
    label_pad_id = int(samples[0].get("label_pad_id", -100))
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(seqs), max_len), label_pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, it in enumerate(seqs):
        L = len(it["input_ids"])
        input_ids[i, :L] = torch.tensor(it["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(it["labels"], dtype=torch.long)
        attn[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def sequence_logprobs(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """逐序列的 token 对数概率之和 (只统计 labels 有效位)。"""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = out.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    tok_logp = torch.gather(logp, 2, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    return (tok_logp * mask).sum(-1)


def train_dpo(model, loader, optimizer, scheduler, scaler, cfg, dist_info,
              start_step: int, start_epoch: int) -> int:
    """DPO 主循环 (可选): 偏好对 -> 参考模型 logprob -> -logsigmoid 损失。"""
    train_cfg = get_in(cfg, "train", {}) or {}
    dpo_cfg = get_in(cfg, "train.dpo", {}) or {}
    beta = float(dpo_cfg.get("beta", 0.1))
    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
    log_every = max(1, int(train_cfg.get("logging_steps", 10)))
    save_every = max(0, int(train_cfg.get("save_steps", 0)))
    save_dir = Path(train_cfg.get("save_dir", "runs/mmllm"))
    save_limit = int(train_cfg.get("save_total_limit", 3) or 0)
    precision = train_cfg.get("precision", "bf16")
    device_type = "cuda" if dist_info.device.type == "cuda" else "cpu"
    autocast_ctx, _ = make_autocast(precision, device_type)

    # 参考模型: 优先用指定 checkpoint; 否则 LoRA 场景下关掉 adapter 等价于参考模型
    ref_model = None
    ref_path = dpo_cfg.get("ref_model")
    if ref_path:
        ref_model = MultimodalLM.from_pretrained(ref_path, map_location="cpu")
        ref_model.to(dist_info.device).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        log_info(f"DPO 参考模型: {ref_path}")
    else:
        warn("dpo.ref_model 未设置: 若是 LoRA 模式将用 '关闭 adapter 的策略模型' 作为参考; "
             "全参微调场景下没有参考模型会导致 DPO 退化, 建议先做 SFT 再指定 ref_model")

    global_step = start_step
    meter = Meter()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, int(train_cfg.get("epochs", 1))):
        for micro_step, batch in enumerate(loader):
            batch = to_device(batch, dist_info.device)
            is_last_micro = (micro_step + 1) == len(loader)
            sync_now = (micro_step + 1) % grad_accum == 0 or is_last_micro
            sync_ctx = contextlib.nullcontext()
            if dist_info.enabled and not sync_now and hasattr(model, "no_sync"):
                sync_ctx = model.no_sync()

            with sync_ctx:
                with autocast_ctx:
                    pi = sequence_logprobs(model, batch)
                    if ref_model is not None:
                        with torch.no_grad():
                            ref = sequence_logprobs(ref_model, batch)
                    else:
                        ref = torch.zeros_like(pi)
                    half = pi.shape[0] // 2
                    pi_c, pi_r = pi[:half], pi[half:]
                    ref_c, ref_r = ref[:half], ref[half:]
                    margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
                    loss = -F.logsigmoid(margin).mean()
                    scaled = loss / grad_accum
                if not scaled.requires_grad:
                    anchor = _zero_grad_anchor(optimizer)
                    if anchor is None:
                        raise RuntimeError("DPO loss 没有计算图且没有可训练参数")
                    scaled = scaled + anchor
                scaler.scale(scaled).backward()

            meter.update(float(loss.detach().item()), int(batch["input_ids"].numel()))
            if not sync_now:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = clip_grads(model, optimizer,
                                   float(train_cfg.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if global_step % log_every == 0:
                log_info(f"[dpo] step {global_step} | loss {meter.avg_loss:.4f} | "
                         f"grad_norm {grad_norm:.3f} | mem "
                         f"{(torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0:.1f}GB",
                         dist_info.rank)
                meter.reset()
            if save_every and global_step % save_every == 0:
                _save_step(save_dir, model, optimizer, scheduler, scaler, global_step,
                           epoch, cfg, dist_info, save_limit)
    _save_step(save_dir, model, optimizer, scheduler, scaler, global_step,
               int(train_cfg.get("epochs", 1)) - 1, cfg, dist_info, save_limit, force=False)
    return global_step


# =========================================================================== #
# 14. 命令行
# =========================================================================== #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多模态大模型训练入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                   help="覆盖任意配置项, 可多次, 例如 --set train.batch_size=4")
    p.add_argument("--resume", type=str, default=None, help="断点续训目录")
    p.add_argument("--output-dir", type=str, default=None, help="覆盖 train.save_dir")
    p.add_argument("--base-model", type=str, default=None,
                   help="覆盖 model.base_model (空=从零预训练)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr-projector", type=float, default=None)
    p.add_argument("--lr-llm", type=float, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--precision", type=str, default=None, choices=["bf16", "fp16", "fp32"])
    p.add_argument("--dist-backend", type=str, default=None, choices=["fsdp", "ddp", "none"])
    p.add_argument("--max-steps", type=int, default=None, help="调试用: 限制总步数")
    p.add_argument("--print-config", action="store_true", help="只打印最终配置后退出")
    p.add_argument("--local_rank", "--local-rank", type=int, default=None,
                   help=argparse.SUPPRESS)
    return p.parse_args(argv)


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "resume": ("train.resume", args.resume),
        "output_dir": ("train.save_dir", args.output_dir),
        "base_model": ("model.base_model", args.base_model),
        "seed": ("seed", args.seed),
        "epochs": ("train.epochs", args.epochs),
        "batch_size": ("train.batch_size", args.batch_size),
        "grad_accum": ("train.grad_accum", args.grad_accum),
        "lr_projector": ("train.lr.projector", args.lr_projector),
        "lr_llm": ("train.lr.llm", args.lr_llm),
        "max_seq_len": ("train.max_seq_len", args.max_seq_len),
        "precision": ("train.precision", args.precision),
        "dist_backend": ("train.dist_backend", args.dist_backend),
        "max_steps": ("train.max_steps", args.max_steps),
    }
    for _, (path, value) in mapping.items():
        if value is not None:
            set_in(cfg, path, value)
    return cfg


# =========================================================================== #
# 15. main
# =========================================================================== #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.config:
        here = Path(__file__).resolve().parent
        candidates = sorted((here / "configs").glob("*.yaml"))
        print("请用 --config 指定配置文件。可选:\n" +
              "\n".join(f"  --config {c}" for c in candidates))
        return 2

    cfg = load_yaml(args.config)
    cfg = apply_set_overrides(cfg, args.sets)
    cfg = apply_cli_overrides(cfg, args)

    if args.print_config:
        print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        return 0

    dist_info = init_dist(cfg)
    try:
        return _run(cfg, dist_info)
    finally:
        cleanup_dist(dist_info)


def _run(cfg: Dict[str, Any], dist_info: DistInfo) -> int:
    set_seed(int(cfg.get("seed", 42)), dist_info.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- 配置与模型 ----
    cfg_obj = build_model_cfg(cfg)
    model_cfg = get_in(cfg, "model", {}) or {}
    base_model = (model_cfg.get("base_model") or "").strip()
    if base_model and model_cfg.get("hf_override", True):
        override_from_hf_config(cfg_obj, base_model,
                                trust_remote_code=bool(model_cfg.get("trust_remote_code", False)))

    tokenizer = build_tokenizer(cfg_obj, cfg)
    model = build_model(cfg_obj, cfg)
    freeze = apply_freeze(model, cfg)
    check_stage_consistency(cfg, freeze)
    model, lora_info = maybe_apply_lora(model, cfg)

    train_cfg = get_in(cfg, "train", {}) or {}
    if train_cfg.get("gradient_checkpointing", True):
        enable_gradient_checkpointing(model)

    # ---- 数据 ----
    train_path = get_in(cfg, "data.train_path")
    records = load_records(train_path) if train_path else []
    if not records:
        raise RuntimeError("没有训练数据: 请在 data.train_path 指定 jsonl/json 文件")

    dpo_enabled = bool(get_in(cfg, "train.dpo.enabled", False))
    loader, collator = build_loader(records, cfg_obj, tokenizer, cfg, dist_info)

    # ---- 分布式包装 ----
    model = wrap_distributed(model, dist_info, cfg)

    # ---- 优化器 / 调度器 ----
    param_groups = build_param_groups(model, cfg)
    opt_cfg = get_in(cfg, "train.optimizer", {}) or {}
    betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
    default_lr = get_in(cfg, "train.lr.default", 1e-5)
    default_lr = 1e-5 if default_lr is None else float(default_lr)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=default_lr,
        betas=betas,
        eps=float(opt_cfg.get("eps", 1e-8)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        fused=torch.cuda.is_available(),
    )

    steps_per_epoch = max(1, len(loader) // max(1, int(train_cfg.get("grad_accum", 1))))
    total_steps = steps_per_epoch * int(train_cfg.get("epochs", 1))
    warmup_steps = int(train_cfg.get("warmup_steps", 0) or 0)
    if warmup_steps <= 0:
        warmup_steps = int(float(train_cfg.get("warmup_ratio", 0.03)) * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps,
                                float(train_cfg.get("min_lr_ratio", 0.1)))
    log_info(f"总步数={total_steps} (每 epoch {steps_per_epoch} 步) warmup={warmup_steps}")

    precision = train_cfg.get("precision", "bf16")
    device_type = "cuda" if dist_info.device.type == "cuda" else "cpu"
    scaler = make_scaler(precision, device_type)

    # ---- 断点续训 ----
    global_step, start_epoch = 0, 0
    resume = train_cfg.get("resume")
    if resume:
        resume_path = resolve_resume_path(resume)
        global_step, start_epoch = load_checkpoint(resume_path, model, optimizer,
                                                   scheduler, scaler, dist_info)

    # ---- 开训 ----
    if dpo_enabled:
        max_len = int(train_cfg.get("max_seq_len", 4096))
        pairs = build_dpo_dataset(records, tokenizer, cfg_obj, max_len)
        if not pairs:
            raise RuntimeError("DPO 需要含 chosen/rejected 字段的偏好数据")
        for p in pairs:
            p["pad_id"] = tokenizer.pad_token_id or 0
            p["label_pad_id"] = cfg_obj.label_pad_id
        dpo_loader = DataLoader(pairs, batch_size=int(train_cfg.get("batch_size", 2)),
                                shuffle=True, drop_last=False, collate_fn=dpo_collate)
        log_info(f"DPO 偏好对: {len(pairs)} 组 (每组 chosen/rejected)")
        train_dpo(model, dpo_loader, optimizer, scheduler, scaler, cfg, dist_info,
                  global_step, start_epoch)
    else:
        train_sft(model, loader, optimizer, scheduler, scaler, cfg, dist_info,
                  global_step, start_epoch, cfg_obj=cfg_obj, tokenizer=tokenizer)

    log_info("训练结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
