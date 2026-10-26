"""
mmllm — 从零构建文本/图像/音频多模态大模型的参考实现
=======================================================
模块划分:
    config.py     统一配置契约 (三塔 + 投影器 + 特殊 token)
    modeling.py   网络结构实现 (ViT / Whisper 音频塔 / 投影器 / GQA+RoPE+SwiGLU LLM)
    data.py       多模态数据集、对话模板、批处理
    tokenizer_ext 特殊 token 扩展与占位符静态展开
"""

from .config import MMConfig, LLMConfig, VisionConfig, AudioConfig, preset
from .modeling import (
    MultimodalLM,
    LanguageModel,
    VisionTower,
    AudioTower,
    MLPProjector,
    PerceiverResampler,
    AudioProjector,
    convert_hf_llm_state_dict,
)

__all__ = [
    "MMConfig", "LLMConfig", "VisionConfig", "AudioConfig", "preset",
    "MultimodalLM", "LanguageModel", "VisionTower", "AudioTower",
    "MLPProjector", "PerceiverResampler", "AudioProjector",
    "convert_hf_llm_state_dict",
]

__version__ = "0.1.0"
