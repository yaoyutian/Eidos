# Eidos
Spatial Intelligence for Thought and Action.

# 从零构建文本 / 图像 / 音频多模态大模型

> 一条序列、一座大脑、三座感官塔。
> 本仓库给出一套**结构清晰、可训练、可部署**的多模态大模型完整方案 + 可运行的 PyTorch 参考实现。

---

## 0. 一句话结论

不要建三个模型。建**三个编码塔 + 一个投影桥 + 一个 decoder-only LLM**：

```
文本 → TokenEmbedding ─────────────────────────────┐
图像 → ViT → Resampler → Projector ──┐              │
音频 → WhisperEnc → 时域池化 → Proj ──┼→ 同一条序列 → LLM → 文本输出
                                     └──────────────┘
```

关键在于：**所有模态最终都变成 token，共享同一个自回归解码器**。
这样只有一条 KV cache、只有一条训练路径、只有一条推理路径，
可直接吃 vLLM / SGLang 的标准多模态接口，不需要改引擎内核。

---

## 1. 架构范式选择（第一个必须定下来的决策）

| 维度 | **早融合 / 单序列（推荐）** | Cross-Attn 旁路 (Flamingo) | 独立模型 + 工具调用 |
|---|---|---|---|
| KV cache | 1 条，完全统一 | 1 条 + 旁路状态 | 多条，需编排 |
| 推理引擎 | vLLM / SGLang 原生支持 | 需改引擎内核 | 需多服务编排 + 胶水 |
| 视觉 token | 占用上下文预算 | 不占主序列 | 经文本中转，有损 |
| 长视频扩展 | token 线性增长 | 可交叉注意力池化 | 需先摘要 |
| 训练复杂度 | 低 | 中高（分层冻结训练） | 高（多模型对齐） |
| 跨模态融合深度 | 共享全部层 | 仅隔层注入 | 弱 |
| 适合自建团队 | ✅ | ❌ 引擎改造是硬伤 | ❌ 编排成本高 |

**结论**：除非有明确的引擎改造能力与长视频需求，否则选早融合。
它的代价（视觉 token 占用上下文）可以用**压缩策略**调节，而 Cross-Attn 的代价（改引擎）不可调节。

> 本仓库的 `preset("base")` 采用早融合。改造成 cross-attn 时，
> 只需替换 `MultimodalLM.forward` 中特征注入的方式，其余训练/部署逻辑不变。

---

## 2. 网络结构定义

### 2.1 文本塔

| 项 | 规格 |
|---|---|
| Tokenizer | byte-level BPE，32K 词表；**增加 `<image>` / `<audio>` / `<video>` 特殊 token** |
| Embedding | `nn.Embedding(32000, 4096)` |
| 特殊 token id | `<image>`=32001，`<audio>`=32002（写入 config.json，训练与部署共用） |

用 byte-level 而非词表级，是为了保证任何字符串都能编码，不会在多模态数据里出现 UNK。

### 2.2 视觉塔（ViT）

```
输入 (B, 3, 336, 336)
  → Conv2d(3→1024, k=14, s=14)          # patch embedding
  → + pos_embed                          # (B, 576, 1024)
  → 24 × [LN → MHA(16头) → LN → GELU-MLP(4096)]
  → LN                                   # (B, 576, 1024)
```

- 权重建议取 **CLIP ViT-L/14-336**：它自带图文对齐先验，比从零训 ViT 省掉一整个阶段。
- 双向注意力，无位置掩码（用可学习绝对位置编码）。
- `projector_hidden_mult` 等关键超参全部在 `VisionConfig` 里。

### 2.3 音频塔（Whisper 式编码器）

```
输入 (B, 128, 3000)   # 16kHz log-Mel, 30s
  → Conv1d(128→1280, k=3, s=1) + GELU
  → Conv1d(1280→1280, k=3, s=2) + GELU  # 下采样 2×
  → + pos_embed                         # (B, 1500, 1280)
  → 32 × [LN → MHA(20头) → LN → GELU-MLP(5120)]
  → LN                                  # (B, 1500, 1280)
```

只取 Whisper 的 **encoder**，丢弃 decoder（解码不需要）。
Whisper 已在 68 万小时音频上训练过，声学表示质量足够，不用自己训。

### 2.4 投影器（模态空间 → LLM token 空间的桥）

**视觉侧**（两种模式，token 数都由 `out_tokens` 唯一决定）：

| 模式 | 结构 | token 数 | 说明 |
|---|---|---|---|
| `perceiver`（默认） | 256 个可学习 latent → 2 层跨注意力 → MLP | 576 → 256 | 对输入长度不敏感，压缩率可调，质量最好 |
| `mlp` | patch 网格 2D 重采样 → MLP | 576 → **任意目标值** | 极简，LLaVA-1.5 思路 |

`mlp` 模式的做法：把 `(gh, gw)` 的 patch 网格用 2D 自适应平均池化重采样到 `(th, tw)`，
其中 `th*tw == out_tokens` 且尽量保持长宽比 —— 24×24→16×16=256、14×14→8×8=64、24×24→12×12=144 都能精确命中。
**不用固定 2×2 的 PixelShuffle**，因为它只能降为 1/4，落不到任意目标值上。

**音频侧**：

```
(B, 1500, 1280) → Conv1d(k=8,s=4) → Conv1d(k=5,s=5) → AdaptiveAvgPool(25)
                → MLP(1280→16384→4096) → LayerNorm
```

**两条必须遵守的规则**：

1. **输出必须做 LayerNorm**。文本 embedding 与视觉特征的尺度天然差几个数量级，
   不归一化会导致注意力 softmax 饱和、训练直接不收敛。这是最高频的踩坑点。
2. **每侧的 token 数必须精确等于配置里声明的值**。不能做成「按内容长度动态变」，
   否则 batch shape 不静态，连续批处理与 CUDA Graph 都用不了；
   更糟的是 tokenizer 按 `out_tokens` 展开占位符、模型却吐出别的数量，
   特征会**静默错位**到别的图片上 —— 训练看起来在跑，指标却莫名其妙。

### 2.5 LLM 主体

| 项 | 7B 档 | 70B 档 |
|---|---|---|
| 层数 / 维度 | 32 / 4096 | 80 / 8192 |
| 注意力头 | 32 Q 头 / 8 KV 头（GQA） | 64 Q 头 / 8 KV 头 |
| FFN | SwiGLU，14336 | SwiGLU，28672 |
| 归一化 | RMSNorm（pre-norm） | RMSNorm |
| 位置编码 | RoPE（θ=10000） | RoPE |
| 上下文 | 32K | 32K |

**为什么用 GQA**：KV cache = `2 × layers × kv_heads × head_dim × 2 bytes`。
70B 用 8 个 KV 头而非 64 个，KV cache 直接降到 1/4——这是长上下文部署的生死线。

### 2.6 序列组装规则（整套方案的核心约束）

对话文本中的占位符在送入模型前**静态展开**：

```
"<image>"  →  [32001] × 256     # 每张图 256 个视觉占位符
"<audio>"  →  [32002] × 25      # 每 30 秒音频 25 个音频占位符
```

模型内部通过掩码把特征**原位替换**回 embedding：

```python
mask = (input_ids == cfg.image_token_id)          # (B, L) bool
embeds = MultimodalLM._scatter(embeds, mask, visual_feats)
```

> **实现陷阱**：同一 batch 内不同样本的图片数量不同（有的 0 张，有的 2 张）。
> 模态特征是沿 batch 维堆叠的 `(N_items, K, D)`，**不能**假设第 b 行属于样本 b。
> 必须按「每样本模态项数」算出全局偏移量再取回。仓库里的 `_scatter` 已按此实现，
> 并对「占位符数不是 K 的整数倍」直接报错——宁可崩，也不要让特征静默错位到别的图片上。

### 2.7 规模档位

| 预设 | LLM | 视觉 token | 音频 token/chunk | 最低训练配置 |
|---|---|---|---|---|
| `tiny` | 24L / 1024d | 64 | 12 | 1× RTX 4090 |
| `small` | 32L / 2560d | 256 | 25 | 4× A100-80G |
| `base` | 32L / 4096d | 256 | 25 | 8× A100-80G |
| `large` | 80L / 8192d | 256 | 25 | 16× A100-80G（+PP） |

---

## 3. 数据

### 3.1 数据源

| 阶段 | 文本 | 图像 | 音频 |
|---|---|---|---|
| Stage 1 | — | CC3M / LAION / COYO，595K 图文对 | LibriSpeech / Common Voice，100K 转写对 |
| Stage 2 | ShareGPT / UltraChat | LLaVA-Instruct-150K、多图对话 158K、DocVQA 233K、ChartQA、ScienceQA | 语音指令、音频描述 |
| Stage 3 | 指令数据 | 混合保留 | AudioCaps、ESC-50、AudioQA、语音对话 |

### 3.2 统一数据格式

```json
{
  "conversations": [
    {"role": "user",      "content": "<image>\n这张图里是什么?"},
    {"role": "assistant", "content": "一只橘猫趴在窗台上。"}
  ],
  "images": ["/data/001.jpg"],
  "audios": []
}
```

一个 `<image>` 对应 `images` 里的一张图，顺序严格对应。音频同理，但**长音频会自动切成多个 30s chunk**，
每个 chunk 展开 25 个 `<audio>` 占位符。

### 3.3 标签掩码

只有 assistant 轮次的 token 参与 loss：

```
<|im_start|>user\n<image>...<|im_end|>          → 全部 label = -100
<|im_start|>assistant\n一只橘猫。<|im_end|>    → 参与 loss
```

实现上采用**分段 tokenize 再拼接**，而不是整段 tokenize 后找区间——
BPE 在拼接边界上会改写前一段的切分，整段 tokenize 再用字符串匹配定位标签是不可靠的。

### 3.4 混合采样

分模态分桶，按比例采样，避免某个模态在每个 batch 里缺席导致梯度抖动：

```
text-only 40% | image 35% | audio 15% | image+audio+text 10%
```

---

## 4. 训练

### 4.1 四阶段

| 阶段 | 解冻 | 学习率 | 数据 | 目的 |
|---|---|---|---|---|
| **0 基座** | 全部（若从零） | 3e-4 | 文本 2T+ token | 打底座。**除非有数据优势，否则直接用开源基座** |
| **1 对齐** | 仅投影器 | 投影器 1e-3 | 595K 图文对 | 让投影器学会「翻译」 |
| **2 指令** | LLM + 投影器 | LLM 2e-5 / 投影器 2e-4 | 多模态指令数据 | 建立「指令→视觉推理」映射 |
| **3 全模态 SFT** | LLM + 投影器 + 音频塔 | 全部 1e-5 量级 | 加入音频，混合采样 | 同时听/看/说 |

**Stage 1 为什么最重要**：这是唯一一个「用极小的可训练参数量（实测 tiny 预设下占比 8.82%）
去学习整个模态接口」的阶段。如果 Stage 1 没做好（投影器没学到尺度对齐），
后面无论解冻多少层都救不回来。而且 Stage 1 最便宜——值得多跑几组超参。

### 4.2 参数分组学习率

不同模块的学习率要差 2～3 个数量级：

| 分组 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| 投影器 / Resampler | 1e-3 | 2e-4 | 2e-4 |
| LLM 主干 | 0（冻） | 2e-5 | 1e-5 |
| 视觉塔 | 0（冻） | 0（冻） | 0（冻，或 5e-6） |
| 音频塔 | 0（冻） | 0（冻） | 2e-5 |

norm 与 bias 不做 weight decay。梯度裁剪 `max_norm=1.0`。bf16 + cosine + 3% warmup。

### 4.3 并行策略

```
7B  全参:  FSDP (use_orig_params=True) + 梯度检查点 + bf16   → 8× A100-80G
70B 全参:  FSDP + TP=8 + PP=2 + 激活检查点                  → 32× A100-80G
7B  LoRA:  单机 4 卡即可                                    → 4× A100-80G
```

`use_orig_params=True` 很重要：分片后参数必须保持原始视图，
否则 `model.trainable_parameters()` 的分组逻辑会错。

### 4.4 显存估算

**训练（AdamW 全参）** ≈ `16 bytes/param` + 激活

| 规模 | 权重+梯度+优化器 | 建议卡数 |
|---|---|---|
| 7B | 112 GB | 8× A100-80G |
| 70B | 1120 GB | 16× A100-80G 起（ZeoRO-3 分片） |
| 7B LoRA | ~1.5 GB | 1× A100-80G |

**推理 KV cache** = `2 × layers × kv_heads × head_dim × 2 bytes / token`

| 规模 | 8K 上下文 | 32K 上下文 |
|---|---|---|
| 7B | 1.0 GB / 并发 | 4.0 GB / 并发 |
| 70B | 2.5 GB / 并发 | 10 GB / 并发 |

### 4.5 训练期高频坑

1. **投影输出不做 LayerNorm** → 尺度失配，loss 不降。必须做。
2. **占位符数量与特征数量不匹配** → 特征静默错位到别的图片上，训练看着能收敛但效果诡异。
   必须强校验 + 报错。
3. **一个 batch 里样本的模态数量不同** → `_scatter` 的偏移量算错（见 2.6）。
4. **标签掩码用整段 tokenize 定位** → BPE 边界改写导致掩码错位。
5. **超长对话从头截断** → 最终回答被整段切掉，该样本所有 label 变成 -100，
   监督信号全丢、白跑一个 batch。正确做法是**丢弃最早的对话轮次**，保住最近的上下文与最终回答。
6. **音频塔和 LLM 用同一个 LR** → 音频梯度量级大得多，会冲垮语义能力。
7. **Stage 1 用了全参微调** → 基座被冲垮，后面几个阶段都救不回来。
8. **RoPE 加在 KV 拼接之后** → 缓存里的 key 被用错误位置重复旋转，
   增量解码与一次性前向结果不一致（贪心采样下肉眼可见跑偏）。必须先旋转再拼接。
9. **投影器实际吐出的 token 数与配置声明不一致** → tokenizer 按 `out_tokens` 展开占位符，
   模型却吐出别的数量，特征静默错位。投影器的压缩方式必须让输出**精确等于** `out_tokens`，
   并用断言守住。

---

## 5. 部署

### 5.1 导出

本仓库权重格式是 `config.json` + `model.safetensors`。两种落地路径：

- **路径 A（推荐）**：`src/deploy/export_hf.py` 导出成 HF `LlavaForConditionalGeneration` 目录，
  直接 `vllm serve <dir>`，图像走 `image_url` 字段，零引擎改造。
- **路径 B（完全自定义）**：用 vLLM 的 `MULTIMODAL_REGISTRY` 注册自定义 `MultiModalProcessor`，
  完整控制视觉/音频预处理与特征注入。

音频路径注意：vLLM 对音频模态的支持晚于图像，且各版本差异大。
生产上更稳的做法是**把 Whisper encoder 前置为独立服务**，输出离散 token 或文本再进 LLM。

### 5.2 服务

```
客户端 → API 网关(限流/熔断/占位符展开) → vLLM OpenAI 兼容服务 → GPU 池
```

- 图像：下载 → RGB → 双三次缩放 → CLIP 归一化 → `(3, 336, 336)`
- 音频：解码 → 16kHz 重采样 → 30s 分块 → log-Mel(128, 3000) → 归一化
- **占位符展开在网关侧完成**，模型侧只见到 token id

### 5.3 性能调优

| 杠杆 | 效果 | 代价 |
|---|---|---|
| 视觉 token 256→64 | prefill −75% | OCR/图表掉点 |
| Prefix Caching | 系统提示与同一张图跨轮复用 | 前缀需稳定 |
| GQA | KV cache 降至 1/4 | 极长上下文略降 |
| AWQ W4A16 / FP8 | 显存减半，并发翻倍 | 需重跑评测 |
| Chunked Prefill | 长视觉前缀不阻塞其他请求 | 单请求 TTFT 略增 |
| 投机解码 | 输出吞吐 1.5–2× | 需调度配合 |

### 5.4 生产化清单

限流 / 熔断 / 超时分级 / 鉴权 / 多副本 / 灰度发布 / TTFT·TPOT·P99 监控 / GPU 利用率 / 错误率。

---

## 6. 评测

| 维度 | Benchmark |
|---|---|
| 文本通用 | MMLU、GSM8K、HumanEval、IFEval |
| 视觉推理 | MMBench、MME、MathVista、SEED-Bench、MMStar |
| 文档/图表 | DocVQA、ChartQA、InfoVQA、OCR-VQA |
| 音频 | AISHELL-2 / WenetSpeech（CER）、LibriSpeech（WER）、AudioBench、MMAU |
| 全模态 | MMMU、OmniBench |

**必须做消融**：视觉 token 数、投影器结构、Stage 1 学习率与数据量，
这三项对最终效果的影响远大于 LLM 主干的选择。

---

## 7. 实施路线图

### 路线 A：最小可行验证（1 人 · 约 2 周）

1. `preset("tiny")` + `Stage 1` 对齐训练，先让模型能对图片做一句话描述
2. 打通 `save_pretrained` → `from_pretrained` → `generate` 的完整回路
3. 验证 `export_hf.py` → vLLM 起服务
4. 目标：证明通路成立，不追求指标

### 路线 B：可用产品（3–5 人 · 约 2–3 月）

1. `preset("base")`（7B）+ Stage 1/2/3 全流程
2. 数据流水线工程化（去重、过滤、混合采样、断点续训）
3. vLLM 部署 + 量化 + 监控
4. 建立评测集与消融基线

### 路线 C：自建基座（10+ 人 · 6 月+）

仅在有数据与算力优势时考虑：Stage 0 从零预训练基座（7B 需约 15T token）。
**绝大多数团队应该走路线 B**——把算力投在模态对齐与数据质量上，
而不是重复训练一个已经存在的语言基座。

---

## 8. 参考实现文件

```
src/mmllm/config.py        三塔配置契约、规模预设          ← 唯一事实来源
src/mmllm/modeling.py      网络结构全部实现
src/mmllm/data.py          图像/音频预处理、对话模板、标签掩码、批处理
src/mmllm/tokenizer_ext.py 特殊 token 扩展与占位符静态展开
src/tests/test_multimodal.py 回归测试（4 条不变量）
src/train/                 四阶段训练脚本 + YAML 配置 + 说明
src/deploy/                vLLM 多模态注册、HF 权重导出、FastAPI 服务
docs/architecture.html     可视化架构蓝图
```

**已验证**（Python 3.12 + PyTorch 2.8，CPU 环境实跑）：

```
配置: LLM 24L/1024d | 视觉 196 patch -> 64 token | 音频 1500 帧 -> 12 token
logits         (2, 91, 33000)
视觉特征       (2, 64, 1024)      音频特征       (1, 12, 1024)
占位符校验     图像 128 / 视觉特征 128  ✓   音频 12 / 音频特征 12  ✓
梯度 vision_tower |grad| = 147143   audio_tower |grad| = 14553
     vision_projector |grad| = 66391  audio_projector |grad| = 9200
     language_model |grad| = 346611
Stage1(只训投影器) 可训练 49.46M / 总计 560.47M (8.82%)

[PASS] T1 形状与梯度
[PASS] T2 变长批次对齐      (同 batch 内 2图/1图/0图 混排, 特征不串位)
[PASS] T3 增量解码一致性    (带 KV cache == 逐 token 全量前向)
[PASS] T4 超长对话保留回答  (保留 337 个监督 token)
[PASS] T5 token 数契约      (perceiver/mlp 两种模式 × 64/100/256 均精确命中)
```

> T2 / T3 / T4 / T5 是**已修 bug 的回归护栏**：跨 batch 变长特征错位、RoPE 增量解码错位、
> 超长对话截断丢回答、投影器实际 token 数与配置声明不符。
> 这几条都属于「训练看起来在跑、但效果莫名不对」的隐形坑，没有断言守着几乎不可能靠肉眼发现。

复现：

```bash
cd src
python -c "from mmllm.modeling import _selfcheck; _selfcheck()"
python -m mmllm.data
python tests/test_multimodal.py
```

> T2 / T3 / T4 是**已修 bug 的回归护栏**：跨 batch 变长特征错位、RoPE 增量解码错位、
> 超长对话截断丢回答。这三条都属于「训练看起来在跑、但效果莫名不对」的隐形坑，
> 没有断言守着几乎不可能靠肉眼发现。

---

## 9. 速查清单

**架构**
- [ ] 融合范式选定（默认早融合）
- [ ] 每模态 token 预算定死（视觉 256 / 音频 25 per 30s）
- [ ] 占位符 id 写进 config，训练与部署共用
- [ ] 投影输出做 LayerNorm

**训练**
- [ ] Stage 1 只训投影器，LR 1e-3
- [ ] 分组学习率，norm/bias 不做 weight decay
- [ ] 标签只在 assistant 轮
- [ ] 占位符 / 特征数量强校验
- [ ] bf16 + 梯度检查点 + 梯度裁剪 1.0
- [ ] 断点续训可用

**部署**
- [ ] 导出格式与推理引擎对齐
- [ ] 网关侧完成占位符展开
- [ ] Prefix Caching 开启
- [ ] TTFT / TPOT / P99 上监控
- [ ] 量化后重跑评测
