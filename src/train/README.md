# 训练模块使用说明 (src/train)

本目录是多模态大模型（文本 / 图像 / 音频）的**训练侧入口**。网络结构不在这里实现，
一律复用 `src/mmllm/modeling.py` 的 `MultimodalLM`；数据管线复用 `src/mmllm/data.py`
的 `MMDataset` / `MMCollator`。

```
src/train/
├── train.py                      # 训练入口 (唯一可执行文件)
├── configs/
│   ├── stage1_align.yaml         # 阶段1: 模态对齐 (只训 projector)
│   ├── stage2_instruct.yaml      # 阶段2: 多模态指令微调 (LLM + projector)
│   └── stage3_omni_sft.yaml      # 阶段3: 全模态 SFT (含音频, 可选 LoRA / DPO)
└── README.md                     # 本文件
```

---

## 1. 五分钟上手

### 1.1 单卡（单机单卡，不初始化分布式）

```bash
cd D:\8yaodn-tmp\minimax-home\ttai

# 阶段1: 模态对齐 (先跑这个)
python -m src.train.train --config src/train/configs/stage1_align.yaml \
  --set data.train_path=data/stage1_captions.jsonl \
  --set train.save_dir=runs/stage1_align
```

### 1.2 多卡（torchrun）

```bash
cd D:\8yaodn-tmp\minimax-home\ttai

# 阶段2: 指令微调 (8 卡)
torchrun --nproc_per_node 8 --master_port 29501 -m src.train.train \
  --config src/train/configs/stage2_instruct.yaml \
  --set data.train_path=data/stage2_instruct.jsonl \
  --set train.save_dir=runs/stage2_instruct

# 阶段3: 全模态 SFT (含音频; 显存紧张就开 LoRA)
torchrun --nproc_per_node 8 --master_port 29501 -m src.train.train \
  --config src/train/configs/stage3_omni_sft.yaml \
  --set data.train_path=data/stage3_omni_sft.jsonl \
  --set train.lora.enabled=true
```

> `python -m src.train.train` 与 `python src/train/train.py` 两种写法都可用（脚本内部会把
> `src/` 加入 `sys.path`）。Windows 下 `torchrun` 用法与 Linux 完全一致。

### 1.3 断点续训

```bash
# latest 是一个文本文件, 里面记录着最新的 step_xxx 目录名
python -m src.train.train --config src/train/configs/stage2_instruct.yaml \
  --set data.train_path=data/stage2_instruct.jsonl \
  --resume runs/stage2_instruct/latest
```

### 1.4 从零预训练 vs 复用开源基座

| 模式 | 配置 | 说明 |
|---|---|---|
| **从零预训练** | `model.base_model: ""`（默认） | 全部参数随机初始化，LLM 主干也从头学 |
| **复用开源基座** | `--base-model meta-llama/Llama-2-7b-hf` 或 `--set model.base_model=...` | 用 `convert_hf_llm_state_dict()` 把 HF Llama/Qwen 权重映射后灌入 `model.language_model` |

```bash
# 复用开源基座 (注意: 转换函数产出的 key 不带 language_model. 前缀, 脚本会自动补齐)
python -m src.train.train --config src/train/configs/stage1_align.yaml \
  --base-model meta-llama/Llama-2-7b-hf \
  --set model.preset=base
```

`model.hf_override: true`（默认）会用 HF config 的 `hidden_size / num_layers / num_heads /
num_kv_heads / head_dim / intermediate_size / vocab_size` 等字段覆盖本项目的 `LLMConfig`，
避免"preset 写错 → 权重 shape 对不上"。shape 仍不匹配的权重会被跳过并打印告警。

### 1.5 小模型冒烟（几分钟验证整条链路）

```bash
python -m src.train.train --config src/train/configs/stage1_align.yaml \
  --set model.preset=tiny \
  --set train.batch_size=2 --set train.grad_accum=1 \
  --set train.max_steps=5 --set train.precision=fp32 \
  --set train.save_dir=runs/smoke
```

---

## 2. 三个阶段在做什么（分工逻辑）

三个阶段不是"把同一份数据训三遍"，而是**能力逐层叠加**，每阶段只解冻必要的那部分参数：

| | 阶段1 对齐 | 阶段2 指令微调 | 阶段3 全模态 SFT |
|---|---|---|---|
| **解冻什么** | 只有 projector（含 `vision_resampler` / `audio_projector`） | LLM + projector；视觉塔、音频塔仍冻结 | LLM + projector + **音频塔**；视觉塔默认仍冻结 |
| **用什么数据** | 图文 caption 对（短、单图、答案短）；可选音频-文本对齐对 | 多模态指令数据（多图、多轮、OCR、图表问答） | 语音问答 / 语音指令 / ASR+解释 / 图+音+文混合；可选偏好对（chosen/rejected） |
| **学到什么** | 模态特征 → LLM 词嵌入空间的**对齐**：让视觉/音频 token 对 LLM 来说是"可读"的 | **指令跟随 + 多轮对话格式** + 视觉内容与语言输出的联合建模 | **听觉能力** + 跨模态联合推理；再做 DPO 可把**回答质量/风格**往上抬 |
| **为什么这样切** | LLM 冻结 → 不会把基座的语言能力冲掉；projector 用大 lr（1e-3）快速收敛 | 解冻 LLM 让模型学会"按人的要求回答"，但视觉塔冻结保住通用视觉表征 | 音频塔从零学代价大（需与 LLM 一起训），故放在最后、且用更小 lr；视觉塔已有 CLIP 先验，默认不再动 |
| **显存** | 最省（LLM 冻结 → 无优化器状态） | 最贵（LLM 全参 → 优化器状态 ×2.5） | Stage2 + 音频塔参数量 |

流水线是 **Stage1 → Stage2 → Stage3 串行**：Stage2/3 的 `model.base_model` 建议填上一阶段
产出的目录（`runs/stage1_align/latest`），而不是重新随机初始化。

---

## 3. 配置说明

三个 YAML 结构完全一致，按需改。以下是最常用的几组：

### 3.1 冻结与分组学习率

```yaml
train:
  freeze:            # -> model.set_trainable(llm, vision, audio, projector)
    llm: true
    vision: false
    audio: true
    projector: true
  lr:                # 四组独立学习率; 填 0 等价于"该组不训"
    projector: 1.0e-4
    llm: 1.0e-5
    vision: 0.0
    audio: 1.0e-5
  weight_decay:
    default: 0.01
    projector: 0.0
  no_decay_keywords: ["norm", "bias", "ln_", "layernorm"]   # norm/bias 一律不做 wd
```

参数分组是**依据 `model.trainable_parameters()` 的 id** 做的：脚本遍历 `named_parameters()`，
只把 `requires_grad=True` 且名字归属 projector / llm / vision / audio 的参数收进对应组；
每组再按 `no_decay_keywords` 拆成 decay / no_decay 两个子组。

### 3.2 精度 / 显存 / 性能

```yaml
train:
  precision: bf16                  # bf16 | fp16 | fp32；GradScaler 只在 fp16 启用
  gradient_checkpointing: true     # 对 decoder layer 挂 torch.utils.checkpoint
  max_grad_norm: 1.0              # 梯度裁剪
  batch_size: 4                    # 每卡 batch
  grad_accum: 8                    # 有效 batch = batch_size × grad_accum × world_size
  max_seq_len: 4096
```

### 3.3 调度与保存

```yaml
train:
  epochs: 1
  warmup_ratio: 0.03               # warmup_steps > 0 时优先用 warmup_steps
  warmup_steps: 200
  min_lr_ratio: 0.1                # cosine 衰减到 (1 - 0.9) 的 lr
  save_dir: "runs/stage2_instruct"
  save_steps: 500
  eval_steps: 500                  # 需要 data.eval_path
  logging_steps: 10
  save_total_limit: 3              # 只保留最近 3 个 checkpoint
  resume: null
```

### 3.4 分布式

```yaml
train:
  dist_backend: fsdp               # fsdp | ddp | none
  fsdp:
    sharding_strategy: full_shard  # full_shard / shard_grad_op / hybrid_shard / no_shard
    cpu_offload: false
    sync_module_states: true
  ddp:
    find_unused_parameters: true
```

- `use_orig_params=True` 是**必须的**：只有保留原始 `nn.Parameter` 对象，才能按
  projector/llm 分组设置不同学习率。
- `auto_wrap_policy=ModuleWrapPolicy({TransformerBlock})`：逐个 decoder block 分片。
- 单机单卡（不经过 `torchrun`）时**不会初始化 process group**，直接单进程训练。

### 3.5 命令行覆盖

```bash
python -m src.train.train --config <yaml> \
  --set train.batch_size=2 --set train.lr.projector=5.0e-4 \
  --resume runs/stage2/latest --output-dir runs/exp2 --base-model Qwen/Qwen2-7B \
  --precision bf16 --max-steps 1000
```

`--set` 支持任意点分路径（`train.lr.projector=1e-3`），值按 YAML 标量解析。

---

## 4. 数据格式

`data.train_path` 支持 `.jsonl`（每行一条）或 `.json`（一个 list）：

```json
{"conversations": [{"role": "user", "content": "<image>\n描述这张图"},
                   {"role": "assistant", "content": "这是一只橘猫。"}],
 "images": ["/data/a.jpg"],
 "audios": ["/data/a.wav"]}
```

- `<image>` / `<audio>` 是**占位标记**，送入模型前会被静态展开成固定数量的特殊 token
  （每图 `vision.out_tokens` 个，每 30s 音频 `audio.out_tokens_per_chunk` 个）。
- 只有 assistant 轮次计入 loss（`data.py` 内部用 `label_pad_id=-100` 掩码）。
- 图像/音频预处理用 `data.py` 内置的 `SimpleImageProcessor` / `SimpleAudioProcessor`
  （只依赖 PIL + numpy + torch，不依赖 torchvision / torchaudio）。

---

## 5. 显存估算口径

**口径（混合精度训练，master weight 保持 fp32）**：

```
训练显存 ≈ 16 bytes × 可训练参数量          # fp32 master(4) + fp32 grad(4) + Adam m(4) + Adam v(4)
         + 激活                             # 开梯度检查点后约 4~10 GB，随 seq_len / batch 变化
         + logits 临时张量                  # B × L × vocab × 4B（loss 里的 float() 拷贝）
         + 冻结塔的 fp32 权重               # 视觉塔 ~0.3B + 音频塔 ~0.6B ≈ 3.7 GB
```

7B LLM 的实测口径：

| 场景 | 优化器+梯度状态 | 建议配置 | 结论 |
|---|---|---|---|
| 7B **全参**微调 | 7B × 16B ≈ **112 GB** | FSDP full_shard，**4 × A100 80G** | 4 卡：112/4=28GB + 激活 ≈ 38GB/卡，舒适；2 卡：66GB/卡，勉强；**1 卡 80G 装不下** |
| 7B 全参 + 开梯度检查点 | 同上 | 8 × A100 80G | 14GB + 激活 ≈ 24GB/卡，最稳 |
| 7B **LoRA** (r=16) | 可训参数 ~0.1% → **< 1 GB** | **1 × A100 80G** | 权重常驻 28GB + 激活 ≈ 40GB，单卡可行 |
| Stage1（LLM 冻结，只训 projector） | ≈ 0 | **1 × A100 80G** | projector 约 0.05B，显存主要是前向激活 |
| Stage3（再解冻音频塔 0.6B） | +0.6B × 16B ≈ **+10 GB** | 4~8 × A100 80G | 或用 LoRA 把优化器状态压回 O(1GB) |

> 换算示例：8 卡 A100 80G + 7B 全参 + `seq_len=4096` + `batch_size=1` + 梯度检查点，
> 每卡 ≈ 14GB(状态) + 8GB(激活) + 1GB(logits) ≈ 23GB，余量充足。
> 反过来，**单卡 80G 训 7B 全参是做不到的**，必须 FSDP 分片或换 LoRA。

---

## 6. 已知坑与注意事项

1. **占位符数量必须严格对齐**。`modeling.py` 会校验 `input_ids == image_token_id` 的个数
   是否等于 `B_img × vision.out_tokens`，不等就直接 raise。改 `out_tokens` 时，训练与推理
   的 tokenizer 扩展必须同步。
2. **词表要留够特殊 token**。脚本在拿到 tokenizer 后调用 `add_multimodal_tokens()`，
   并把 `cfg.llm.vocab_size` 抬到 `len(tokenizer)`，所以 embedding 一定够大；
   但 HF 基座的 embedding 比我们少 3 行（新增的 `<image>/<audio>/<video>`），
   这部分保持随机初始化——脚本按行部分拷贝，不会报错。
3. **logits 显存是隐藏大头**。`modeling.py` 的 loss 把 logits `.float()` 了一份，
   7B + 32k 词表下每个样本约 0.5~1 GB。`max_seq_len` 建议先从 2048~4096 起步。
4. **Stage1 遇到纯文本 batch** 时 loss 没有任何可训练参数参与图，`backward()` 会直接报错。
   脚本用一个 0 权重锚点兜底（保证 DDP/FSDP 的梯度同步语义不被破坏），
   但该 batch 实际上不更新参数——对齐阶段请确保每条样本都带图/音频。
5. **`tie_word_embeddings: true`** 时 `lm_head.weight` 与 `embed_tokens.weight` 共享显存，
   safetensors 会拒绝保存共享张量。脚本在保存前临时解绑、保存后恢复，所以不报错，
   但加载回来的模型靠 `config.json` 里的 `tie_word_embeddings` 重新 tie。
6. **换卡数续训**：optimizer 状态按 world_size 分片，换卡数后 `optimizer.load_state_dict`
   可能失败——脚本已 try/except 并告警，此时从模型权重继续、优化器状态从头开始。
7. **peft 未安装时 LoRA 自动退化**：`train.lora.enabled: true` 但环境没有 `peft` 时，
   脚本会打印醒目告警并**退化为 LLM 全参微调**（功能不缺，显存按全参估算）。
8. **peft 与本项目 forward 的兼容性**：peft 会向 LLM 传 `output_attentions` 等本项目
   `LanguageModel.forward` 不认的参数，脚本在运行时给 `LanguageModel.forward` 打了
   兼容补丁（**不修改 `src/mmllm/` 下的任何文件**）。
9. **DPO 是纯文本偏好对**：`<image>` / `<audio>` 占位符在 DPO 路径里会被剔除；
   `train.dpo.ref_model` 强烈建议填一个已 SFT 好的目录，否则没有真正的参考模型，
   DPO 会退化（脚本会告警）。
10. **视觉塔 / 音频塔权重默认随机初始化**。脚本目前只负责把 HF 基座灌进 LLM 主干；
    ViT / Whisper 需要自己另外加载（Stage1 本来就是要把它们训出来）。
11. **`data.py` 的长文本截断是从头截**（保留头部），超长样本的结尾回复可能被切掉，
    数据预处理阶段建议先过滤掉超长样本。
12. **checkpoint 目录结构**：
    ```
    runs/stage2_instruct/
    ├── step_000500/{model.safetensors, config.json, trainer_state.pt, train_config.yaml}
    ├── step_001000/...
    ├── latest            # 文本文件, 内容是最新 step 目录名
    └── train_log.jsonl   # 每行一条 {step, loss, tokens_per_sec, grad_norm, mem_gb}
    ```
    `model.safetensors` 可直接被 `MultimodalLM.from_pretrained(dir)` 读回做推理。

---

## 7. 训练脚本能力自检清单

- [x] YAML(PyYAML) + argparse，无新依赖
- [x] 四组（projector / llm / vision / audio）独立 lr 与 weight decay，norm/bias 不做 wd
- [x] bf16 autocast；GradScaler 仅 fp16 启用；fp32 可跑 CPU 冒烟
- [x] 梯度检查点（`torch.utils.checkpoint`，`use_reentrant=False`，与 FSDP 兼容）
- [x] `torchrun` 兼容：FSDP（按 TransformerBlock 自动包裹）/ DDP / 单机单卡不初始化 dist
- [x] 梯度累积 + `no_sync`；梯度裁剪 max_norm=1.0
- [x] warmup + cosine decay
- [x] 断点续训：model / optimizer / scheduler / scaler / step / epoch
- [x] 日志：loss / 各组 lr / grad_norm / tokens_per_sec / `torch.cuda.max_memory_allocated`
- [x] checkpoint 走 `model.save_pretrained(out_dir)`，同时保存 optimizer 状态
- [x] HF Llama/Qwen 权重映射（`convert_hf_llm_state_dict` + `language_model.` 前缀）
- [x] 从零预训练 vs 复用开源基座开关
