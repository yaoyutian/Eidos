# mmllm 部署手册

> 面向"从零构建文本 / 图像 / 音频多模态大模型"项目的部署模块。
> 覆盖三种形态：**HF Llava 导出**（路径 A）、**vLLM 自定义注册**（路径 B）、**本项目自带的 FastAPI 门面**（serve.py）。

---

## 0. 目录与文件

| 文件 | 作用 |
| --- | --- |
| `src/deploy/export_hf.py` | 把 `model.safetensors + config.json` 导出成 HF `LlavaForConditionalGeneration` 目录（含分片 + index） |
| `src/deploy/vllm_multimodal.py` | vLLM 多模态注册：`MultiModalProcessor`（路径 B）、`ModelRegistry.register_model`、跨版本兼容层、纯 Python 自检 |
| `src/deploy/serve.py` | 对外服务：OpenAI 兼容 `/v1/chat/completions`、裸接口 `/generate`、`/healthz`、`/metrics` |
| `src/deploy/README.md` | 本文件 |

### 占位符契约（训练与部署必须一致）

```
每张图            ->  256 个 <image> token      (cfg.vision.out_tokens)
每 30 秒音频      ->   25 个 <audio> token      (cfg.audio.out_tokens_per_chunk)
```

模型内部按 `input_ids == cfg.image_token_id` / `cfg.audio_token_id` 的掩码位置**原位替换**特征，
因此 prompt 里占位符的数量必须与实际输入的模态数量严格对应。

---

## 1. 部署拓扑

### 1.1 路径 A（推荐用于 HF 生态）：导出 → vLLM 直接加载

```
   客户端 (OpenAI SDK / curl)
            │  HTTP  /v1/chat/completions
            ▼
   ┌──────────────────────────────────────────┐
   │  vLLM (LlavaForConditionalGeneration)     │   ← export_hf.py 产出的 HF 目录
   │  · paged KV cache / continuous batching    │
   │  · prefix caching / chunked prefill       │
   │  · TP=4 · PP=2 · AWQ W4A16 / FP8          │
   └───────────────┬──────────────────────────┘
                   │  image_url (base64 / http)
                   ▼
        ┌────────────────────────────┐
        │  GPU: CLIP-ViT → Projector │
        │        → LLM (GQA+RoPE)    │
        └────────────────────────────┘
```

### 1.2 路径 B（完全自定义，数值 100% 对齐本项目）

```
   客户端
            │
            ▼
   ┌──────────────────────────────────────────────┐
   │  vLLM  (自定义模型类 MMForCausalLM)             │
   │  ┌──────────────────────────────────────────┐ │
   │  │ MultiModalProcessor (我们实现)             │ │
   │  │  · <image> → 256 × image_token_id          │ │
   │  │  · <audio> →  25 × audio_token_id / chunk  │ │
   │  │  · dummy data / profiling / 缓存            │ │
   │  └───────────────────┬──────────────────────┘ │
   │      MultiModalInputs(pixel_values,            │
   │                      image_features, audio)    │
   │                      ▼                         │
   │  ┌──────────────────────────────────────────┐ │
   │  │ MMForCausalLM                             │ │
   │  │  ViTTower→Resampler→Projector             │ │
   │  │  AudioTower→AudioProjector                │ │
   │  │  → 按掩码原位替换进文本 embedding → LLM     │ │
   │  └──────────────────────────────────────────┘ │
   └──────────────────────────────────────────────┘
```

### 1.3 本项目自带门面（serve.py）

```
            ┌─────────────────────────────────────────────────────────┐
 客户端 ──▶│ serve.py (FastAPI)                                       │
            │  /v1/chat/completions  /generate  /healthz  /metrics     │
            │  · 下载/解码/归一化 image & audio                        │
            │  · 占位符静态展开 (256 / 25)                             │
            │  · 并发闸门 + 超时 + 优雅关闭                             │
            └───────┬───────────────────────────┬─────────────────────┘
                    │ backend=transformers       │ backend=vllm
                    ▼                           ▼
        ┌───────────────────────┐   ┌──────────────────────────────┐
        │ MultimodalLM         │   │ vllm serve 子进程 / 引擎 / 代理 │
        │ .from_pretrained     │   │ (OpenAI 协议转发)              │
        │ + generate (带打点)  │   └──────────────┬───────────────┘
        └───────────┬───────────┘                  ▼
                    └──────────────▶  GPU: 三塔 + LLM
```

```mermaid
flowchart LR
    C[客户端<br/>OpenAI SDK / curl] --> F[serve.py<br/>FastAPI 门面]
    F -->|transformers| M[MultimodalLM<br/>from_pretrained]
    F -->|vllm| V[vLLM<br/>serve / engine]
    V -->|路径A| L[LlavaForConditionalGeneration]
    V -->|路径B| R[MMForCausalLM<br/>自定义注册]
    M --> G[(GPU<br/>ViT + Whisper + LLM)]
    L --> G
    R --> G
    F --> MET[/metrics<br/>TTFT TPOT QPS 显存]
```

---

## 2. 完整可执行命令序列

### 2.0 依赖

```bash
# 部署侧最小依赖
pip install fastapi uvicorn pillow numpy soundfile
# 路径 A / B 还需要 (必须在有 CUDA 的机器上装)
pip install "vllm>=0.6"
# 本项目本地推理后端
pip install torch transformers safetensors
```

### 2.1 导出成 HF Llava 目录

```bash
cd <repo>
python src/deploy/export_hf.py \
    --src  runs/exp1 \
    --dst  out/hf_llava \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --shards 2 \
    --verify
```

产物：

```
out/hf_llava/
├── config.json                    # LlavaConfig (vision_config / text_config / image_token_index ...)
├── generation_config.json
├── preprocessor_config.json       # CLIPImageProcessor
├── processor_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json   # weight_map + total_size
├── mmllm_extra.safetensors        # HF 无对应结构的权重 (音频塔 / Perceiver ...)
├── mmllm_extra.json
└── export_report.json             # 体检报告: blocker / warning / token 数对照
```

> **`export_report.json` 一定要看。** HF `LlavaForConditionalGeneration` 把视觉 token 数硬绑定在
> CLIP patch 网格上，而本项目的 token 数由 `pool_mode` 决定（Perceiver / PixelShuffle），
> 两者不一定相等。脚本会逐条列出 blocker（详见 §6）。

### 2.2 起服务

```bash
# 方式 1: vLLM 直接吃 HF 目录 (路径 A, 最省事)
vllm serve out/hf_llava \
    --served-model-name mmllm \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90

# 方式 2: 本项目 FastAPI 门面 + vLLM 子进程 (推荐: 统一鉴权/限流/指标)
python src/deploy/serve.py \
    --model-dir runs/exp1 \
    --backend vllm --vllm-mode subprocess \
    --host 0.0.0.0 --port 8000 --max-concurrency 16

# 方式 3: 本项目 FastAPI 门面 + 本地 transformers 推理 (功能最全, 含音频, 性能最弱)
python src/deploy/serve.py \
    --model-dir runs/exp1 \
    --backend transformers --device cuda --dtype bfloat16 \
    --host 0.0.0.0 --port 8000
```

### 2.3 发请求（含 base64 图片 / base64 wav）

```bash
# ---- 准备 base64 ----
# Linux / macOS
IMG_B64=$(base64 -w0 demo.jpg)
WAV_B64=$(base64 -w0 demo.wav)
# Windows PowerShell
# $IMG_B64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("demo.jpg"))
# $WAV_B64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("demo.wav"))
```

```bash
# ---- 1) OpenAI 兼容对话: 图 + 音 ----
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"mmllm\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"<image> <audio> 请描述图片内容并复述听到的声音\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMG_B64\"}},
        {\"type\": \"input_audio\", \"input_audio\": {\"data\": \"$WAV_B64\", \"format\": \"wav\"}}
      ]
    }],
    \"max_tokens\": 256,
    \"temperature\": 0.2
  }"
```

```bash
# ---- 2) 裸接口 ----
curl -s http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"<image> <audio> 描述\",
    \"images\": [\"data:image/jpeg;base64,$IMG_B64\"],
    \"audios\": [\"data:audio/wav;base64,$WAV_B64\"],
    \"max_new_tokens\": 128
  }"
```

```bash
# ---- 3) 远程 URL (不用 base64) ----
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mmllm",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<image> 这张图里有什么?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
      ]
    }]
  }'
```

```bash
# ---- 4) 健康检查 / 指标 ----
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/metrics | grep -E "ttft|tpot|gpu_memory"
```

Python 客户端（OpenAI SDK）：

```python
import base64, openai

client = openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
img = base64.b64encode(open("demo.jpg", "rb").read()).decode()
wav = base64.b64encode(open("demo.wav", "rb").read()).decode()

r = client.chat.completions.create(
    model="mmllm",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "<image> <audio> 描述"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        {"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}},
    ]}],
)
print(r.choices[0].message.content)
```

### 2.4 路径 A 的音频怎么走

HF `LlavaForConditionalGeneration` **没有音频塔**，所以 vLLM 直接起 HF 目录时音频不可用。三种可选方案：

| 方案 | 做法 | 保真度 | 工程量 |
| --- | --- | --- | --- |
| **A-1 降级（推荐先落地）** | 服务端用 Whisper encoder 前置，把音频转成文本或"音频描述"再拼进 prompt | 丢失细粒度声学信息 | 小（一个预处理器） |
| **A-2 走本项目门面** | `serve.py --backend transformers`（原生音频塔）；或 `serve.py --backend vllm --vllm-mode proxy` 转发到路径 B 的 vLLM | 完整 | 中 |
| **A-3 路径 B** | 自定义 `MultiModalProcessor` 直接产 `audio` 特征，vLLM 内部承载音频塔 | 完整 | 大 |

A-1 的降级实现要点：Whisper encoder 前置 → log-Mel → 自己的塔算出 25 token/30s 的 embedding →
以"文本前缀"形式注入（把音频 token 的语义用一段自然语言/转写文本替代）。
本项目 `audio_tower` 输出的正是 25 token/chunk，若要接回 vLLM，需走路径 B 的 `MultiModalInputs`。

### 2.5 路径 B 的注册入口

```python
# src/deploy/vllm_multimodal.py 已经实现好, 直接:
from src.deploy.vllm_multimodal import register_model, register_processor

register_model("MMForCausalLM")          # ModelRegistry.register_model
register_processor()                     # MULTIMODAL_REGISTRY.register_processor
```

然后 `vllm serve <你的模型目录> --model-impl MMForCausalLM`（vLLM ≥0.9 用 `--model-impl`，
旧版本用 `--model MMForCausalLM`）。

---

## 3. 性能与延迟调优

### 3.1 视觉 token 数对 TTFT 的影响

单张图的 prefill 增量近似为 `2 × N_LLM_params × N_visual_tokens`（LLM 部分）+ 视觉塔前向（~0.3 TFLOPs/图）。

以 7B（4096 hidden / 32 层）为例，bf16，假设 A100 有效算力 150 TFLOPS：

| 视觉 token | prefill 增量 (LLM) | TTFT 增量 (估) | 每图 KV cache (128 KiB/token) |
| --- | --- | --- | --- |
| 576（HF CLIP 网格） | 8.1 TFLOPs | ≈ 54 ms | 72 MiB |
| **256（本项目默认）** | **3.6 TFLOPs** | **≈ 24 ms** | **32 MiB** |
| 144（PixelShuffle） | 2.0 TFLOPs | ≈ 14 ms | 18 MiB |
| 64（极低延迟） | 0.9 TFLOPs | ≈ 6 ms | 8 MiB |

结论：

* **单图短请求**：`256 → 64` 大约只省 18 ms，收益有限；视觉塔前向（~2 ms）和图片下载解码才是大头。
* **多图 / 长上下文 / 高并发**：视觉 token 直接占 KV cache（7B 下 256 token = 32 MiB/图），
  256 → 64 让可并发序列数提升约 4×，这是真正的收益点。
* **折中**：默认 256；单图问答场景可降到 144；实时语音/字幕场景可用 64。

改法：训练前把 `cfg.vision.out_tokens` 改成目标值（`pool_mode="perceiver"` 时是 latent 数量），
部署侧 `serve.py` 会自动从 checkpoint 的 `config.json` 读取该值，无需改代码。

### 3.2 prefix caching

```bash
vllm serve out/hf_llava --enable-prefix-caching --max-model-len 8192
```

* **能复用什么**：系统提示词 + 固定 few-shot + **相同图片**（多模态 hashing，vLLM 0.9+ 支持
  mm 输入的 prefix hash；0.6~0.8 只对纯文本前缀生效）。
* **效果**：多轮对话里第 2 轮起，系统提示 + 图片的 256 token 的 KV 全部命中，
  TTFT 从"整段 prefill"降到"只 prefill 增量"，典型下降 30%~60%（取决于 prompt 长度占比）。
* **前提**：多模态预处理必须**确定性**（同样的图 → 同样的 pixel_values → 同样的 token 序列）。
  我们的预处理是确定性的（固定 resize + center crop），满足要求。
* **注意**：命中与否取决于 token 级前缀完全一致，图片 URL 变了但内容相同也要靠 mm hashing 才能命中。

### 3.3 chunked prefill

```bash
vllm serve out/hf_llava --enable-chunked-prefill --max-num-batched-tokens 2048
```

* 长 prompt（系统提示 + 多图 + 长历史）一次性 prefill 会阻塞 decode，表现为 **ITL 抖动 / TPOT 突刺**。
* chunked prefill 把 prefill 切成 2048 token 的块与 decode 混批，显著降低 TPOT 方差。
* 代价：prefill 总时长略增（多几次 kernel launch），但尾延迟（p99）明显变好。
* 与 prefix caching 同开效果最好：命中的块直接跳过。

### 3.4 TP / PP 并行度选择

| 模型 | 配置 | 说明 |
| --- | --- | --- |
| 7B | 单卡 A100 80G | 权重 14 GB，剩余空间给 KV cache |
| 7B 高并发 | TP=2 | 跨卡通信少，NVLink 内几乎不损失吞吐 |
| 70B | TP=4, PP=1 | 单机 4×A100 80G；TP≤4 通信在 NVLink 上可接受 |
| 70B 跨机 | TP=4, PP=2 | 2 机 8 卡：机内 TP 走 NVLink，机间 PP 走 IB（只传激活，量小） |

* **TP**：每层 2 次 all-reduce，层数越多通信次数越多。TP 增大会**略微拉长 TTFT**（collective 延迟），
  但显著提高吞吐；经验值 **TP ≤ 4**（单机 NVLink）。
* **PP**：把层切成若干段，显存/权重分摊更好，但存在**流水线气泡**，对 TTFT 不利（首 token 要等最慢段）。
  适合"模型大过单机显存"的场景，不适合追求低延迟。
* **多模态塔放哪**：视觉塔/音频塔在 TP 下会被切分，**建议只对 LLM 做 TP**（vLLM 默认对整个模型做 TP，
  ViT/Whisper 的列切分在数值上没问题，但要注意 `--limit-mm-per-prompt` 下的显存占用）。

### 3.5 GQA 对 KV cache 的影响

本项目 `LLMConfig` 默认 `num_heads=32, num_kv_heads=8`（GQA 4:1）：

```
KV/token = 2 (K,V) × layers × kv_heads × head_dim × bytes
7B :  2 × 32 × 8 × 128 × 2B = 131072 B = 128 KiB/token   (MHA 则是 512 KiB, 4×)
70B:  2 × 80 × 8 × 128 × 2B = 327680 B = 320 KiB/token   (MHA 则是 2.5 MiB, 8×)
```

调 `num_kv_heads` 是"用质量换显存"最直接的手段：`kv_heads=4` 再省一半（质量通常掉 0.5~1 分）。

### 3.6 量化

| 方案 | 显存 | 吞吐 | 质量 | 适用 |
| --- | --- | --- | --- | --- |
| bf16 / fp16 | 1× | 1× | 基线 | 7B 单卡 / 70B 多卡 |
| **AWQ W4A16** | ~0.3× | 1.5~2× | 困惑度 +0.1~0.3；OCR 密集的 VQA 可能掉 1~3 分 | 显存吃紧、批量推理 |
| FP8 W8A8 | ~0.55× | 1.8~2.2× | 几乎无损 | Hopper (H100/H800) 首选 |
| INT8 W8A8 | ~0.55× | 1.5× | 接近无损 | 通用 |
| KV cache FP8 | KV 减半 | 解码更快 | 长上下文掉点明显 | 长上下文场景 |

建议：

* **权重 4bit + 激活 16bit** 是当前性价比最高的组合；**KV cache 保持 fp16**（不要一起压），
  否则长对话的检索类任务（"第 3 页那句话是什么"）会明显掉点。
* **多模态塔不要量化**：ViT / Whisper 参数量小、显存占比低，量化收益低而风险高。
  vLLM 里可以只量化语言塔（`--quantization awq` 作用于整个模型时需注意这一点）。
* 量化后必须做**回归评测**（见 §8 验收清单），尤其是中文 OCR / 图表理解 / 语音转写这三类。

---

## 4. 显存估算

### 4.1 公式

```
总显存 ≈ 权重 + KV cache + 激活 + 框架开销(CUDA context / NCCL / 碎片)

KV cache (bytes) ≈ 2 × layers × kv_heads × head_dim × seq_len × bytes_per_elem × batch
                   └──────────────── 每 token 每序列 ────────────────┘

其中 2 = K 和 V 各一份
```

多模态会额外占用：`(视觉 token 数 + 音频 token 数)` 也在 seq_len 里
（256 视觉 token ≈ 256 个普通 token 的 KV 开销）。

### 4.2 7B 实例（layers=32, kv_heads=8, head_dim=128, bf16）

```
每 token 每序列 = 2 × 32 × 8 × 128 × 2B = 131072 B = 128 KiB
```

| seq_len | batch=1 | batch=8 | batch=32 |
| --- | --- | --- | --- |
| 2,048 | 256 MiB | 2.0 GiB | 8.0 GiB |
| 4,096 | 512 MiB | 4.0 GiB | 16.0 GiB |
| 8,192 | 1.0 GiB | 8.0 GiB | 32.0 GiB |
| 32,768（max） | 4.0 GiB | 32.0 GiB | 128.0 GiB |

权重 bf16 = 13.5 GiB。**A100 80G** 在 `--gpu-memory-utilization 0.90`（可用 72 GiB）下：
72 − 13.5 − 3（激活/框架） ≈ 55 GiB 给 KV → 约 55 GiB / 128 KiB ≈ 440k token 预算
→ seq 4096 时约 **100 路并发**；若每请求带 1 张图（+256 token，占 6.25%）则约 **94 路**。

### 4.3 70B 实例（layers=80, kv_heads=8, head_dim=128, bf16）

```
每 token 每序列 = 2 × 80 × 8 × 128 × 2B = 327680 B = 320 KiB
```

| seq_len | batch=1 | batch=8 | batch=32 |
| --- | --- | --- | --- |
| 4,096 | 1.25 GiB | 10.0 GiB | 40.0 GiB |
| 8,192 | 2.5 GiB | 20.0 GiB | 80.0 GiB |
| 32,768 | 10.0 GiB | 80.0 GiB | 320.0 GiB |

权重 bf16 = 130 GiB → **4×A100 80G (TP=4)**：每卡权重 32.5 GiB，
每卡 KV 预算 = 72 − 32.5 − 3 ≈ 36 GiB → 约 115k token/卡（TP 下 KV 按 head 切分，
总量即上表，每卡是 1/4）→ seq 4096 时每卡约 7 路，总计约 **28 路并发**。

**换成 MHA（kv_heads=64）会变成 2.5 MiB/token**，70B 上表的 KV 全部 ×8 —— 8K 上下文 batch=8
就要 160 GiB，根本放不下。这就是 GQA 的价值。

### 4.4 视觉/音频 token 的 KV 开销

```
每张图  = 256 token → 7B: 32 MiB ; 70B: 80 MiB
每 30s 音频 = 25 token → 7B: 3.2 MiB ; 70B: 8 MiB
```

一个带 1 图 + 1 段 30s 音频的请求，比纯文本多占 7B 下 **35 MiB**、70B 下 **88 MiB** 的 KV。
10 路并发就是 350 MiB / 880 MiB —— 不算大，但**长音频（10 分钟 = 20 chunk = 500 token）** 会变成
7B 下 64 MiB/请求，10 路就是 640 MiB，必须纳入容量规划。

---

## 5. 监控与可观测

`serve.py` 自带一个零依赖的 Prometheus 文本端点 `/metrics`：

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| `mm_requests_total{endpoint,status}` | counter | 请求数 / 错误数（`status=ok|error|timeout`） |
| `mm_request_duration_seconds` | histogram | 端到端耗时 |
| `mm_ttft_seconds` | histogram | 首 token 延迟 |
| `mm_tpot_seconds` | histogram | 每输出 token 延迟 |
| `mm_prompt_tokens_total` / `mm_output_tokens_total` | counter | token 吞吐 |
| `mm_gpu_memory_used_bytes` / `mm_gpu_memory_total_bytes` | gauge | GPU 显存（nvidia-smi 采集） |
| `mm_gpu_utilization_percent` | gauge | GPU 利用率 |
| `mm_inflight_requests` | gauge | 在途请求数（= 并发闸门水位） |

建议的告警阈值（7B，A100 单卡）：

| 指标 | 告警条件 | 含义 |
| --- | --- | --- |
| TTFT p95 | > 1.5 s | prefill 被打满 / prefix cache 失效 |
| TPOT p95 | > 60 ms | decode 被长 prefill 抢占 → 开 chunked prefill |
| 错误率 | > 1%（5 分钟） | OOM / 超时 / 后端崩溃 |
| GPU 利用率 | < 40% 且 QPS > 0 | 瓶颈在 CPU（图像解码 / 音频重采样） |
| 显存水位 | > 92% | 接近 OOM，降 batch 或减 max_model_len |
| 在途请求 | 持续 = max_concurrency | 闸门打满，需要扩容或限流 |

vLLM 自身的 `/metrics`（`vllm:num_requests_running`、`vllm:gpu_cache_usage_perc`、
`vllm:time_to_first_token_seconds`）应与本服务的指标一起采集，**以 vLLM 的为准做容量判断**，
本服务的指标用于对外 SLA 统计。

---

## 6. 路径 A / 路径 B 取舍（重要）

### 6.1 结论先说

| 维度 | 路径 A（HF Llava + `vllm serve`） | 路径 B（自定义 vLLM 模型） |
| --- | --- | --- |
| 工程量 | **小**（导出脚本 + 起服务） | 大（processor + 模型类 + 权重加载 + 版本适配） |
| 数值一致性 | **受限于视觉 token 契约**（见下） | 100% 一致 |
| 音频 | HF Llava 无音频塔 → 需降级 | 完整支持 |
| 生态复用 | **完全复用**（量化、LoRA、张量并行、投机解码） | 需自己适配 |
| 升级维护 | 跟 vLLM 走 | **要跟 vLLM 版本走**（0.6/0.7/0.8/0.9 API 差异大） |
| 适用 | 文本为主的场景 / 视觉塔已是 CLIP 网格对齐 | **本项目默认 checkpoint**（Perceiver + 音频塔） |

**推荐做法**：先用路径 A 打通链路（把服务、监控、鉴权、限流这些"跟模型无关"的东西先落地），
同时用路径 B 保证最终数值正确；路径 A 里的"图像 token 数不一致"以 `export_report.json` 为准。

### 6.2 为什么路径 A 会有"数值不一致"

HF `LlavaForConditionalGeneration` 的视觉 token 数 = **CLIP patch 网格**（`(image_size/patch_size)²`，
默认 336/14 → 576），且 `multi_modal_projector` 是固定的 2 层 MLP。而本项目：

* `pool_mode="perceiver"` → 用 Perceiver Resampler 压到 `out_tokens`（默认 256）→ **HF 无对应结构**；
* `pool_mode="mlp"` → 投影器里写死了 `pixel_shuffle=True` → token 数 = 网格/4 → **HF 无对应结构**；
* `projector_hidden_mult` 默认 4 → HF 的投影器是固定 2 层 MLP → **形状不兼容**；
* 投影器末端 LayerNorm → HF 没有这一步；
* CLIP 多一个 CLS token（位置编码 577 vs 我们的 576）→ 导出时补零，CLS 仍参与第 0 层注意力。

`export_hf.py` 会把这些逐条列在 `export_report.json` 里（R1~R9），**不会静默导出错误权重**。

### 6.3 怎样让路径 A 变得"数值正确"

需要训练侧的配置满足（`modeling.py` 的当前限制见"当前限制"）：

1. `vision.pool_mode = "mlp"`；
2. 投影器的网格重采样后 token 数 == `cfg.vision.out_tokens`；
   （`mlp` 模式已支持任意目标值：把 patch 网格 `(gh, gw)` 用 2D 自适应平均池化
   重采样到 `(th, tw)`，`th*tw == out_tokens`。例如 24×24 → 16×16 = 256，正好等于默认契约；
   14×14 → 8×8 = 64）
3. `projector_hidden_mult = 1`；
4. `projector_norm_out = False`；
5. `llm.vocab_size >= 32004`（`image_token_id=32001`、`audio_token_id=32002`、`video_token_id=32003`）。

> **当前限制**：
> * HF `LlavaForConditionalGeneration` 的 `multi_modal_projector` 是固定的
>   `vision_hidden → text_hidden → text_hidden` 两层 MLP，没有中间维度倍数的概念，
>   因此要求 `projector_hidden_mult = 1`（否则结构对不上）。
> * HF 侧没有"投影输出再做一次 LayerNorm"这一步，要求 `projector_norm_out = False`
>   （否则视觉特征尺度与本项目不同）。
> * HF Llava **只支持图像**，没有音频塔的位置。
> * 因此**没有一种现有配置能被 HF Llava 1:1 表达**。要打通"数值正确的路径 A"，
>   需要训练时就按上述约束配置（或用 `export_hf.py --with-remote-code` 携带本项目代码）。
>   在此之前：
>   * 走**路径 B** 保证数值正确；或
>   * 走 **路径 A + Whisper 前置降级**（§2.4），接受视觉链路的微小差异；或
>   * 用 `export_hf.py --with-remote-code` 产出"HF 目录 + 本项目 modeling 代码"，
>   以 `trust_remote_code=True` 加载（数值 100% 一致，但 vLLM 的 Llava 加载器不认它）。

### 6.4 路径 B 需要在目标 vLLM 上核对的东西

`vllm_multimodal.py` 已按 0.6 / 0.7 / 0.8 / 0.9+ 做了 try/except 适配层，但以下签名**必须在目标版本上核对**：

| 关注点 | 0.6 / 0.7 | 0.8 | 0.9+ |
| --- | --- | --- | --- |
| 注册 processor | `register_processor(processor, model_cls)` | `register_processor(processor)` | `register_processor()`（由 `processor.info` 反查） |
| token 上限接口 | `get_max_image_tokens(w, h)` | `get_max_tokens_per_item(modality)` | 同左（`ProcessingInfo`） |
| `apply()` 返回 | `PromptUpdates` | `PromptUpdates` | `PromptUpdates` |
| 多模态输入 | `MultiModalInputs(mm_placeholders=..., mm_kwargs=...)` | 同左 | 同左（字段可能变 dataclass） |
| 模型侧接口 | `_process_image_input` | `_process_image_input` | `get_multimodal_embeddings` / `get_input_embeddings` |

---

## 7. 生产化清单

- [ ] **限流**：按 API Key / IP 做令牌桶（网关层，如 Envoy/Nginx limit_req 或自研中间件）；
      `serve.py` 已有 `MM_MAX_CONCURRENCY` 并发闸门，但它只保护**单副本**，全局限流要在网关做。
- [ ] **熔断**：后端连续 N 次超时/5xx 就摘除该副本（K8s readiness probe + 熔断器）；
      `serve.py` 已对超时返回 504 并计数 `mm_errors_total{kind="timeout"}`。
- [ ] **超时分级**：连接超时 5s、首字节超时 10s、整体超时 = `max(60, 预估生成时长 × 3)`；
      客户端 SDK 也必须设置超时，否则请求会在网关堆积。
- [ ] **鉴权**：`MM_API_KEY`（本服务内置 Bearer 校验）；生产建议网关做 JWT/OAuth2，
      服务只信任内网。
- [ ] **多副本**：vLLM 每副本独占 GPU（或一组），前面挂 LVS/Envoy；
      **不要**在多副本间共享 prefix cache（命中率会低），只在副本内共享。
- [ ] **灰度**：按流量比例 / 用户分桶路由到"新模型目录" vs "旧模型目录"；
      用 `export_report.json` 里的 token 数对照，防止新旧模型的占位符契约不一致导致线上炸掉。
- [ ] **健康检查**：`/healthz` 供 K8s liveness/readiness；`/metrics` 接 Prometheus + Alertmanager。
- [ ] **日志**：结构化 JSON（`request_id / model / prompt_tokens / completion_tokens / ttft / tpot`），
      **不要打印图片 base64 和音频内容**。
- [ ] **数据安全**：图片/音频可能是用户隐私；落盘缓存（若启用）必须加密 + TTL；
      日志里只留哈希。
- [ ] **容量水位**：GPU 显存 > 85% 告警 + 拒绝新请求（`--max-num-seqs` 是硬闸）。
- [ ] **回滚**：保留上一个 HF 目录 + 上一个镜像 tag，灰度失败时 5 分钟内可切回。

---

## 8. 本机无法验证的部分 & GPU 机器验收清单

### 8.1 本机（无 CUDA、无 vLLM）**已经**验证过的

* `python -m py_compile` 对三个 py 文件全部通过；
* `python src/deploy/vllm_multimodal.py` 自检通过（15 项检查，缺 vLLM 时不失败，只提示安装命令）；
* `export_hf.py` 端到端跑通（tiny checkpoint → 2 分片 + index + extra + report，`--verify` PASS，
  370 + 196 = 566 个张量**零丢失**）；
* 占位符静态展开：`<image>` → 256 个 `image_token_id`、`<audio>` → 25/chunk 个 `audio_token_id`；
* 图像预处理输出 `(3, 336, 336)`、音频 log-Mel 输出 `(128, 3000)`；
* `serve.py` 的消息解析、base64/data URL/URL 加载、transformers 后端 CPU 端到端生成。

### 8.2 本机**无法**验证的（必须在有 GPU 的机器上做）

| 项目 | 为什么本机不行 | 验收方法 |
| --- | --- | --- |
| vLLM 路径 A 加载 HF 目录 | 没装 vLLM，也没 GPU | §8.3 步骤 1~3 |
| vLLM 路径 B 的 `MultiModalProcessor` / `MMForCausalLM` | 同上 | §8.3 步骤 4 |
| TTFT / TPOT 实测数值、吞吐、GPU 利用率 | 没有 GPU | §8.3 步骤 5 |
| 量化（AWQ / FP8）后的质量与显存 | 没装量化工具 | §8.3 步骤 6 |
| prefix caching 命中率、chunked prefill 效果 | 需要真实负载 | §8.3 步骤 5 |
| TP / PP 扩展性 | 单机单卡 | §8.3 步骤 7 |
| 音频在 vLLM 侧的表现 | 依赖具体版本的多模态音频支持 | §8.3 步骤 4 |

### 8.3 GPU 机器验收清单

```bash
# ---- 步骤 0: 环境 ----
nvidia-smi                      # 确认 GPU 与驱动
python -c "import vllm; print(vllm.__version__)"   # 记录版本, 决定用哪套 API
pip install "vllm>=0.6" fastapi uvicorn soundfile

# ---- 步骤 1: 导出并查看体检报告 ----
python src/deploy/export_hf.py --src runs/exp1 --dst out/hf_llava --shards 2 --verify
cat out/hf_llava/export_report.json | python -m json.tool | head -60
#   重点看: path_a_ready / image_tokens.project vs image_tokens.hf_llava / issues[]

# ---- 步骤 2: 起 vLLM (路径 A) ----
vllm serve out/hf_llava --served-model-name mmllm --port 8000 \
    --max-model-len 8192 --enable-prefix-caching --enable-chunked-prefill \
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90

# ---- 步骤 3: 数值对齐 (最关键的一步) ----
#   同一个 prompt + 同一张图, 分别用:
#     a) 本项目 transformers 后端 (src/deploy/serve.py --backend transformers)
#     b) vLLM 路径 A
#   比较两边的输出 logits / 贪心解码结果。
#   期望: 若 export_report.json 显示 image_tokens.project != image_tokens.hf_llava,
#         则 (b) 的视觉链路与 (a) 不等价, 必须换路径 B 或按 §6.3 调整训练配置。

# ---- 步骤 4: 路径 B ----
python src/deploy/vllm_multimodal.py --register          # 注册 processor + 模型类
vllm serve runs/exp1 --model-impl MMForCausalLM --port 8001 \
    --max-model-len 8192 --enable-prefix-caching
#   对照 §6.4 的表格核对接口名; 如报错, 按目标版本调整 vllm_multimodal.py 的兼容层

# ---- 步骤 5: 压测 + 指标 ----
#   用真实流量 (建议 1/10/50 并发) 打 10 分钟, 观察:
#   - /metrics 的 ttft / tpot p95
#   - vllm:num_requests_running / vllm:gpu_cache_usage_perc
#   - 打开/关闭 --enable-prefix-caching 的 TTFT 差异 (同图多轮)
python - <<'PY'
import base64, time, requests
img = base64.b64encode(open("demo.jpg","rb").read()).decode()
body = {"model":"mmllm","messages":[{"role":"user","content":[
    {"type":"text","text":"<image> 描述"},
    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img}"}}]}],"max_tokens":128}
t0=time.time(); r=requests.post("http://127.0.0.1:8000/v1/chat/completions", json=body, timeout=120)
print("status", r.status_code, "elapsed %.2fs" % (time.time()-t0))
print(r.json()["choices"][0]["message"]["content"][:200])
PY

# ---- 步骤 6: 量化回归 ----
#   AWQ:  vllm serve out/hf_llava --quantization awq
#   FP8:  vllm serve out/hf_llava --quantization fp8
#   回归集: 中文 OCR / 图表 QA / 语音转写 各 200 条, 对比 bf16 基线

# ---- 步骤 7: 并行度 ----
#   TP=2/4, PP=1/2 分别起, 记录 TTFT/吞吐/显存, 选拐点
```

---

## 9. 常见问题

**Q：`vllm serve` 报 "image token count mismatch" / 形状不匹配？**
A：占位符数量与模型期望不一致。检查 `export_report.json` 的 `image_tokens` 段：
`project` 是本项目的契约（256），`hf_llava` 是 HF 侧期望的数量（CLIP 网格）。
两者不等说明该 checkpoint 不能走路径 A（§6.2）。

**Q：服务返回 503？**
A：后端还没就绪。`serve.py --backend vllm --vllm-mode subprocess` 会等待 `/health` 返回 200，
若超时会在日志里打印 `vLLM 子进程提前退出`；多半是显存不够（`--gpu-memory-utilization` 太高）或
`--max-model-len` 超过了模型能力。

**Q：音频请求报 "音频塔不在 HF Llava 内"？**
A：路径 A 的 HF 目录没有音频塔。改用 `--backend transformers`（原生音频塔）或路径 B（§2.4）。

**Q：怎么确认占位符展开对了？**
A：调用 `/generate`，响应里的 `multimodal` 字段会给出 `image_tokens_each` / `audio_tokens_each_chunk`，
应分别是 256 和 25（或你的 checkpoint 里的值）。
