"""
端到端验收脚本 (本地 CPU)
=========================
把整条链路在本机跑通一次:
    1. 造一个 tiny checkpoint (config.json + model.safetensors)
    2. export_hf.py 导出成 HF Llava 目录, 并 --verify 校验
    3. serve.py 的请求侧管线: 读 config -> 图像/音频预处理 -> 占位符静态展开
    4. 用真 checkpoint 跑一次多模态 generate (图像 + 音频同时输入)

这是部署模块在无 GPU、无 vLLM 环境下能做的最大范围验证。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mmllm.config import preset                                    # noqa: E402
from mmllm.modeling import MultimodalLM                            # noqa: E402

WORK = tempfile.mkdtemp(prefix="mmllm_e2e_")
CKPT = os.path.join(WORK, "ckpt")
EXPORTED = os.path.join(WORK, "hf_out")


def step(n, total, title):
    print(f"\n{'='*68}\n[{n}/{total}] {title}\n{'='*68}")


def main():
    total = 4

    # ---------- 1. 造 checkpoint ----------
    step(1, total, "构造 tiny checkpoint")
    cfg = preset("tiny")
    cfg.llm.vocab_size = 33000
    cfg.image_token_id = 32001
    cfg.audio_token_id = 32002
    model = MultimodalLM(cfg)
    model.save_pretrained(CKPT)
    print(f"checkpoint -> {CKPT}")
    for f in sorted(os.listdir(CKPT)):
        print(f"  {f}  {os.path.getsize(os.path.join(CKPT, f)) / 1024:.1f} KB")

    # ---------- 2. 导出 HF ----------
    step(2, total, "export_hf.py 导出 + --verify")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "deploy", "export_hf.py"),
         "--src", CKPT, "--dst", EXPORTED, "--verify", "--force"],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
    assert r.returncode == 0, "导出失败"
    idx = os.path.join(EXPORTED, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"index.json 权重条目: {len(meta.get('weight_map', {}))}")
    print("HF 目录内容:", sorted(os.listdir(EXPORTED)))

    # ---------- 3. serve.py 请求侧管线 ----------
    step(3, total, "serve.py 请求侧管线 (config 读取 + 预处理 + 占位符展开)")
    sys.path.insert(0, os.path.join(ROOT, "deploy"))
    import serve as S

    print("serve.py 顶层定义:", [n for n in dir(S) if not n.startswith("_")][:20])

    # 用真 checkpoint 的 config, 而不是默认 preset —— 这是子代理修过的一个 bug
    cfg_path = os.path.join(EXPORTED, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            hf_cfg = json.load(f)
        print("导出 config 的 image_token_index:", hf_cfg.get("image_token_index"))

    # 造一张图和一段 wav
    from PIL import Image
    img_path = os.path.join(WORK, "a.png")
    Image.fromarray(
        (np.random.rand(320, 320, 3) * 255).astype(np.uint8)
    ).save(img_path)

    wav_path = os.path.join(WORK, "a.wav")
    import wave
    sr = 16000
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        w.writeframes((np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16).tobytes())

    # 调 serve 里的预处理函数(名字按实际实现自适应)
    prep_img = getattr(S, "preprocess_image", None)
    prep_aud = getattr(S, "preprocess_audio_chunk", None)
    chunker = getattr(S, "chunk_audio", None)
    resampler = getattr(S, "resample_to_16k", None)
    print("serve.py 暴露的预处理函数:",
          [n for n in ("preprocess_image", "preprocess_audio_chunk",
                       "chunk_audio", "resample_to_16k", "expand_prompt_text",
                       "expand_prompt_tokens") if hasattr(S, n)])

    # 造一张图和一段 wav
    from PIL import Image
    img_path = os.path.join(WORK, "a.png")
    Image.fromarray(
        (np.random.rand(320, 320, 3) * 255).astype(np.uint8)
    ).save(img_path)

    wav_path = os.path.join(WORK, "a.wav")
    import wave
    sr = 16000
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        w.writeframes((np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16).tobytes())

    if prep_img:
        from PIL import Image as _I
        pv = prep_img(_I.open(img_path), cfg.vision.image_size)
        print("图像预处理输出:", tuple(pv.shape), pv.dtype)
        assert tuple(pv.shape) == (3, cfg.vision.image_size, cfg.vision.image_size)

    if prep_aud and chunker:
        import wave as _w
        with _w.open(wav_path, "rb") as w:
            raw = w.readframes(w.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sr_in = w.getframerate()
        if resampler:
            audio = resampler(audio, sr_in, cfg.audio.sample_rate)
        chunks = chunker(audio, cfg)
        feats = [prep_aud(c, cfg) for c in chunks]
        print(f"音频 3.0s -> {len(chunks)} chunk, 每块 shape={tuple(feats[0].shape)}")
        assert tuple(feats[0].shape) == (cfg.audio.n_mels, 3000)

    # 占位符展开
    exp = getattr(S, "expand_prompt_text", None)
    if exp:
        out = exp("看图 <image> 听音 <audio> 回答", cfg, n_images=1, n_audio_chunks=1)
        n_img = str(out).count("<image>")
        n_aud = str(out).count("<audio>")
        print(f"占位符展开: <image>×{n_img} (期望 {cfg.vision.out_tokens}), "
              f"<audio>×{n_aud} (期望 {cfg.audio.out_tokens_per_chunk})")
        assert n_img == cfg.vision.out_tokens
        assert n_aud == cfg.audio.out_tokens_per_chunk

    # ---------- 4. 端到端生成 ----------
    step(4, total, "真 checkpoint 多模态生成 (图像 + 音频同时输入)")
    m = MultimodalLM.from_pretrained(CKPT)
    m.eval()

    NV = cfg.vision.out_tokens
    NA = cfg.audio.out_tokens_per_chunk
    row = ([100] * 4 + [cfg.image_token_id] * NV + [101] * 4
           + [cfg.audio_token_id] * NA + [102] * 4)
    ids = torch.tensor([row], dtype=torch.long)
    px = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size)
    af = torch.randn(1, cfg.audio.n_mels, 3000)

    with torch.no_grad():
        out = m.generate(ids, pixel_values=px, input_features=af,
                         max_new_tokens=8, temperature=0.0)
    print("生成 token:", out.tolist())
    assert out.shape[1] == 8

    print(f"\n{'='*68}\n端到端验收全部通过 ✓  (工作目录 {WORK})\n{'='*68}")


if __name__ == "__main__":
    main()
