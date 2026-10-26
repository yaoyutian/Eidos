"""
回归测试集
==========
覆盖多模态方案里最容易"悄悄跑偏"的三条不变量:

  T1  形状自检      —— 三塔前向形状、占位符与特征数量对齐、三塔梯度回流
  T2  变长批次对齐  —— 同一 batch 内样本图片/音频数量不同, 特征不能错位
  T3  增量解码一致  —— 带 KV cache 的 generate 必须等于逐 token 全量前向

T2 与 T3 分别对应两个真实存在过的 bug:
  T2: _scatter 曾假设"模态特征按 batch 维一一对应", 导致跨样本错位
  T3: RoPE 曾施加在 KV 拼接之后, 缓存 key 被错误位置重复旋转

运行:
    cd src && python -m pytest tests -q
或  cd src && python tests/test_multimodal.py     # 无 pytest 时直接跑
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmllm.config import preset                     # noqa: E402
from mmllm.modeling import MultimodalLM, _selfcheck  # noqa: E402


def _tiny_cfg():
    cfg = preset("tiny")
    cfg.llm.vocab_size = 33000
    cfg.image_token_id = 32001
    cfg.audio_token_id = 32002
    return cfg


def test_t1_shape_and_gradients():
    """T1: 形状对齐 + 三塔梯度回流。"""
    model = _selfcheck()
    assert model is not None


def test_t2_variable_item_counts_in_batch():
    """
    T2: 同一 batch 内, 样本 A 有 2 张图, 样本 B 有 1 张图, 样本 C 没有图。
    验证特征按"样本内出现顺序"落到正确的占位符位置, 而不是按 batch 维错位。
    """
    torch.manual_seed(7)
    cfg = _tiny_cfg()
    NV = cfg.vision.out_tokens
    model = MultimodalLM(cfg)
    model.eval()

    # 三张"可区分"的图片: 让每张图被编码成明显不同的特征
    n_items = 3
    px = torch.randn(n_items, 3, cfg.vision.image_size, cfg.vision.image_size)
    with torch.no_grad():
        vis = model.encode_images(px)                     # (3, NV, D)
    # 给每张图一个可识别的指纹
    for i in range(n_items):
        vis[i] += (i + 1) * 3.0

    # 组装: sample0 用图0+图1, sample1 用图2, sample2 无图
    per_sample_imgs = [[0, 1], [2], []]
    rows = []
    for imgs in per_sample_imgs:
        row = [100] * 3
        for _ in imgs:
            row += [cfg.image_token_id] * NV
        row += [101] * 3
        rows.append(row)
    L = max(len(r) for r in rows)
    ids = torch.zeros(3, L, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)

    embeds = model.language_model.embed_tokens(ids)
    mask = ids == cfg.image_token_id
    scattered = model._scatter(embeds, mask, vis)

    for si, imgs in enumerate(per_sample_imgs):
        pos = [j for j in range(L) if ids[si, j] == cfg.image_token_id]
        assert len(pos) == len(imgs) * NV
        for k, img_idx in enumerate(imgs):
            seg = scattered[si, pos[k * NV:(k + 1) * NV], :]
            ref = vis[img_idx]
            # 位置 k 的特征必须来自第 k 张图, 而不是别的样本的图
            assert torch.allclose(seg, ref, atol=1e-5), \
                f"样本 {si} 的第 {k} 张图特征错位"
    print("T2 变长批次特征对齐 ✓")


def test_t3_cache_vs_nocache_consistency():
    """T3: 带 KV cache 的 generate == 逐 token 全量前向。"""
    torch.manual_seed(11)
    cfg = _tiny_cfg()
    model = MultimodalLM(cfg)
    model.eval()

    prompt = torch.tensor([[100, 101, 102]], dtype=torch.long)
    with_cache = model.generate(prompt, max_new_tokens=10, temperature=0.0)

    manual = []
    cur = prompt.clone()
    with torch.no_grad():
        for _ in range(10):
            lg, _ = model.language_model(input_ids=cur)
            nxt = lg[:, -1, :].argmax(dim=-1, keepdim=True)
            manual.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
    manual = torch.cat(manual, dim=1)

    assert torch.equal(with_cache, manual), (
        f"KV cache 路径与全量前向不一致:\n"
        f"  cache    = {with_cache.tolist()}\n"
        f"  no-cache = {manual.tolist()}"
    )
    print("T3 增量解码一致性 ✓")


def test_t4_long_conversation_keeps_the_answer():
    """
    T4: 超长多轮对话被裁剪时, **最终回答必须完整保留**。
    曾经的 bug: 朴素地从头截断, 把最后的 assistant 回答整段切掉,
    导致该样本所有 label 都变成 -100 —— 监督信号全丢, 白跑一个 batch。
    """
    from mmllm.data import MMDataset
    from mmllm.tokenizer_ext import IMAGE_TOKEN

    cfg = _tiny_cfg()
    NV = cfg.vision.out_tokens

    class _FakeTok:
        bos_token_id = 1
        pad_token_id = 0
        eos_token = "<|im_end|>"

        def __call__(self, text, add_special_tokens=False):
            ids = []
            i = 0
            while i < len(text):
                if text.startswith(IMAGE_TOKEN, i):
                    ids.append(cfg.image_token_id)
                    i += len(IMAGE_TOKEN)
                else:
                    ids.append(ord(text[i]) % 30000 + 100)
                    i += 1
            return {"input_ids": ids}

    # 5 轮对话, 每轮都很长, 预算只够 2 轮
    convs = []
    for r in range(5):
        convs.append({"role": "user", "content": f"第{r}轮问题" + "问" * 300})
        convs.append({"role": "assistant", "content": f"第{r}轮回答" + "答" * 300})

    ds = MMDataset([{"conversations": convs, "images": [], "audios": []}],
                   cfg, _FakeTok(), max_text_len=600)
    sample = ds[0]

    assert len(sample["input_ids"]) <= 600, "超出预算"
    n_sup = sum(1 for t in sample["labels"] if t != cfg.label_pad_id)
    assert n_sup > 0, "监督信号被完全截断 —— 回答丢了"

    # 被保留的必须是**最后**一轮的回答: 其最后一个 token 必须被监督
    assert sample["labels"][-1] != cfg.label_pad_id, "最终回答的结尾未被监督"
    print(f"T4 超长对话保留回答 ✓ (保留 {n_sup} 个监督 token)")


def test_t5_pool_modes_honor_out_tokens():
    """
    T5: 两种 pool_mode 下, 视觉 token 数都必须**精确等于** cfg.vision.out_tokens。

    这条不变量是占位符契约成立的前提 —— tokenizer 按 out_tokens 展开 <image>,
    模型必须吐出同样数量的视觉 token, 否则特征会错位。
    曾经的 bug: mlp 模式把 pixel_shuffle 写死, 实际输出 num_patches/4,
    与配置声明的 out_tokens 不符(576/4=144 vs 声明的 256)。
    """
    from mmllm.modeling import MultimodalLM as _M

    for mode in ("perceiver", "mlp"):
        for out_tokens in (64, 100, 256):
            cfg = _tiny_cfg()
            cfg.vision.pool_mode = mode
            cfg.vision.out_tokens = out_tokens
            model = _M(cfg)
            model.eval()
            px = torch.randn(2, 3, cfg.vision.image_size, cfg.vision.image_size)
            with torch.no_grad():
                vis = model.encode_images(px)
            assert vis.shape[1] == out_tokens, (
                f"{mode} 模式 out_tokens={out_tokens} 时实际输出 {vis.shape[1]} 个 token"
            )
    print("T5 两种 pool_mode 的 token 数契约 ✓")


if __name__ == "__main__":
    failures = []
    for name, fn in [
        ("T1 形状与梯度", test_t1_shape_and_gradients),
        ("T2 变长批次对齐", test_t2_variable_item_counts_in_batch),
        ("T3 增量解码一致性", test_t3_cache_vs_nocache_consistency),
        ("T4 超长对话保回答", test_t4_long_conversation_keeps_the_answer),
        ("T5 token 数契约", test_t5_pool_modes_honor_out_tokens),
    ]:
        try:
            fn()
            print(f"[PASS] {name}\n")
        except Exception as e:          # noqa: BLE001
            failures.append((name, e))
            print(f"[FAIL] {name}: {e}\n")
    if failures:
        sys.exit(1)
    print("全部回归测试通过 ✓")
