"""Where do the matmuls' 18 milliseconds actually go?

The kernel breakdown said the matmuls are 18000 us of 22831 us - 79 per
cent of the GPU time - while the weights need only 4.6 ms of bandwidth.
So the matmuls are running at about a quarter of the bandwidth a
standalone GEMV reached.

Two explanations:

  * the shapes. The standalone test used (8 x 2048) @ (2048 x 2048).
    The model's k_proj and v_proj are (8 x 2048) @ (2048 x 1024) - half
    the output - and the attention then expands the 8 key/value heads to
    16 so the bmm runs on twice the data. A different shape can pick a
    different cuBLAS kernel.

  * the expansion. Grouped-query attention means the key and value heads
    are materialised at 16 heads before the attention can use them, and
    that is a cat and a copy of the whole KV cache, four times a layer.

This times each of the model's seven projections on its own, against the
bandwidth floor for the bytes it has to read, so the slow ones can be
named. Then it times the same projections when the key/value ones are
doing the expansion the attention path requires.

    python scripts/check_gqa_cost.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402

from la import hf_cache                          # noqa: E402
from la.inference import load_model              # noqa: E402

BW = 756e9   # bytes per second, the 4070 SUPER's bandwidth


def bench(fn, n=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main() -> int:
    print(f"cache: {hf_cache.describe()}")
    print()

    tok, model = load_model("Qwen/Qwen3-1.7B-Base", "runs/vslm")
    dev = model.device
    cfg = model.config
    H = cfg.hidden_size
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    d = H // nq
    L = cfg.num_hidden_layers
    print(f"model: {L} layers, hidden {H}")
    print(f"  query heads {nq}, key/value heads {nkv}, head dim {d}")
    print(f"  => {nq // nkv} query heads per key/value head")

    bs = 8
    x = torch.randn(bs, H, device=dev, dtype=model.dtype)
    layer = model.model.layers[0]
    attn = layer.self_attn

    # --- each projection, against its own bandwidth floor --------------
    print()
    print("=== each projection, on its own ===")
    print(f"  {'projection':<14} {'shape':<24} {'ms':>7} {'MB':>7} "
          f"{'GB/s':>7} {'floor':>8} {'x floor':>8}")
    total_ms = 0.0
    total_floor = 0.0
    for name, mod in (("q_proj", attn.q_proj), ("k_proj", attn.k_proj),
                      ("v_proj", attn.v_proj), ("o_proj", attn.o_proj),
                      ("gate_proj", layer.mlp.gate_proj),
                      ("up_proj", layer.mlp.up_proj),
                      ("down_proj", layer.mlp.down_proj)):
        w = mod.weight
        # each projection has its own input width: the attention ones take
        # the hidden size, down_proj takes the expanded intermediate size
        in_features = w.shape[1]
        xi = torch.randn(bs, in_features, device=dev, dtype=model.dtype)
        # the weight is stored (out_features, in_features); the matmul is
        # x @ w.T, and torch.nn.functional.linear does the same thing
        t = bench(lambda: torch.nn.functional.linear(xi, w))
        mb = w.numel() * 2 / 1e6
        # the floor: bytes / bandwidth, in milliseconds
        floor = (mb * 1e6) / (BW / 1e9) / 1e6
        total_ms += t
        total_floor += floor
        shape = f"({bs}x{in_features})@({in_features}x{w.shape[0]})"
        gbs = (mb * 1e6) / (t / 1000) / 1e9
        print(f"  {name:<14} {shape:<24} {t:7.4f} {mb:7.1f} "
              f"{gbs:7.0f} {floor:8.4f} {t/floor:7.1f}x")
    print(f"  {'TOTAL per layer':<14} {'':<24} {total_ms:7.4f} "
          f"{'':>7} {'':>7} {total_floor:8.4f} {total_ms/total_floor:7.1f}x")
    print(f"  x {L} layers = {total_ms*L:.2f} ms of the step")
    print(f"  the bandwidth floor for all of it = {total_floor*L:.2f} ms")

    # --- the grouped-query expansion ------------------------------------
    print()
    print("=== what the grouped-query expansion costs ===")
    print("  the attention needs 16 key/value heads; the projections only")
    print("  make 8. The expansion copies the whole KV cache every step.")

    ctx = 200
    k8 = torch.randn(bs, nkv, ctx, d, device=dev, dtype=model.dtype)
    v8 = torch.randn(bs, nkv, ctx, d, device=dev, dtype=model.dtype)
    q16 = torch.randn(bs, nq, 1, d, device=dev, dtype=model.dtype)

    def expand():
        # what transformers' repeat_kv does: expand then reshape, which
        # materialises a copy of the whole kv cache
        k = k8[:, :, None, :, :].expand(bs, nkv, nq // nkv, ctx, d)
        v = v8[:, :, None, :, :].expand(bs, nkv, nq // nkv, ctx, d)
        return (k.reshape(bs, nq, ctx, d),
                v.reshape(bs, nq, ctx, d))

    t_exp = bench(expand, n=200)
    mb = 2 * bs * nq * ctx * d * 2 / 1e6
    print(f"  expanding k and v: {t_exp:6.4f} ms  ({mb:.1f} MB written)")
    print(f"  x {L} layers = {t_exp*L:.3f} ms")
    print(f"  that is {100*t_exp*L/32.7:.1f} per cent of the step")

    # --- the attention, three ways --------------------------------------
    print()
    print("=== the attention itself, three ways ===")
    print("  transformers picks between three paths in sdpa_attention_forward:")
    print("    enable_gqa=True   when use_gqa_in_sdpa() is true - no expansion")
    print("    pack_gqa          when q_length == 1 - a reshape, no copy")
    print("    repeat_kv         otherwise - a full materialised copy")
    print("  use_gqa_in_sdpa needs attention_mask is None, so a padded batch")
    print("  never takes that path. The decode path is pack_gqa.")

    def attn_expanded():
        k, v = expand()
        torch.nn.functional.scaled_dot_product_attention(q16, k, v)

    def attn_pack_gqa():
        # what transformers does at q_length == 1: reshape the query so
        # the attention runs on the key/value heads directly. No transpose:
        # the result is (batch, num_kv_heads, n_rep, dim), and sdpa reads
        # dim 2 as the sequence, so the two query rows of a group become
        # two positions. No copy of the kv heads is made.
        q = q16.reshape(bs, nkv, nq // nkv, d)
        torch.nn.functional.scaled_dot_product_attention(q, k8, v8)

    def attn_enable_gqa():
        torch.nn.functional.scaled_dot_product_attention(
            q16, k8, v8, enable_gqa=True)

    for name, fn in (("repeat_kv (prefill)", attn_expanded),
                     ("pack_gqa (decode)", attn_pack_gqa),
                     ("enable_gqa=True", attn_enable_gqa)):
        try:
            t = bench(fn, n=100)
            print(f"  {name:<22} {t:6.4f} ms   x{L} = {t*L:.3f} ms")
        except Exception as exc:
            print(f"  {name:<22} failed: {str(exc)[:70]}")

    # --- the model's own attention path ---------------------------------
    print()
    print("=== the model's own attention, through transformers ===")
    print("  this is the code path that actually runs, with the model's")
    print("  real query, key and value shapes.")
    try:
        from transformers.integrations.sdpa_attention import (
            sdpa_attention_forward)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        iface = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        print(f"  the registered sdpa interface: {iface}")

        # the module the interface needs
        attn_mod = layer.self_attn
        attn_mod.num_key_value_groups = nq // nkv

        q = torch.randn(bs, nq, 1, d, device=dev, dtype=model.dtype)
        k = torch.randn(bs, nkv, ctx, d, device=dev, dtype=model.dtype)
        v = torch.randn(bs, nkv, ctx, d, device=dev, dtype=model.dtype)

        # a padding mask, as a batched decode has
        mask = torch.zeros(bs, 1, 1, ctx, device=dev, dtype=model.dtype)

        def attn_masked():
            sdpa_attention_forward(attn_mod, q, k, v, mask,
                                   dropout=0.0, scaling=d ** -0.5,
                                   is_causal=False)

        def attn_nomask():
            sdpa_attention_forward(attn_mod, q, k, v, None,
                                   dropout=0.0, scaling=d ** -0.5,
                                   is_causal=False)

        for name, fn in (("with a padding mask", attn_masked),
                         ("without a mask", attn_nomask)):
            try:
                t = bench(fn, n=100)
                print(f"  {name:<22} {t:6.4f} ms   x{L} = {t*L:.3f} ms")
            except Exception as exc:
                print(f"  {name:<22} failed: {str(exc)[:70]}")
    except Exception as exc:
        print(f"  could not call transformers' attention directly: {exc}")

    # --- the verdict ----------------------------------------------------
    print()
    print("=== the verdict ===")
    print("  transformers chooses between three paths in")
    print("  sdpa_attention_forward:")
    print("    enable_gqa=True   use_gqa_in_sdpa() is true")
    print("    pack_gqa          q_length == 1, a reshape, no copy")
    print("    repeat_kv         otherwise, a full materialised copy")
    print()
    print("  use_gqa_in_sdpa needs attention_mask is None. A batched decode")
    print("  has a padding mask, so it never takes that path - which is")
    print("  why the first path is unavailable in practice here.")
    print()
    print("  There is an escape hatch: TRANSFORMERS_SDPA_FORCE_GQA=1 makes")
    print("  use_gqa_in_sdpa return True regardless of the mask. Try it:")
    print()
    print("      TRANSFORMERS_SDPA_FORCE_GQA=1 python scripts/check_gqa_cost.py")
    print()
    print("  It may not help: with a mask present, sdpa cannot use a fused")
    print("  kernel anyway, and the math backend would still expand - but")
    print("  it is one environment variable to try before installing")
    print("  anything.")
    print()
    print()
    print("=== the verdict ===")
    prompt = ("Find the determinant of the 3x3 matrix A.\n\n"
              "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}")
    enc = tok([prompt] * bs, return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    past = out.past_key_values
    nxt = torch.randint(0, 1000, (bs, 1), device=dev)

    def step():
        with torch.no_grad():
            model(input_ids=nxt, past_key_values=past, use_cache=True)

    t_step = bench(step, n=20, warmup=5)
    weights = sum(p.numel() for p in model.parameters()) * 2
    floor = weights / BW * 1000
    print(f"  the step            : {t_step:6.2f} ms")
    print(f"  the bandwidth floor : {floor:6.2f} ms")
    print(f"  the step is         : {t_step/floor:.1f}x the floor")
    print()
    print(f"  the seven projections alone cost {total_ms*L:.2f} ms of the")
    print(f"  {t_step:.2f} ms step, and their own floor is {total_floor*L:.2f} ms.")
    if total_ms * L > t_step * 0.5:
        print("  => the projections ARE the step. The question is why they")
        print("     run at a quarter of the bandwidth a standalone GEMV")
        print("     reaches, and the per-projection table above says which")
        print("     ones are slow.")
    print()
    print("  Reminder: flash attention is not an option on this platform.")
    print("  PyTorch has never compiled FA2 into the Windows wheel")
    print("  (pytorch/pytorch#108175), and no community wheel matches")
    print("  torch 2.12.1 + cu130 + cp314. torch.compile needs triton,")
    print("  which is also missing.")
    print()
    print("  So the remaining levers are xformers' memory-efficient")
    print("  backend, a different torch build, or accepting it.")
    print("  Weigh those against the EOS fix, which saves ten times more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
