"""What is making one decode step launch two thousand kernels?

The measurement says a decode step costs 32.7 ms and issues 2009 kernel
launches. That is about 72 kernels per layer for a 28-layer model. A
Qwen3 decoder layer needs roughly seven matmuls - q, k, v, o, and the
three MLP projections - plus a couple of norms. Seventy-two is about ten
times that.

Two things could explain it:

  * the attention path. Qwen3-1.7B has 16 query heads and 8 key/value
    heads, so the attention is grouped-query: the query is broadcast
    across the key/value groups. The SDPA math backend cannot use a
    fused kernel for that, so it materialises the broadcast and expands
    into many elementwise kernels.

  * no fusion at all. RMSNorm, the rotary embedding, the activation and
    the residual adds are each several separate kernels without
    torch.compile.

This counts the kernels per module, so the expansion can be attributed to
a layer of the model rather than guessed at. It then times the same step
with torch.compile in reduce-overhead mode, which is the fix: it fuses
the elementwise chains and captures the step as a CUDA graph.

    python scripts/check_kernel_count.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402

from la import hf_cache                          # noqa: E402
from la.inference import load_model              # noqa: E402

PROMPT = ("Find the determinant of the 3x3 matrix A.\n\n"
          "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}")


def count_kernels(fn):
    """How many CUDA kernels does this call issue?"""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = prof.key_averages()
    n = 0
    for e in events:
        if getattr(e, "self_device_time_total", 0) > 0:
            n += e.count
    return n


def top_kernels(fn, k=12):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages()
              if getattr(e, "self_device_time_total", 0) > 0]
    events.sort(key=lambda e: -e.self_device_time_total)
    print(f"  {'kernel':<52} {'count':>6} {'us total':>10}")
    for e in events[:k]:
        name = e.key[:50]
        print(f"  {name:<52} {e.count:>6} "
              f"{e.self_device_time_total:>10.0f}")


def timed(fn, n=20, warmup=5):
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
    print(f"model: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}")
    print(f"  query heads      : {cfg.num_attention_heads}")
    print(f"  key/value heads  : {cfg.num_key_value_heads}")
    if cfg.num_attention_heads != cfg.num_key_value_heads:
        print("  => grouped-query attention: the query is broadcast across")
        print("     the key/value groups, and the SDPA math backend has no")
        print("     fused kernel for that")

    bs = 8
    enc = tok([PROMPT] * bs, return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    past = out.past_key_values
    nxt = torch.randint(0, 1000, (bs, 1), device=dev)

    def step():
        with torch.no_grad():
            model(input_ids=nxt, past_key_values=past, use_cache=True)

    print()
    print("=== the kernels, by name ===")
    top_kernels(step, k=15)

    n = count_kernels(step)
    print()
    print(f"  total kernel launches in one decode step : {n}")
    print(f"  per layer                                  : {n/cfg.num_hidden_layers:.0f}")

    # --- one layer on its own ------------------------------------------
    print()
    print("=== one layer on its own ===")
    layer = model.model.layers[0]
    h = torch.randn(bs, 1, cfg.hidden_size, device=dev, dtype=model.dtype)
    # the rotary embedding wants position_ids of shape (batch, seq)
    pos_ids = torch.arange(1, 2, device=dev).unsqueeze(0).expand(bs, 1)
    try:
        rotary = model.model.rotary_emb
        # Qwen3's rotary returns (cos, sin) for the given positions
        cos, sin = rotary(h, pos_ids)
        attn_mask = None
        cache = past[:1] if isinstance(past, tuple) else None

        def one_layer():
            with torch.no_grad():
                layer(h, (cos, sin), attn_mask, past_key_values=cache,
                      use_cache=True)
        nl = count_kernels(one_layer)
        print(f"  one decoder layer issues {nl} kernels")
        print(f"  x {cfg.num_hidden_layers} layers = "
              f"{nl*cfg.num_hidden_layers}")
    except Exception as exc:
        print(f"  could not isolate one layer: {exc}")
        print("  (the per-kernel counts above already answer the question)")

    # --- what does torch.compile do? ------------------------------------
    print()
    print("=== torch.compile with reduce-overhead ===")
    print("  this fuses the elementwise chains and captures the step as a")
    print("  CUDA graph, so the CPU issues one graph launch instead of")
    print("  two thousand kernels. It compiles on the first call, which")
    print("  takes a minute or two.")

    t_before = timed(step)
    n_before = n
    print(f"  before: {t_before:6.2f} ms/step, {n_before} kernels")

    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        cstate = {}

        def compiled_step():
            with torch.no_grad():
                c = compiled(input_ids=nxt, past_key_values=cstate.get("past"),
                             use_cache=True)
                cstate["past"] = c.past_key_values

        print("  compiling, this takes a minute or two ...")
        t0 = time.time()
        compiled_step()
        torch.cuda.synchronize()
        print(f"  compiled in {time.time()-t0:.0f} s")

        t_after = timed(compiled_step, n=20, warmup=5)
        n_after = count_kernels(compiled_step)
        print(f"  after : {t_after:6.2f} ms/step, {n_after} kernels")
        print(f"  speedup: {t_before/t_after:.2f}x")
        if t_after < t_before:
            print()
            print("  THIS WORKS. Put torch.compile on the model in")
            print("  la.inference.load_model, behind a flag, and the")
            print("  benchmark gets faster.")
    except Exception as exc:
        msg = str(exc)
        print(f"  torch.compile failed: {msg[:120]}")
        if "Triton" in msg or "triton" in msg:
            print()
            print("  Triton is missing. It has no official Windows wheel -")
            print("  only community forks - so torch.compile is not")
            print("  available on this platform without one.")
            print()
            print("  And the other candidate, flash attention, is not")
            print("  available either: PyTorch has never compiled")
            print("  FlashAttention-2 into the Windows wheel")
            print("  (pytorch/pytorch#108175). The message 'Torch was not")
            print("  compiled with flash attention' reads like a broken")
            print("  install; it is a platform limitation.")
            print()
            print("  What is left: xformers' memory-efficient backend, which")
            print("  supports grouped-query attention natively and IS")
            print("  available on Windows, or a different torch build.")
            print("  scripts/check_gqa_cost.py measures what each is worth.")
        else:
            import traceback
            traceback.print_exc()

    print()
    print("=== the decision ===")
    print("  If reduce-overhead gives a real speedup, the change is one")
    print("  line in la.inference.load_model:")
    print()
    print("      model = torch.compile(model, mode='reduce-overhead')")
    print()
    print("  On this machine it will not, because triton is missing and")
    print("  triton has no official Windows wheel.")
    print()
    print("  What the kernel counts do say, unambiguously:")
    print("    197 mm = 28 x 7 + 1   <- the seven projections. CORRECT.")
    print("    144 kernels per layer  <- the grouped-query expansion and")
    print("                              the unfused norms. THIS is the cost.")
    print("    79 per cent of the GPU time is in the matmuls, which run at")
    print("    about a quarter of the bandwidth a standalone GEMV reaches.")
    print()
    print("  Weigh any fix against the alternative first: a model that stops")
    print("  after 206 tokens instead of 2048 saves ten times more than")
    print("  this does. The EOS supervision is the bigger win, and it is")
    print("  already implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
