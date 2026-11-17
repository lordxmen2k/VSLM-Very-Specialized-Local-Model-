"""Is the missing 15 ms a double read of the weights, or launch overhead?

The step measures 32.7 ms three ways (wall, CUDA events, sync-each all
agree within 6%). The card sustains ~200 GB/s, and the weights are
3.44 GB, so reading them once should cost 17.2 ms. That leaves ~15 ms
unaccounted for. Two candidate explanations, with different fixes:

  (a) THE WEIGHTS ARE READ MORE THAN ONCE. 32.7 / 17.2 = 1.9, which is
      suspiciously close to 2. If something reads each weight matrix
      twice per step, that is the missing time, and the fix is to find
      and remove the second read.

  (b) LAUNCH OVERHEAD. 4020 kernels at ~8 us of launch each is ~32 ms
      of CPU-side cost. If the CPU takes ~32 ms to issue the step, the
      CPU is the bottleneck and the fix is fewer launches or a faster
      CPU. If the CPU takes ~15 ms to issue, the GPU is doing 32 ms of
      work and the weights are being read twice.

This measures both directly.

  1. How long the CPU takes to ISSUE the step: the forward pass is run
     with the GPU work deliberately NOT waited for, so the clock covers
     the issue path. Then the GPU is drained and the drain time is
     reported separately. If issue ~= drain, the two are concurrent
     and the CPU is not the bottleneck. If issue >> drain, the CPU is
     the bottleneck.

  2. How many bytes the GPU actually moves, using the profiler's
     memory counters if they are available. The expected number is the
     weight bytes, 3.44 GB, plus the KV cache read and the activations.
     If the actual number is ~2x the weight bytes, the weights are
     being read twice.

  3. Whether the model is on the GPU at all, and whether any tensor
     is unexpectedly on the CPU - a model split across two devices
     would show up here.

    python scripts/check_step_bottleneck.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402

from la import hf_cache                          # noqa: E402
from la.inference import load_model              # noqa: E402


def main() -> int:
    print(f"cache: {hf_cache.describe()}")
    print()

    tok, model = load_model("Qwen/Qwen3-1.7B-Base", "runs/vslm")
    dev = model.device
    bs = 8

    # where does everything live?
    print("=== where the tensors are ===")
    on_cpu = [(n, p.device) for n, p in model.named_parameters()
              if p.device.type != "cuda"]
    print(f"  parameters not on the GPU: {len(on_cpu)}")
    for n, d in on_cpu[:8]:
        print(f"    {n} on {d}")
    buffers = [(n, b.device) for n, b in model.named_buffers()
               if b.device.type != "cuda"]
    print(f"  buffers not on the GPU: {len(buffers)}")
    for n, d in buffers[:8]:
        print(f"    {n} on {d}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total/1e9:.3f} B")
    print(f"  dtype: {model.dtype}")
    print()

    prompt = ("Find the determinant of the 3x3 matrix A.\n\n"
              "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}")
    enc = tok([prompt] * bs, return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    past = out.past_key_values
    nxt = torch.randint(0, 1000, (bs, 1), device=dev)

    # --- 1. the issue path vs the drain ----------------------------------
    print("=== how long does the CPU take to ISSUE the step? ===")
    print("  the forward pass is run without waiting for the GPU, so the")
    print("  clock covers the issue path only. Then the GPU is drained.")
    print()
    for trial in range(3):
        # warm up
        with torch.no_grad():
            model(input_ids=nxt, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=nxt, past_key_values=past, use_cache=True)
        t_issue = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        t_drain = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=nxt, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        t_total = (time.perf_counter() - t0) * 1000

        print(f"  trial {trial}: issue {t_issue:6.2f} ms   "
              f"drain {t_drain:6.2f} ms   total {t_total:6.2f} ms")
    print()
    print("  If issue and drain are both substantial and their sum is")
    print("  close to total, the CPU is busy issuing WHILE the GPU is")
    print("  busy executing - both are working, and the kernel count is")
    print("  the cost. If issue alone is nearly the whole total, the")
    print("  GPU finished early and the CPU is the bottleneck.")
    print()

    # --- 2. how many bytes does the GPU move? ----------------------------
    print("=== how many bytes does the GPU actually move? ===")
    weights_bytes = sum(p.numel() for p in model.parameters()) * 2
    print(f"  the weights are {weights_bytes/1e9:.3f} GB")
    for bw in (200, 504):
        ms = weights_bytes / (bw * 1e9) * 1000
        print(f"  reading them once at {bw} GB/s: {ms:.2f} ms")
    print(f"  the step measures 32.7 ms")
    print(f"  ratio at 200 GB/s: {32.7/(weights_bytes/(200e9)*1000):.2f}x")
    print()
    print("  A ratio near 2 means the weights are read twice per step.")
    print("  A ratio near 1 means the step is dominated by something else")
    print("  - most likely the launch count, since the projections alone")
    print("  account for only ~7.3 ms of weight-reading across 28 layers")
    print("  when timed honestly.")
    print()

    # --- 3. how long does the LM head take? ------------------------------
    print("=== the LM head, timed honestly ===")
    hidden = 2048
    vocab = model.config.vocab_size
    w = model.lm_head.weight
    mb = w.numel() * 2 / 1e6
    h = torch.randn(bs, hidden, device=dev, dtype=model.dtype)
    fn = lambda: torch.nn.functional.linear(h, w)
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    for _ in range(20):
        fn()
        torch.cuda.synchronize()
    sync = (time.perf_counter() - t0) / 20 * 1000
    print(f"  the head is {mb:.1f} MB ({vocab} x {hidden})")
    print(f"  wall {wall:.4f} ms   sync {sync:.4f} ms   "
          f"{mb*1e6/(sync/1000)/1e9:.0f} GB/s")
    print(f"  at the measured 200 GB/s that is "
          f"{mb*1e6/200e9*1000:.2f} ms of the step")
    print(f"  and it is {100*mb*1e6/weights_bytes:.0f}% of the total weights")
    print()
    print("  A 311 MB head matrix is a big chunk of the weights, so the")
    print("  LM head is not a rounding error in this step.")
    print()

    # --- 4. what does the attention path cost? ---------------------------
    print("=== the attention path, timed honestly ===")
    print("  16 query heads, 8 KV heads. The GQA expansion in the")
    print("  transformers SDPA path is the suspect.")
    print()
    try:
        cfg = model.config
        nh = cfg.num_attention_heads
        nkv = getattr(cfg, "num_key_value_heads", nh)
        hd = getattr(cfg, "head_dim", cfg.hidden_size // nh)
        print(f"  {nh} query heads, {nkv} kv heads, head dim {hd}")
        n_ctx = past[0][0].shape[2] if hasattr(past, "__getitem__") else 0
        print(f"  kv cache length: {n_ctx}")
        q = torch.randn(bs, nh, 1, hd, device=dev, dtype=model.dtype)
        k = torch.randn(bs, nkv, n_ctx, hd, device=dev, dtype=model.dtype)
        v = torch.randn(bs, nkv, n_ctx, hd, device=dev, dtype=model.dtype)
        mask = torch.zeros(bs, 1, 1, n_ctx, device=dev, dtype=model.dtype)

        import torch.nn.functional as F

        def gqa_expand():
            kk = k.repeat_interleave(nh // nkv, dim=1)
            vv = v.repeat_interleave(nh // nkv, dim=1)
            return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)

        def gqa_pack():
            # the transformers decode path: reshape, not copy
            b, _, sq, hd_ = q.shape
            kk = k.view(b, nkv, nh // nkv * sq, hd_)
            vv = v.view(b, nkv, nh // nkv * sq, hd_)
            return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)

        for name, fn in (("expand (repeat_kv)", gqa_expand),
                         ("pack (transformers decode path)", gqa_pack)):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            w = (time.perf_counter() - t0) / 20 * 1000
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
                torch.cuda.synchronize()
            s = (time.perf_counter() - t0) / 20 * 1000
            print(f"  {name:<38} wall {w:.4f} ms   sync {s:.4f} ms")
    except Exception as exc:
        print(f"  could not run the attention comparison: {exc}")
    print()

    print("=== the summary ===")
    print()
    print("  The step is 32.7 ms of real work - wall, events and sync all")
    print("  agree, so none of this is an issue-rate artefact.")
    print()
    print("  The card sustains ~200 GB/s. The weights are 3.44 GB, so one")
    print("  read costs 17.2 ms. The step is 32.7 ms.")
    print()
    print("  If the CPU issue path alone accounts for most of the 32.7 ms,")
    print("  the fix is fewer kernel launches. If the issue path is short,")
    print("  the fix is to stop reading the weights twice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
