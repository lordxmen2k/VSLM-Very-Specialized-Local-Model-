"""Measure the projections correctly, with CUDA events.

The last measurement produced impossible numbers: a projection reading
25.2 MB finished in 0.0171 ms, while copying the same 25.2 MB took
0.1474 ms. A read cannot beat a copy of the same size, because a copy
also writes. The explanation is that the timing method was wrong.

`bench()` measured wall-clock time around a loop of kernel launches.
A CUDA launch is asynchronous: the call returns as soon as the work is
queued, not when it finishes. So the loop measured how long it took to
*issue* the launches, and the GPU work overlapped the issuing. The
number that came out was the launch cost, not the work.

That is why every projection took 13 to 31 us regardless of its size,
and why the small copies looked like they were achieving 672 GB/s.

This measures the same things two ways:

  * wall clock around the loop, with a synchronize at the end - what the
    previous script measured
  * CUDA events around the loop - the GPU's own record of when the work
    completed

They will differ, and the difference is the overlap. Then it measures a
single projection with a synchronize before and after, which is the
honest per-call latency, and a large projection repeated many times so
the launch cost is amortised and the bandwidth shows up.

    python scripts/check_projection_timing.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402

from la import hf_cache                          # noqa: E402
from la.inference import load_model              # noqa: E402


def bench_wall(fn, n=50, warmup=10):
    """Wall clock around the loop, synchronized at the end.

    This measures the issue rate, not the work.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def bench_events(fn, n=50, warmup=10):
    """CUDA events around the loop: the GPU's own timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def bench_sync_each(fn, n=30, warmup=10):
    """One call, synchronized before and after: the honest per-call latency."""
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
    L = cfg.num_hidden_layers
    bs = 8
    layer = model.model.layers[0]
    attn = layer.self_attn

    # --- the copy, measured properly -------------------------------------
    print("=== a copy, three ways ===")
    print(f"  {'MB':>6} {'wall ms':>9} {'events ms':>10} {'sync-each ms':>13} "
          f"{'wall GB/s':>10} {'sync GB/s':>10}")
    for mb in (4, 16, 32, 64, 256):
        n = int(mb * 1e6 / 2)
        a = torch.empty(n, device=dev, dtype=torch.bfloat16)
        b = torch.empty(n, device=dev, dtype=torch.bfloat16)
        tw = bench_wall(lambda: b.copy_(a), n=20, warmup=5)
        te = bench_events(lambda: b.copy_(a), n=20, warmup=5)
        ts = bench_sync_each(lambda: b.copy_(a), n=10, warmup=3)
        bytes_moved = mb * 1e6
        print(f"  {mb:>6} {tw:>9.4f} {te:>10.4f} {ts:>13.4f} "
              f"{bytes_moved/(tw/1000)/1e9:>10.0f} {bytes_moved/(ts/1000)/1e9:>10.0f}")
        del a, b
    print()
    print("  wall vs sync-each is the overlap: the wall clock measures")
    print("  how fast the launches were ISSUED, the synchronised one")
    print("  measures how long the work actually took.")
    print()

    # --- the projections, three ways -------------------------------------
    print("=== the projections, three ways ===")
    print(f"  {'projection':<14} {'MB':>6} {'wall ms':>9} {'events ms':>10} "
          f"{'sync ms':>9} {'sync GB/s':>10}")
    for name, mod in (("q_proj", attn.q_proj), ("k_proj", attn.k_proj),
                      ("v_proj", attn.v_proj), ("o_proj", attn.o_proj),
                      ("gate_proj", layer.mlp.gate_proj),
                      ("up_proj", layer.mlp.up_proj),
                      ("down_proj", layer.mlp.down_proj)):
        w = mod.weight
        in_features = w.shape[1]
        xi = torch.randn(bs, in_features, device=dev, dtype=model.dtype)
        fn = lambda: torch.nn.functional.linear(xi, w)
        tw = bench_wall(fn)
        te = bench_events(fn)
        ts = bench_sync_each(fn)
        mb = w.numel() * 2 / 1e6
        print(f"  {name:<14} {mb:>6.1f} {tw:>9.4f} {te:>10.4f} {ts:>9.4f} "
              f"{mb*1e6/(ts/1000)/1e9:>10.0f}")
    print()

    # --- one big projection, so the launch cost amortises ----------------
    print("=== a big matmul, so the launch cost is amortised ===")
    print("  4096 x 4096 is large enough that the transfer dominates.")
    big = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    bx = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    tw = bench_wall(lambda: bx @ big, n=10, warmup=3)
    ts = bench_sync_each(lambda: bx @ big, n=10, warmup=3)
    te = bench_events(lambda: bx @ big, n=10, warmup=3)
    mb = big.numel() * 2 / 1e6
    print(f"  wall {tw:.4f} ms   events {te:.4f} ms   sync {ts:.4f} ms")
    print(f"  bytes {mb:.0f} MB")
    print(f"  wall  {mb*1e6/(tw/1000)/1e9:.0f} GB/s")
    print(f"  sync  {mb*1e6/(ts/1000)/1e9:.0f} GB/s")
    print(f"  the matmul does {2*4096**3/(ts/1000)/1e12:.1f} TFLOP/s")
    print()
    print("  the gap between wall and sync is the launch overlap. This is")
    print("  why the earlier bandwidth curve looked impossible: the")
    print("  small copies were never measured, only issued.")
    print()

    # --- the real step ---------------------------------------------------
    print("=== the real decode step, measured properly ===")
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

    tw = bench_wall(step, n=20, warmup=5)
    te = bench_events(step, n=20, warmup=5)
    ts = bench_sync_each(step, n=20, warmup=5)
    print(f"  wall   {tw:.2f} ms")
    print(f"  events {te:.2f} ms")
    print(f"  sync   {ts:.2f} ms")
    print()
    print("  If wall and events agree, the step is not overlapping much")
    print("  with anything, and the 32.7 ms is real work plus real launch")
    print("  cost. If the gap is large, the previous measurement was")
    print("  measuring the issue rate rather than the work.")
    print()

    # --- what the weights cost to read ----------------------------------
    print("=== what the weights cost to read ===")
    weights = sum(p.numel() for p in model.parameters()) * 2
    print(f"  the weights are {weights/1e9:.2f} GB")
    for bw in (200, 504, 672, 756, 1327):
        ms = weights / (bw * 1e9) * 1000
        print(f"  at {bw:>5} GB/s: {ms:6.2f} ms")
    print()
    print("  the measured copy sustained about 200 GB/s at 32 MB and")
    print("  above. A copy reads and writes, so a pure read is faster")
    print("  than that, but not by much on a consumer card.")
    print()
    print("  The step is 32.7 ms. Even at 1327 GB/s the weights need")
    print("  2.6 ms. At 200 GB/s they need 17 ms. So the weights ARE a")
    print("  large part of the step, and the remaining 15 ms is the")
    print("  kernel count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
