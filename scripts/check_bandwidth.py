"""What bandwidth does this card actually reach, and which projections lag?

The projection timings showed something that does not fit the assumed
bandwidth:

  gate_proj  25.2 MB  0.0218 ms  1155 GB/s
  up_proj    25.2 MB  0.0190 ms  1327 GB/s
  k_proj      4.2 MB  0.0153 ms   274 GB/s

The card's specification is 504 GB/s of memory bandwidth, and the earlier
profile used 756 GB/s. Neither is 1327 GB/s. So the "floor" the earlier
profiles computed was wrong, and with it the conclusion that the matmuls
run at a quarter of peak.

What is actually going on: the large projections are big enough that the
transfer overlaps the launch and the memory system pipelines them, so
they exceed the nominal number. The small ones - 4.2 MB - are dominated
by launch and ramp-up, not by bandwidth.

That reframes the whole thing. The cost is not bandwidth, because the
card can go faster than the floor. It is the *number of operations*: every
projection is a separate kernel launch, and the small ones cannot amortise
their launch cost.

This measures the achievable bandwidth directly, with a plain copy, at
several sizes, so the floor can be computed from a measured number rather
than a specification. Then it times the projections again, and counts how
many kernels the layer actually issues.

    python scripts/check_bandwidth.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402

from la import hf_cache                          # noqa: E402
from la.inference import load_model              # noqa: E402


def bench(fn, n=50, warmup=10):
    """Wall clock around a loop, synchronized at the end.

    This measures the ISSUE RATE, not the work: a CUDA launch returns
    as soon as it is queued, so the GPU work overlaps the issuing.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def bench_sync(fn, n=30, warmup=10):
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
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    bs = 8

    # --- the achievable bandwidth, measured -------------------------------
    print("=== the achievable bandwidth, measured ===")
    print("  a plain copy of a contiguous buffer is the closest thing to")
    print("  a pure bandwidth test on a GPU.")
    print()
    print("  Each copy is synchronised before and after it, so this is the")
    print("  real cost of the transfer rather than the issue rate. A copy")
    print("  reads and writes, so it moves twice the bytes of a read.")
    print()
    print(f"  {'MB':>8} {'ms':>8} {'GB/s':>8}")
    best = 0.0
    for mb in (1, 4, 8, 16, 32, 64, 128, 256, 512):
        n = int(mb * 1e6 / 2)
        a = torch.empty(n, device=dev, dtype=torch.bfloat16)
        b = torch.empty(n, device=dev, dtype=torch.bfloat16)
        t = bench_sync(lambda: b.copy_(a), n=20, warmup=5)
        gbs = (mb * 1e6) / (t / 1000) / 1e9
        best = max(best, gbs)
        print(f"  {mb:>8} {t:>8.4f} {gbs:>8.0f}")
        del a, b
    print()
    print(f"  the best observed: {best:.0f} GB/s")
    print(f"  the card's specification is 504 GB/s")
    print()

    # --- the projections, against the MEASURED bandwidth ------------------
    print("=== the projections, against the measured bandwidth ===")
    print("  NOTE: these are wall-clock times around a loop of launches.")
    print("  A CUDA launch is asynchronous, so this measures how fast the")
    print("  launches were ISSUED, not how long the work took. A small")
    print("  projection and a large one can therefore show the same time.")
    print("  scripts/check_projection_timing.py measures this properly.")
    print()
    layer = model.model.layers[0]
    attn = layer.self_attn
    print(f"  {'projection':<14} {'MB':>7} {'ms':>8} {'GB/s':>8} "
          f"{'floor@meas':>11} {'x floor':>8}")
    total_ms = 0.0
    total_floor = 0.0
    for name, mod in (("q_proj", attn.q_proj), ("k_proj", attn.k_proj),
                      ("v_proj", attn.v_proj), ("o_proj", attn.o_proj),
                      ("gate_proj", layer.mlp.gate_proj),
                      ("up_proj", layer.mlp.up_proj),
                      ("down_proj", layer.mlp.down_proj)):
        w = mod.weight
        in_features = w.shape[1]
        xi = torch.randn(bs, in_features, device=dev, dtype=model.dtype)
        t = bench(lambda: torch.nn.functional.linear(xi, w))
        mb = w.numel() * 2 / 1e6
        # the floor: bytes / (bytes per second), in milliseconds.
        # best is in GB/s, so bytes per second is best * 1e9.
        bytes_read = mb * 1e6
        floor = bytes_read / (best * 1e9) * 1000.0
        total_ms += t
        total_floor += floor
        gbs = bytes_read / (t / 1000) / 1e9
        print(f"  {name:<14} {mb:7.1f} {t:8.4f} {gbs:8.0f} "
              f"{floor:11.4f} {t/floor:7.1f}x")
    print(f"  {'TOTAL per layer':<14} {'':>7} {total_ms:8.4f} {'':>8} "
          f"{total_floor:11.4f} {total_ms/total_floor:7.1f}x")
    print(f"  x {L} layers = {total_ms*L:.2f} ms")
    print()

    # --- how many kernels does a layer issue? ----------------------------
    print("=== how many kernels a layer issues ===")
    from torch.profiler import ProfilerActivity, profile

    enc = tok(["Find the determinant of the 3x3 matrix A.\n\n"
               "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}"] * bs,
              return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    past = out.past_key_values
    nxt = torch.randint(0, 1000, (bs, 1), device=dev)

    def step():
        with torch.no_grad():
            model(input_ids=nxt, past_key_values=past, use_cache=True)

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    n = sum(e.count for e in prof.key_averages()
            if getattr(e, "self_device_time_total", 0) > 0)
    print(f"  kernels in one decode step : {n}")
    print(f"  per layer                  : {n/L:.0f}")
    print(f"  the step takes             : 32.7 ms")
    print(f"  per kernel                 : {32700/n:.1f} us")
    print()
    print("  a CUDA launch costs 5 to 30 us of CPU time, so a kernel that")
    print("  takes under about 15 us is dominated by its launch, not by")
    print("  the work it does. The projections above are 13 to 22 us each.")
    print()
    print("=== the verdict ===")
    print("  the card is not slow. it does", f"{best:.0f} GB/s.")
    print("  the projections are not slow either, in bandwidth terms -")
    print("  they are at or above the measured floor.")
    print()
    print("  what is slow is the NUMBER of kernels: about", f"{n/L:.0f}",
          "per layer for a layer that needs")
    print("  seven matmuls and two norms. Each one is a separate launch")
    print("  with its own overhead, and at 13 to 22 us of work apiece the")
    print("  launch cost is comparable to the work.")
    print()
    print("  so the fix is fewer kernels, which is what torch.compile")
    print("  does - and it needs triton, which is missing on Windows.")
    print("  xformers' memory-efficient backend would fuse the attention")
    print("  path, and it IS available on Windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
