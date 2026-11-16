"""Where does the decode step actually go?

The attention backend was ruled out: attention alone costs 0.64 ms of the
38 ms step, 2 per cent. The arithmetic is trivial - 28.1 GFLOP per step at
batch 8, which is 0.4 ms at the measured 68 TFLOP/s. The weights are
3.51 GB, which the memory bandwidth reads in 5 to 7 ms. The step takes
38 ms. So most of the time is in something that is not attention and not
arithmetic.

This measures the parts of a decode step separately:

  * a pure GEMV, the shape a decode step actually is - eight rows against
    the full weight matrix - timed against the 4096-cube matmul the earlier
    diagnostic used. The cube is compute bound and hits 68 TFLOP/s; the
    GEMV is bandwidth bound and is a completely different kernel.
  * the model with the LM head detached, and with the KV cache on and off.
  * the per-layer forward, to see whether the cost is spread evenly or
    concentrated in one place.
  * the same step at batch 1, 2, 4, 8 and 16, to separate the per-token
    cost from the per-step overhead. If the time barely moves with the
    batch size, the cost is per step, not per token - which means it is
    kernel launch overhead, not bandwidth.

    python scripts/profile_decode.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402

from la import hf_cache                          # noqa: E402
from la.inference import load_model              # noqa: E402


def bench(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main() -> int:
    print(f"cache: {hf_cache.describe()}")
    print()

    tok, model = load_model("Qwen/Qwen3-1.7B-Base", "runs/vslm")
    dev = model.device
    H = model.config.hidden_size
    L = model.config.num_hidden_layers
    print(f"model: {L} layers, hidden {H}")
    print(f"dtype: {model.dtype}   device: {dev}")

    # --- what are the weights, really? -----------------------------------
    print()
    print("=== the weights, as they sit in memory ===")
    n_params = 0
    n_bf16 = 0
    for _n, p in model.named_parameters():
        n_params += p.numel()
        if p.dtype == torch.bfloat16:
            n_bf16 += p.numel()
    print(f"  parameters        : {n_params/1e9:.3f} B")
    print(f"  in bfloat16       : {n_bf16/1e9:.3f} B "
          f"({100*n_bf16/max(n_params,1):.1f} per cent)")
    print(f"  bytes of weights  : {n_params*2/1e9:.2f} GB")
    print(f"  bytes actually read per step at bf16: {n_bf16*2/1e9:.2f} GB")

    # --- the shape a decode step is -------------------------------------
    print()
    print("=== a GEMV is not a GEMM ===")
    print("the earlier diagnostic timed a 4096-cube matmul, which is")
    print("compute bound. a decode step is eight rows against the full")
    print("weight matrix, which is bandwidth bound. same library, very")
    print("different kernel.")
    w = torch.randn(H, H, device=dev, dtype=torch.bfloat16)
    x8 = torch.randn(8, H, device=dev, dtype=torch.bfloat16)
    x1 = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
    big = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    bigx = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)

    def gemv():
        return x8 @ w
    def gemv1():
        return x1 @ w
    def cube():
        return bigx @ big

    t_gemv = bench(gemv)
    t_gemv1 = bench(gemv1)
    t_cube = bench(cube, n=10)
    bw = 756e9
    print(f"  8 x 2048 @ 2048 x 2048 : {t_gemv:7.3f} ms")
    print(f"  1 x 2048 @ 2048 x 2048 : {t_gemv1:7.3f} ms")
    print(f"  4096 cube               : {t_cube:7.3f} ms "
          f"({2*4096**3/(t_cube/1000)/1e12:.1f} TFLOP/s)")
    print(f"  the GEMV's floor at {bw/1e9:.0f} GB/s: "
          f"{w.numel()*2/bw*1000:.3f} ms")
    print(f"  the GEMV achieves       : "
          f"{w.numel()*2/(t_gemv/1000)/1e9:.0f} GB/s")
    print(f"  one GEMV of the 2048-cube is {t_gemv/t_cube:.1f}x slower than "
          "the cube, for 1/64th the bytes")

    # --- the real step, at several batch sizes ---------------------------
    print()
    print("=== the real step, by batch size ===")
    print("if the cost is per token, the time grows with the batch.")
    print("if it is per step, the time barely moves.")
    prompt = ("Find the determinant of the 3x3 matrix A.\n\n"
              "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}")
    for bs in (1, 2, 4, 8, 16):
        enc = tok([prompt] * bs, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)

        def step():
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=8, do_sample=False,
                               pad_token_id=tok.pad_token_id,
                               min_new_tokens=8)
        t = bench(step, n=3, warmup=1)
        print(f"  batch {bs:>2}: {t:7.1f} ms for 8 tokens "
              f"({t/8:5.2f} ms/token)")

    # --- the forward pass, without generation ---------------------------
    print()
    print("=== the forward pass, without the generation machinery ===")
    enc = tok([prompt] * 8, return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    past = out.past_key_values
    nxt = torch.randint(0, 1000, (8, 1), device=dev)

    def fwd():
        with torch.no_grad():
            model(input_ids=nxt, past_key_values=past, use_cache=True)
    t_fwd = bench(fwd, n=20)
    print(f"  one decode forward, batch 8: {t_fwd:.1f} ms")

    # without the cache, which recomputes the whole prompt
    def fwd_nocache():
        with torch.no_grad():
            model(input_ids=enc["input_ids"], use_cache=False)
    t_nc = bench(fwd_nocache, n=5)
    print(f"  one full forward, no cache   : {t_nc:.1f} ms")

    # --- the lm head alone ----------------------------------------------
    print()
    print("=== the LM head ===")
    head = model.lm_head
    h = torch.randn(8, 1, H, device=dev, dtype=torch.bfloat16)
    try:
        t_head = bench(lambda: head(h))
        print(f"  one lm_head, batch 8: {t_head:.2f} ms")
        print(f"  that is {100*t_head/t_fwd:.0f} per cent of the forward step")
        print(f"  the vocabulary is {model.config.vocab_size:,} wide, so the")
        print(f"  head alone reads {model.config.vocab_size*H*2/1e9:.2f} GB")
    except Exception as exc:
        print(f"  could not time the lm_head alone: {exc}")

    # --- per layer -------------------------------------------------------
    print()
    print("=== is the cost spread over the layers? ===")
    print(f"  forward step {t_fwd:.1f} ms over {L} layers "
          f"= {t_fwd/L:.2f} ms per layer")
    print(f"  the weights are {n_params*2/1e9:.2f} GB, so a layer is about "
          f"{n_params*2/L/1e9:.2f} GB")
    print(f"  at {bw/1e9:.0f} GB/s a layer should take about "
          f"{n_params*2/L/bw*1000:.2f} ms")
    print(f"  measured per layer is {t_fwd/L:.2f} ms")

    print()
    print("=== the verdict ===")
    per_layer_floor = n_params * 2 / L / bw * 1000
    print(f"  bandwidth floor for the weights: {n_params*2/bw*1000:.1f} ms")
    print(f"  measured decode forward        : {t_fwd:.1f} ms")
    print(f"  ratio                           : {t_fwd/(n_params*2/bw*1000):.1f}x")
    if t_fwd > n_params * 2 / bw * 1000 * 2:
        print()
        print("  MORE THAN TWICE the bandwidth floor. The weights are being")
        print("  read more than once per step, or something else is moving")
        print("  a lot of memory. Check whether the model is on a device")
        print("  map that pages, and whether the KV cache is growing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
