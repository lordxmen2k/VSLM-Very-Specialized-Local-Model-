"""Which SDPA backend is the attention actually using?

The decode loop runs at about 38 ms per step at batch 8. The memory
bandwidth floor is about 5 ms, so something is costing seven times what it
should.

The torch build is fine - 2.12.1+cu130, cuDNN 9.2, sm_8.9, the model on
cuda:0, and a bf16 matmul that does 68 TFLOP/s. So this is not a driver
problem and not a wrong-CUDA problem.

The next suspect is the attention backend. PyTorch's own documentation says
that with FP16/BF16 inputs the SDPA *math* backend upcasts to FP32, which
"increases memory usages and may cause the performance regressions". If the
model is on the math backend rather than flash, every one of the 28 layers
pays for that on every step.

    python scripts/check_attention.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402

from la.inference import load_model              # noqa: E402


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


def main() -> int:
    print("=== what backends does this build have? ===")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        print("  torch.nn.attention is available")
        backends = [SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.MATH]
    except Exception as exc:
        print(f"  torch.nn.attention not available: {exc}")
        backends = []

    tok, model = load_model("Qwen/Qwen3-1.7B-Base", "runs/vslm")
    print(f"  attn_implementation: "
          f"{getattr(model.config, '_attn_implementation', 'default')}")

    print()
    print("=== is flash attention even compiled in? ===")
    try:
        from torch.backends.cuda import (
            can_use_flash_attention, can_use_efficient_attention)
        params = torch.nn.Linear(2048, 2048, bias=False, device="cuda",
                                 dtype=torch.bfloat16)
        q = torch.randn(8, 16, 128, 200, device="cuda", dtype=torch.bfloat16)
        print(f"  flash usable    : {can_use_flash_attention(params, q)}")
        print(f"  mem-efficient   : {can_use_efficient_attention(params, q)}")
    except Exception as exc:
        print(f"  could not query: {exc}")

    print()
    print("=== a decode step, per backend ===")
    enc = tok(["Find the determinant of the 3x3 matrix A.\n\n"
               "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}"] * 8,
              return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    n = 64

    def run(label):
        t = time.time()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        dt = time.time() - t
        print(f"  {label:<24} {dt/n*1000:6.1f} ms/step")
        return dt / n

    run("default")

    if backends:
        from torch.nn.attention import sdpa_kernel
        for b in backends:
            try:
                with sdpa_kernel(b):
                    run(str(b).split(".")[-1])
            except Exception as exc:
                print(f"  {str(b).split('.')[-1]:<24} unavailable: "
                      f"{str(exc)[:60]}")

    print()
    print("=== attention alone, at a decode-like shape ===")
    L, H, KVH, D = 28, 2048, 8, 128
    q = torch.randn(8, KVH, 1, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, KVH, 200, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(8, KVH, 200, D, device="cuda", dtype=torch.bfloat16)

    def sdpa():
        with torch.no_grad():
            torch.nn.functional.scaled_dot_product_attention(q, k, v)

    dt = bench(sdpa)
    print(f"  one layer, 8 sequences, 200 cached tokens: {dt*1000:.3f} ms")
    print(f"  x {L} layers = {dt*1000*L:.2f} ms/step of attention alone")
    print(f"  that is {dt*1000*L/38.1*100:.0f}% of the 38 ms step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
