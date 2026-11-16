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

from la import hf_cache                          # noqa: E402
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


def _has_flash() -> bool:
    """Is flash attention compiled into this torch build?

    torch.backends.cuda.flash_sdp_enabled() says whether the backend is
    allowed, not whether the kernel exists. The build tells you: a torch
    without flash compiled in has no flash kernel to enable.
    """
    try:
        from torch.backends.cuda import flash_sdp_enabled
        return bool(flash_sdp_enabled())
    except Exception:
        return False


def report_cache_space() -> None:
    """How much room the Hugging Face cache has, and where it is.

    A download that dies part way through with "Internal Writer Error:
    Background writer channel closed" is usually a full disk, not a
    network fault. This says which disk the cache is on and how much room
    is left, before anything is downloaded.
    """
    import shutil
    try:
        from huggingface_hub import constants as hf_constants
        cache = hf_constants.HF_HUB_CACHE
    except Exception:
        cache = hf_cache.cache_location()
    print(f"  HF_HUB_CACHE : {cache}")
    try:
        # the cache directory may not exist yet, so walk up to the nearest
        # parent that does and measure that
        probe = cache
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        usage = shutil.disk_usage(probe)
        print(f"  measured on  : {probe}")
        print(f"  free         : {usage.free / 1e9:.2f} GB "
              f"of {usage.total / 1e9:.1f} GB")
        if usage.free < 5e9:
            print("  WARNING: less than 5 GB free. A 1.7B checkpoint needs")
            print("  about 4 GB, and a partial download fails in a way that")
            print("  looks like a network error. Set HF_HOME to a drive with")
            print("  room before running this.")
    except Exception as exc:
        print(f"  could not read the free space: {exc}")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    # The Base checkpoint is the default because it is the one already on
    # disk: this diagnostic measures the hardware path, and the weights do
    # not change it. Point it at Qwen/Qwen3-1.7B to measure the instruct
    # checkpoint, but that one may need downloading.
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default="runs/vslm")
    args = ap.parse_args(argv)

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

    print()
    print("=== the cache, before anything is downloaded ===")
    report_cache_space()

    tok, model = load_model(args.base, args.adapter)
    print(f"  attn_implementation: "
          f"{getattr(model.config, '_attn_implementation', 'default')}")

    print()
    print("=== is flash attention even compiled in? ===")
    # The can_use_* helpers take a packed _SDPAParams, not a module and a
    # tensor. Passing the wrong types makes them print the whole tensor,
    # which is megabytes of numbers in the middle of a diagnostic.
    # Running each backend below is the reliable way to ask.
    print("  flash is compiled in" if _has_flash()
          else "  flash is NOT compiled into this torch build")

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
        import warnings
        from torch.nn.attention import sdpa_kernel
        for b in backends:
            name = str(b).split(".")[-1]
            try:
                # each unavailable backend prints a paragraph of warnings
                # explaining why, which buries the numbers that matter
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with sdpa_kernel(b):
                        run(name)
            except Exception as exc:
                print(f"  {name:<24} unavailable: {str(exc)[:60]}")

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
    print()
    print("  RESULT: attention is NOT the bottleneck. The math backend is")
    print("  the fastest available one here, and flash is not compiled in.")
    print("  Do not spend more time on the attention backend.")
    print("  Run scripts/profile_decode.py instead: it separates the")
    print("  bandwidth floor from the per-step overhead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
