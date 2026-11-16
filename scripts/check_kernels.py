"""Is the decode loop running on the fast path?

The benchmark's decode rate is about 34 ms per step at batch 8. The
memory-bandwidth floor for this model on this card is about 5 ms, so
something is costing seven times what it should.

TF32 is not the answer: it governs float32 matmuls, and this model is
bf16, which already runs on the bf16 tensor cores. The candidates that
remain are an attention kernel falling back to a software path, and the
model not being on the GPU we think it is.

    python scripts/check_kernels.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402

from la.inference import load_model              # noqa: E402


def main() -> int:
    print("=== torch build ===")
    print(f"  version      : {torch.__version__}")
    print(f"  cuda version : {torch.version.cuda}")
    print(f"  cudnn        : {torch.backends.cudnn.version()}")
    print(f"  is_available : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  no CUDA device")
        return 1

    print()
    print("=== the device ===")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i}  {p.name}  {p.total_memory/1e9:.2f} GB  "
              f"sm {p.major}.{p.minor}  multiprocessors {p.multi_processor_count}")
    print(f"  current device: {torch.cuda.current_device()}")

    print()
    print("=== the matmul precision flags ===")
    print(f"  cuda.matmul.allow_tf32 : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cudnn.allow_tf32       : {torch.backends.cudnn.allow_tf32}")
    try:
        print(f"  float32_matmul_precision: "
              f"{torch.get_float32_matmul_precision()}")
    except Exception:
        pass

    print()
    print("=== the model ===")
    tok, model = load_model("Qwen/Qwen3-1.7B-Base", "runs/vslm")
    print(f"  device              : {model.device}")
    print(f"  dtype               : {model.dtype}")
    print(f"  attn_implementation : "
          f"{getattr(model.config, '_attn_implementation', 'default')}")
    print(f"  use_cache           : {model.config.use_cache}")

    # A parameter's device says where the weights really live.
    devs = {str(p.device) for p in model.parameters()}
    print(f"  parameter devices   : {devs}")

    print()
    print("=== can this build do the fast matmul? ===")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    import time
    t = time.time()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t) / 20
    tflops = 2 * 4096 ** 3 / dt / 1e12
    print(f"  bf16 4096^3 matmul: {dt*1000:.2f} ms -> {tflops:.0f} TFLOP/s")
    print("  a 4070 SUPER does roughly 200 TFLOP/s bf16 with tensor cores")
    if tflops < 40:
        print("  *** THIS IS SLOW. The card is not using its tensor cores. ***")
        print("      On Windows this usually means the CUDA build is the one")
        print("      from PyPI, which has no CUDA at all, or that a cuBLAS")
        print("      kernel is missing and cuBLAS is falling back.")

    print()
    print("=== a real decode step, timed ===")
    enc = tok(["Find the determinant of the 3x3 matrix A.\n\n"
               "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}"] * 8,
              return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    for n in (32, 128):
        t = time.time()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        dt = time.time() - t
        print(f"  {n:4d} tokens: {dt:6.2f}s  ({dt/n*1000:5.1f} ms/step)")

    print()
    print("If the matmul is slow, no amount of prompt work will make this")
    print("benchmark fast, and the fix is the CUDA build, not the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
