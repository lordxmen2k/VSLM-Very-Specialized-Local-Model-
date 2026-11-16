"""Diagnose a stalled benchmark batch. Run this on the machine with the GPU.

    python diagnose_stall.py

It answers three questions in under a minute:
  1. is the KV cache on?           (model.config.use_cache)
  2. what dtype are the weights?   (fp32 would be ~2x slower and 2x the memory)
  3. how fast does it actually generate, and how long does 2048 tokens take?

It does NOT need the 660-problem benchmark, so it finishes before you can
give up on it.
"""
import time

import torch

from la.inference import load_model

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "runs/vslm"

PROMPT = (
    "Find the determinant of the 3x3 matrix A.\n\n"
    "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}\n\n"
    "Show all intermediate steps. Put your final numerical answer inside \\boxed{}."
)


def main() -> int:
    print("loading ...", flush=True)
    t0 = time.time()
    tok, model = load_model(BASE, ADAPTER)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    print()
    print("=== the model as it will actually run ===")
    print(f"  use_cache          : {model.config.use_cache}")
    print(f"  dtype              : {model.dtype}")
    print(f"  device             : {model.device}")
    print(f"  attn_implementation: "
          f"{getattr(model.config, '_attn_implementation', 'default')}")
    print(f"  training mode      : {model.training}  (must be False)")

    if model.training:
        print()
        print("  *** THE MODEL IS STILL IN TRAINING MODE ***")
        print("  Dropout is active and the cache is off. Call model.eval().")

    if not getattr(model.config, "use_cache", True):
        print()
        print("  *** THE KV CACHE IS OFF ***")
        print("  Every token re-runs the whole prefix. That is quadratic.")
        print("  Fix: model.config.use_cache = True  (and .eval())")

    print()
    print("=== nvidia-smi ===")
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
             "temperature.gpu,clocks.sm,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        print("  " + out.stdout.strip())
        print("  (utilisation should be ~100% while generating, and 0 when idle)")
    except Exception as exc:
        print(f"  could not read nvidia-smi: {exc}")

    print()
    print("=== measured decode rate, batch 8 ===")
    enc = tok([PROMPT] * 8, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    print(f"  prompt shape: {tuple(enc['input_ids'].shape)}")
    for n in (32, 128):
        t = time.time()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        dt = time.time() - t
        per_step_ms = dt / n * 1000
        print(f"  {n:4d} tokens: {dt:6.2f}s  ({per_step_ms:5.1f} ms/step)"
              f"  -> 2048 tokens would take {dt * 2048 / n / 60:5.1f} min")

    print()
    print("=== one problem, generation length ===")
    t = time.time()
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=512, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    for j in range(8):
        n_new = g[j][enc["input_ids"].shape[1]:].shape[0]
        print(f"  sequence {j}: {n_new} new tokens")
    print(f"  ({time.time()-t:.1f}s)")
    print()
    print("If the sequences stop well short of 512, the model emits EOS and")
    print("the benchmark will be fast. If they all hit 512, it never stops,")
    print("and every one of the 660 problems costs the full max_new_tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
