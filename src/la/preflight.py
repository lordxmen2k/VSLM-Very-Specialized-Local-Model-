"""Check the machine before it is asked to train.

A CUDA mismatch inside bitsandbytes otherwise surfaces as a thousand-line
traceback in the middle of model loading, after the corpus has been built and
the GPU has been woken up. This prints the answer in a dozen lines.

    python -m la.preflight
"""
from __future__ import annotations

import os
import sys


def main(argv=None) -> int:
    ok = True
    print("=" * 66)
    print("  VSLM preflight")
    print("=" * 66)
    print(f"  python            {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch
    except Exception as exc:
        print(f"  torch             NOT INSTALLED ({exc})")
        print("\n  pip install -e \".[train,dev]\"")
        return 2
    print(f"  torch             {torch.__version__}")
    print(f"  cuda available    {torch.cuda.is_available()}")
    print(f"  torch cuda build  {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\n  no CUDA device: training cannot run here.")
        print("  the corpus and the tests still work on CPU:")
        print("      make data && make test")
        return 2

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  gpu               {name}")
    print(f"  vram              {total:.1f} GB")

    from .train import bitsandbytes_works
    usable, reason = bitsandbytes_works()
    if usable:
        print("  bitsandbytes      4-bit NF4 quantisation WORKS")
        print("  precision         4bit (about 2 GB of VRAM for the model)")
    else:
        ok = False
        print(f"  bitsandbytes      4-bit quantisation UNAVAILABLE")
        print(f"                     {reason}")
        print("  precision         bf16 LoRA (about 7 GB of VRAM in total)")
        print()
        print("  this still fits a 12 GB card, so `make train` will work.")
        print("  to fix the wheel, match bitsandbytes to your torch CUDA build:")
        print("      pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128")
        print("      pip install --force-reinstall bitsandbytes")

    for path, label in (("data/train.jsonl", "training corpus"),
                        ("data/test.jsonl", "held-out set")):
        if os.path.exists(path):
            n = sum(1 for _ in open(path, encoding="utf-8"))
            print(f"  {label:<17} {path}  ({n:,} rows)")
        else:
            ok = False
            print(f"  {label:<17} MISSING - run `make data`")

    print("=" * 66)
    if not ok:
        print("  some checks failed; the lines above say what to do about each.")
    else:
        print("  ready. `make train`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
