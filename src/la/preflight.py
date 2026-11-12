"""Check the machine before it is asked to train.

A CUDA mismatch inside bitsandbytes otherwise surfaces as a thousand-line
traceback in the middle of model loading, after the corpus has been built and
the GPU has been woken up. This prints the answer in a dozen lines.

    python -m la.preflight
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys


def bitsandbytes_binaries() -> tuple[str, list[str]]:
    """The CUDA versions this bitsandbytes install actually ships.

    bitsandbytes bundles one native library per CUDA release it supports.
    Listing them is the ground truth; the version check inside the library
    only tells you when you are already too late.
    """
    try:
        import bitsandbytes
    except Exception:
        return "", []
    directory = os.path.dirname(bitsandbytes.__file__)
    found: set[str] = set()
    for pattern in ("libbitsandbytes_cuda*.so", "libbitsandbytes_cuda*.dll"):
        for path in glob.glob(os.path.join(directory, pattern)):
            m = re.search(r"cuda(\d{2,3})", os.path.basename(path))
            if m:
                digits = m.group(1)
                found.add(f"{int(digits) // 10}.{int(digits) % 10}")
    ordered = sorted(found, key=lambda s: tuple(int(x) for x in s.split(".")))
    return getattr(bitsandbytes, "__version__", "?"), ordered


def nvidia_smi() -> list[dict]:
    """What nvidia-smi says about each GPU right now.

    PyTorch's own counters do not tell you whether anything else is using
    the GPU. On Windows the display driver time-slices the GPU between
    processes, so a background application that holds a context will
    interrupt a training step for an unpredictable length of time - which
    looks exactly like a slow GPU and is not one.
    """
    query = ("name,memory.used,memory.total,utilization.gpu,"
             "temperature.gpu,clocks.sm")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    if out.returncode != 0:
        return []
    fields = ["name", "used_mib", "total_mib", "util_pct", "temp_c", "clock_mhz"]
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        rows.append(dict(zip(fields, parts)))
    return rows


def main(argv=None) -> int:
    ok = True
    print("=" * 68)
    print("  VSLM preflight")
    print("=" * 68)
    print(f"  python            {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch
    except Exception as exc:
        print(f"  torch             NOT INSTALLED ({exc})")
        print()
        print("  pip install -e \".[train,dev]\"")
        print("  and install torch from the index that matches your GPU:")
        print("      scripts/setup_windows_gpu.bat     (Windows)")
        print("      pip install torch --index-url "
              "https://download.pytorch.org/whl/cu130")
        return 2
    print(f"  torch             {torch.__version__}")
    print(f"  cuda available    {torch.cuda.is_available()}")
    print(f"  torch cuda build  {torch.version.cuda}")

    if not torch.cuda.is_available():
        print()
        print("  no CUDA device: training cannot run here.")
        print("  the corpus and the tests still work on CPU:")
        print("      make data && make test")
        return 2

    count = torch.cuda.device_count()
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  gpu {i}             {name}")
        print(f"  vram {i}            {total:.1f} GB")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    for row in nvidia_smi():
        try:
            used = int(row["used_mib"])
            cap = int(row["total_mib"])
            util = int(row["util_pct"])
            temp = int(row["temp_c"])
        except (ValueError, KeyError):
            continue
        print(f"  nvidia-smi        {row['name']}")
        print(f"    memory          {used:,} / {cap:,} MiB in use")
        print(f"    utilisation     {util}%   temperature {temp} C   "
              f"sm clock {row['clock_mhz']} MHz")
        if util > 5:
            print("    WARNING: the GPU is busy. Something else is using it.")
            print("    On Windows the display driver time-slices the GPU between")
            print("    processes, so a browser, a video call or a container can")
            print("    interrupt a training step for an unpredictable time. Close")
            print("    them before a long run and watch the step times settle.")

    bnb_version, binaries = bitsandbytes_binaries()
    torch_cuda = torch.version.cuda
    print(f"  bitsandbytes      {bnb_version}")
    if binaries:
        print(f"  bnb cuda builds   {', '.join(binaries)}")
    if torch_cuda and binaries:
        short = ".".join(torch_cuda.split(".")[:2])
        if short in binaries:
            print(f"  cuda match        YES - torch is CUDA {torch_cuda} and "
                  f"bitsandbytes ships it")
        else:
            ok = False
            newest = binaries[-1]
            print(f"  cuda match        NO  - torch is CUDA {short}, "
                  f"bitsandbytes has no binary for it")
            print(f"                     the newest bitsandbytes binary is CUDA {newest}")
            print()
            print("  this is the mismatch that produces the long traceback:")
            print("  the bitsandbytes import succeeds, and the failure arrives")
            print("  later, inside model loading.")
            print()
            print("  fix it by moving torch down to a CUDA version bitsandbytes")
            print("  ships, not by looking for a newer bitsandbytes:")
            print("      scripts/setup_windows_gpu.bat     (Windows)")
            print("      pip install --index-url "
                  "https://download.pytorch.org/whl/cu130 \\")
            print("          torch==2.12.1+cu130 torchvision==0.27.1+cu130")
            print("      pip install --no-deps bitsandbytes==0.50.2")

    from .train import bitsandbytes_works
    usable, reason = bitsandbytes_works()
    if usable:
        print("  4-bit quantisation WORKS")
        print("  precision         4bit (about 2 GB of VRAM for the model)")
    else:
        print(f"  4-bit quantisation UNAVAILABLE")
        print(f"                     {reason}")
        print("  precision         bf16 LoRA (about 7 GB of VRAM in total)")
        print()
        print("  this still fits a 12 GB card, so `make train` will work.")
        print("  it only gives up about 2.5 GB of headroom.")

    for path, label in (("data/train.jsonl", "training corpus"),
                        ("data/test.jsonl", "held-out set")):
        if os.path.exists(path):
            n = sum(1 for _ in open(path, encoding="utf-8"))
            print(f"  {label:<17} {path}  ({n:,} rows)")
        else:
            ok = False
            print(f"  {label:<17} MISSING - run `make data`")

    print("=" * 68)
    if not ok:
        print("  some checks failed; the lines above say what to do about each.")
    else:
        print("  ready. `make train`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
