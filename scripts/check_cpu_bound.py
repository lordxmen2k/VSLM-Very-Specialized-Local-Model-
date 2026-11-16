"""Is the decode step CPU-bound?

The measurements so far:

  * a decode step costs about 32.7 ms at batch 8, whatever the batch size,
    so the cost is per step rather than per token;
  * the weights are 3.44 GB, which the memory bandwidth reads in 4.6 ms;
  * the arithmetic is 0.4 ms and the attention is 0.64 ms;
  * a bare GEMV achieves 521 GB/s, so the hardware is healthy.

That leaves about 27 ms unaccounted for, and it does not depend on how much
work the step does. The remaining explanation is that the GPU is idle most
of the step, waiting for the CPU to launch the kernels. A Qwen3 layer is
around thirty kernel launches; twenty-eight layers is around eight hundred
and forty, and at twenty to thirty microseconds of launch overhead each that
is the right magnitude for the missing time.

This measures the two clocks separately:

  * the CPU time to *issue* the step, which is the queueing cost
  * the GPU time to *execute* the step, which is the work

If the CPU time is close to the wall time and the GPU time is small, the
step is CPU-bound, and the fix is fewer launches or CUDA graphs - not a
faster GPU, not a different attention backend, and not a different build.

    python scripts/check_cpu_bound.py
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
    L = model.config.num_hidden_layers
    print(f"model: {L} layers, dtype {model.dtype}, device {dev}")
    print(f"cpu threads: {torch.get_num_threads()}")

    prompt = ("Find the determinant of the 3x3 matrix A.\n\n"
              "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}")

    # --- the two clocks ------------------------------------------------
    print()
    print("=== the CPU clock and the GPU clock, separately ===")
    for bs in (1, 8):
        enc = tok([prompt] * bs, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model(**enc, use_cache=True)
        past = out.past_key_values
        nxt = torch.randint(0, 1000, (bs, 1), device=dev)

        def step():
            with torch.no_grad():
                model(input_ids=nxt, past_key_values=past, use_cache=True)

        for _ in range(5):
            step()
        torch.cuda.synchronize()

        # the wall time, which is what the user feels
        t0 = time.perf_counter()
        for _ in range(10):
            step()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / 10 * 1000

        # the CPU time, which is the time to issue the work
        t0 = time.perf_counter()
        for _ in range(10):
            step()
        cpu = (time.perf_counter() - t0) / 10 * 1000
        torch.cuda.synchronize()

        # the GPU time, from the CUDA events the kernels themselves record
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(10):
            step()
        end.record()
        torch.cuda.synchronize()
        gpu = start.elapsed_time(end) / 10

        print(f"  batch {bs}:")
        print(f"    wall clock        : {wall:6.2f} ms")
        print(f"    CPU to issue      : {cpu:6.2f} ms "
              f"({100*cpu/wall:5.1f} per cent of the wall clock)")
        print(f"    GPU to execute    : {gpu:6.2f} ms "
              f"({100*gpu/wall:5.1f} per cent of the wall clock)")
        if cpu > 0.8 * wall:
            print("    => CPU-BOUND. The GPU is waiting for the CPU to")
            print("       queue kernels. The step is launch overhead.")
        elif gpu > 0.8 * wall:
            print("    => GPU-BOUND. The work itself is the cost.")
        else:
            print("    => neither dominates; the step is partly serialised")

    # --- how many kernels are launched ---------------------------------
    print()
    print("=== how much work is one step, on the GPU ===")
    bs = 8
    enc = tok([prompt] * bs, return_tensors="pt", padding=True,
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
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()

    events = prof.key_averages()
    # the profiler reports one entry per distinct kernel, with a count of
    # how many times it ran. Summing the counts gives the number of
    # launches in the step.
    n_launch = sum(e.count for e in events
                   if getattr(e, "device_type", None) is not None
                   and "CUDA" in str(e.device_type))
    if not n_launch:
        # fall back to every event that ran on the device
        n_launch = sum(e.count for e in events if e.self_device_time_total > 0)
    print(f"  kernel launches in one step : {n_launch}")
    print(f"  the step takes              : 32.7 ms")
    if n_launch:
        print(f"  time per launch             : {32700/n_launch:.1f} us")
    print("  a typical CUDA launch costs 5 to 30 us of CPU time")
    print()

    # --- the fix, if it is launch overhead ------------------------------
    print("=== can CUDA graphs remove the launch overhead? ===")
    print("  a CUDA graph captures the whole step as one launch. It needs")
    print("  fixed shapes and static input buffers, so it is worth trying")
    print("  only if the measurement above says the step is CPU-bound.")
    print()
    print("  a cheaper fix that needs no graph: torch.compile with")
    print("  mode='reduce-overhead', which is the same idea, and CUDA")
    print("  graphs are what reduce-overhead uses.")

    print()
    print("=== a third possibility worth measuring: the CPU is slow ===")
    print(f"  torch.get_num_threads() = {torch.get_num_threads()}")
    print("  a decode step is mostly small kernel launches, and each one")
    print("  has to be prepared on the CPU. A slow CPU - an old chip, a")
    print("  thermally throttled laptop, or antivirus scanning every")
    print("  launch - shows up exactly like this.")
    print()
    print("  the machine matters: a desktop-class CPU launches kernels")
    print("  about five times faster than a laptop one.")

    print()
    print("=== the verdict ===")
    print("  if the CPU issue time is most of the wall clock, the step is")
    print("  CPU-bound and the answers are, in order of effort:")
    print("    1. torch.compile(mode='reduce-overhead') on the model")
    print("    2. a faster CPU")
    print("    3. accepting it - and checking whether it matters at all,")
    print("       because the model stopping after 206 tokens instead of")
    print("       2048 is a tenfold saving that dwarfs any of this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
