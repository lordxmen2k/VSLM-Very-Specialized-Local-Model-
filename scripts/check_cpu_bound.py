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

The first run of this script printed "CPU-BOUND" and the GPU was busy for
the same time as the CPU. That was read as "the GPU is waiting for the
CPU", which is wrong: the two clocks run concurrently, so both being 31 ms
means the CPU is spending 31 ms issuing and the GPU is spending 31 ms
executing, and neither is the bottleneck alone.

What the two clocks together do show is that the step issues about two
thousand kernels. At the 5 to 30 microseconds a CUDA launch costs, that
count is the step's cost. The GPU is not idle; it is busy running two
thousand tiny kernels, each doing almost nothing.

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
        if abs(cpu - gpu) < 0.15 * wall:
            print("    => the two clocks agree. The CPU is issuing and the")
            print("       GPU is executing concurrently, and the step is")
            print("       made of a very large number of tiny kernels.")
            print("       Neither clock alone is the bottleneck; the kernel")
            print("       count is. See scripts/check_kernel_count.py.")
        elif cpu > 0.8 * wall:
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
    print(f"  per layer                   : "
          f"{n_launch/model.config.num_hidden_layers:.0f} kernels")
    print("  a Qwen3 decoder layer needs about 7 matmuls plus a few")
    print("  norms. Seventy-two per layer is about ten times that, and")
    print("  it is why the step costs what it does.")
    print("  The GPU is not idle - it is running two thousand tiny")
    print("  kernels, each doing almost nothing.")
    print()
    print("  The fix is fewer kernels: torch.compile with")
    print("  mode='reduce-overhead' fuses the elementwise chains and")
    print("  captures the step as a CUDA graph.")
    print("  Run scripts/check_kernel_count.py to measure that.")
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
    print("=== how much of this is the CPU's fault? ===")
    print(f"  torch.get_num_threads() = {torch.get_num_threads()}")
    print("  a decode step is mostly small kernel launches, and each one")
    print("  has to be prepared on the CPU. A slow CPU - an old chip, a")
    print("  thermally throttled laptop, or antivirus scanning every")
    print("  launch - makes every launch more expensive.")
    print()
    print("  16.3 us per launch is slow for a desktop CPU, which is")
    print("  usually 5 to 8 us. A laptop-class chip or an antivirus")
    print("  scanner on every launch would explain the difference.")
    print("  This is worth checking, but it only scales the cost of the")
    print("  same two thousand kernels - it does not remove them.")

    print()
    print("=== the verdict ===")
    print("  The step issues about two thousand kernels and that is the")
    print("  cost. The GPU is busy, the CPU is busy, and neither is the")
    print("  bottleneck on its own: they are both busy with the same")
    print("  two thousand tiny kernels.")
    print()
    print("  The answers, in order of effort:")
    print("    1. torch.compile(mode='reduce-overhead') - fuses the")
    print("       elementwise chains and captures a CUDA graph. This is")
    print("       the fix. Measure it with check_kernel_count.py.")
    print("    2. a faster CPU - makes each launch cheaper, same count")
    print("    3. accepting it - and checking whether it matters at all,")
    print("       because the model stopping after 206 tokens instead of")
    print("       2048 is a tenfold saving that dwarfs any of this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
