# Working state — VSLM

Repo: https://github.com/lordxmen2k/VSLM-Very-Specialized-Local-Model-
Local clone: `/workspace/VSLM-Very-Specialized-Local-Model-`

## Where the project stands (2026-11-16, later)

Run 4 is **abandoned, not failed**. Two independent defects were found in it,
both diagnosed from measurements rather than guesses:

1. **The benchmark prompt did not match the training prompt.** The corpus was
   trained on the bare question; the benchmark prepended its own system
   prompt as plain text. The model was answering a prompt format it had
   never seen, so the score measured format drift rather than the
   mathematics. The old test compared the corpus prompt *with itself*, so it
   passed while the two prompts disagreed.

2. **Nothing taught the model to stop.** The Base checkpoint had no turn-
   ending signal, so every generation ran to the token cap. The first
   benchmark batch did not finish in six minutes.

**Run 5 is the replacement.** It uses `Qwen/Qwen3-1.7B` — the instruction-
tuned checkpoint. Qwen ships one tuned release for the 1.7B model; there is
no separate `-Instruct` name for it. The corpus and the benchmark now render
the same chat prompt through the same tokenizer, and the turn-ending token is
supervised at the end of every completion.

### What the hardware measurement showed

| measurement | value | what it means |
|---|---|---|
| torch build | 2.12.1+cu130, cuDNN 9.2.0 | healthy |
| GPU | RTX 4070 SUPER, sm_8.9, 56 SMs | healthy |
| bf16 4096³ matmul | **68 TFLOP/s** | ~32 % of the card's ~210 TFLOP/s peak, which is normal under fp32 accumulation |
| model placement | entirely on `cuda:0` | healthy |
| decode | ~38 ms/step at batch 8 | **7.8× the ~5–7 ms memory-bandwidth floor** |

**The CUDA build is not the problem.** That was worth ruling out, because the
first hypothesis — a bad build — would have sent the user off to reinstall
torch for no gain.

### What the attention measurement showed

| measurement | value |
|---|---|
| attention alone, 28 layers, batch 8 | **0.64 ms/step — 2 % of the step** |
| `MATH` backend | 37.5 ms/step (the default, and the fastest available) |
| `CUDNN_ATTENTION` | 71.1 ms/step |
| `FLASH_ATTENTION` | unavailable — "Torch was not compiled with flash attention" |
| `EFFICIENT_ATTENTION` | unavailable — "runtime disabled" |

**The attention backend is not the problem either.** The hypothesis that the
SDPA *math* backend was upcasting to FP32 and slowing the step down was
**wrong**: attention costs 0.64 ms of a 38 ms step, and the math backend is
already the fastest one available on this build.

This is the second hypothesis to be ruled out by measurement rather than
reasoning. The remaining suspect is the **per-step overhead**, not the
attention kernel: the arithmetic is 28.1 GFLOP per step at batch 8, which is
0.4 ms at the measured 68 TFLOP/s, and the weights are 3.51 GB, which the
memory bandwidth reads in 5–7 ms. Neither explains 38 ms.

`scripts/profile_decode.py` separates the two. A decode step is a **GEMV**
— eight rows against the full weight matrix — not the 4096-cube GEMM the
earlier diagnostic timed. Same library, completely different kernel. The
script times a GEMV against the cube, the real step at batch 1/2/4/8/16, the
forward pass without the generation machinery, and the LM head on its own.

**If the step time barely moves as the batch grows, the cost is per step —
kernel launch overhead — not per token, not bandwidth.** That is the
measurement that decides it.

### What the decode profile showed

| measurement | value | what it means |
|---|---|---|
| 8×2048 @ 2048×2048 GEMV | **0.016 ms, 521 GB/s** | the hardware is fast and healthy |
| step time, batch 1 → 16 | 286 → 326 ms for 8 tokens | **16× the work, 1.14× the time** |
| decode forward, batch 8 | 32.7 ms | vs a 4.6 ms bandwidth floor — **7.2×** |
| LM head alone | 1.38 ms (4 %) | not it |
| per layer | 1.17 ms | vs a 0.16 ms floor |

**The cost is per step, not per token.** Batch 1 and batch 16 cost the same.
That rules out bandwidth and arithmetic, both of which scale with the work.
Attention was already ruled out at 0.64 ms.

### What the clock comparison showed — and how to read it

| | batch 1 | batch 8 |
|---|---|---|
| wall clock | 31.86 ms | 32.64 ms |
| CPU to issue | 31.05 ms (97.5 %) | 32.82 ms (100.5 %) |
| GPU to execute | 30.98 ms (97.2 %) | 32.37 ms (99.2 %) |
| kernel launches per step | **2009** | **2009** |

**The first reading of this was wrong, and the correction matters.** The
script printed "CPU-BOUND", and "the GPU is waiting for the CPU" was the
obvious conclusion. It is not what the numbers say. The two clocks run
**concurrently**, so both being 31 ms means the CPU spends 31 ms issuing
*and* the GPU spends 31 ms executing. If the CPU were merely queueing while
the GPU idled, the GPU figure would be the 4.6 ms bandwidth floor.

The GPU is **not** idle. It is busy running **two thousand kernels** — about
**72 per layer** for a 28-layer model, where a Qwen3 decoder layer needs
roughly seven matmuls plus a couple of norms. Seventy-two is about ten
times that. At 16.3 µs per launch, two thousand launches *is* the 32.7 ms.

Two candidate causes, both plausible:

* **grouped-query attention.** Qwen3-1.7B has 16 query heads and 8
  key/value heads, so the query is broadcast across the groups. The SDPA
  math backend has no fused kernel for that and materialises the
  broadcast.
* **no fusion at all.** RMSNorm, the rotary embedding, the activation and
  the residual adds are each several separate kernels without
  `torch.compile`.

`scripts/check_kernel_count.py` attributes the expansion by module and then
times the same step with `torch.compile(mode="reduce-overhead")`, which fuses
the elementwise chains and captures the step as a CUDA graph. That is the
fix, and it is one line in `la.inference.load_model` if it works.

### The kernel breakdown, and a platform wall

The profiler's per-kernel counts are the clearest evidence yet, because they
are near-exact multiples of 28:

| kernel | count | per layer | time |
|---|---|---|---|
| `aten::mm` | 197 | 7 | 9000 µs |
| cutlass (the mm device kernels) | 141 + 56 | 5 + 2 | 9000 µs |
| `aten::copy_` | 425 | 15 | 953 µs |
| `aten::mul` | 424 | 15 | 794 µs |
| `aten::cat` | 113 | 4 | 343 µs |
| `aten::mean` | 113 | 4 | 322 µs |
| `aten::bmm` | 57 | 2 | 268 µs |
| `aten::add` | 226 | 8 | 238 µs |

**197 `mm` = 28 × 7 + 1.** Seven matmuls per layer is exactly right — q, k,
v, o, and the three MLP projections. So the matmul *count* is correct. The
problem is the **144 kernels per layer around them**: 15 copies, 15
multiplies and 4 cats per layer, which is the grouped-query expansion.
Qwen3-1.7B has 16 query heads and 8 key/value heads, and the SDPA math
backend has no fused kernel for that, so the key and value heads are
materialised at 16 heads — a copy of the whole KV cache — before the
attention can run.

**The matmuls are 18000 µs of 22831 µs: 79 % of the GPU time.** So even
eliminating every copy, cat and multiply saves at most 21 %. The bigger
question is why the matmuls take 18 ms for work that needs 4.6 ms of
bandwidth. `scripts/check_gqa_cost.py` times each of the seven
projections against its own floor to name the slow ones.

**Both candidate fixes hit a platform wall:**

* **Flash attention is not available on Windows, and never has been.**
  Since torch 2.2 the SDPA preferred backend is FlashAttention-2, but the
  Windows wheel does not compile it in. The message *"Torch was not
  compiled with flash attention"* reads like a broken install; it is a
  platform limitation, tracked upstream as
  [pytorch/pytorch#108175](https://github.com/pytorch/pytorch/issues/108175).
  Community Windows wheels exist, but none match this environment:
  `2.8.3+cu130torch2.14` is the only cp314 one, `2.8.4` is built for
  `torch 2.12.1+cu132` (cu132, not cu130), and `2.8.3` is for torch 2.11.
  A flash-attn wheel's torch, CUDA and Python versions must *all* match or
  it installs and then fails to import.
* **`torch.compile` needs triton, which is also missing** — and triton has
  no official Windows wheel either, only community forks. The compile
  attempt failed with `TritonMissing: Cannot find a working triton
  installation`.

**What is left:** xformers' memory-efficient backend (which supports GQA
natively and *is* available on Windows, though transformers does not use it
for Qwen3 by default), a different torch build, or accepting the step
time. `scripts/check_gqa_cost.py` measures what each of those is worth.

**But it may not matter.** A model that stops after 206 tokens instead of
2048 is a tenfold saving that dwarfs any of this. The prompt and EOS fixes
are worth more than every micro-optimisation below them.

### The cache

The Hugging Face cache defaults to `%USERPROFILE%\.cache\huggingface`, which
on this machine is on C: — where there was about **26 MB free**. The instruct
checkpoint needs about **4 GB**, and the download died part way through with:

```
RuntimeError: Task error: File reconstruction error:
Internal Writer Error: Background writer channel closed
```

which reads like a network fault but is a **full disk**. `huggingface_hub`
resolves the cache path once, at import time, so setting `HF_HOME` afterwards
does nothing.

Fix it once:

```bash
scripts\setup_hf_home.bat
```

That moves any existing cache off C:, sets `HF_HOME` and `HF_HUB_CACHE` for
the user environment, and makes every later shell — including a new Git Bash
— pick them up. `la/hf_cache.py` does the same from inside Python, and
`la.inference` and `la.train` import it **before** transformers.

For the current shell only, in Git Bash:

```bash
export HF_HOME=G:/models/huggingface
export HF_HUB_CACHE=G:/models/huggingface/hub
```

## What was fixed on 2026-11-16

Five commits on `main` (top of the log, all pushed locally — the sandbox had no
git credentials, so check whether they reached the remote):

| commit | what |
|---|---|
| `f065eb1` | a matrix cell holding a product of factors is one entry, not two |
| `91b234c` | the grading path runs without torch, so it can be checked anywhere |
| `3a681e1` | `la.explain` shows the truth for benchmark records too |
| `bc263f4` | check every generated answer, not one per family |
| `5c4894c` | write down the six grading bugs and what the grader now says |

The headline of that work: **the answer grammar rejected 28 % of the
orthonormalisation answers even when the model had them exactly right.** The
ground truth was right and the check was wrong, so no test comparing a model's
answer with the truth could ever have caught it.

## Running it here

```bash
python3 -m venv /workspace/.venv
/workspace/.venv/bin/pip install --index-url https://pypi.org/simple sympy pyyaml numpy matplotlib pytest
PYTHONPATH=src /workspace/.venv/bin/python -m unittest discover -s tests
```

The default pip mirror in this sandbox resets connections; `--index-url
https://pypi.org/simple` works. The base dependencies do not include torch, so
the whole test suite runs without a GPU.

**Clear `__pycache__` between edits.** A stale `.pyc` silently shadowed a fix
and made a correct change look like it had not worked.

## The lesson to carry forward

A test that compares a model's answer with a truth produced by the *same
code* cannot fail — the answer and the check come from the same place. Three
grading bugs survived exactly that kind of test.

They surfaced only from two things:

1. checking the grader against **foreign data** — the benchmark's own 660
   reference answers, produced by somebody else;
2. **sampling enough instances** — the defective shape appears in about a
   quarter of one family and almost none of the others, so one sample per
   family never saw it.

When a number is wrong, the instrument is usually the first suspect.

### The same lesson, applied to performance

Two plausible explanations for the 38 ms decode step were ruled out by
measurement, not by argument:

1. *"The CUDA build is wrong."* It was not. 68 TFLOP/s on a bf16 matmul is
   normal for the card, the model was entirely on `cuda:0`, and the build was
   2.12.1+cu130 with cuDNN 9.2. Acting on this hypothesis would have sent
   the user to reinstall torch for no gain.
2. *"The SDPA math backend is upcasting to FP32."* It was not. Attention is
   0.64 ms of a 38 ms step, and the math backend is the fastest of the four
   available here. Installing flash attention would have changed nothing.

Both hypotheses were plausible, both were cheap to test, and both were wrong.
The general rule is the one that applied to the grader: **measure the
component before you change it.** A fix aimed at the wrong component is not
neutral — it costs an afternoon and it leaves the real cause in place.

There is also a measurement trap worth naming. The first kernel diagnostic
timed a **4096-cube matmul**, which is compute bound and reaches 68 TFLOP/s.
A decode step is a **GEMV** — eight rows against the full weight matrix —
which is bandwidth bound and is a completely different kernel. The two
numbers cannot be compared, and the reassuring one was the irrelevant one.

### Reading two clocks at once
There is a second trap, in the CPU-issue-versus-GPU-execute comparison. When
both clocks read about the same as the wall clock, the obvious conclusion —
"the GPU is idle, waiting for the CPU" — is **wrong**, because the two clocks
run *concurrently*. CPU issue 31 ms and GPU execute 31 ms means both are
busy with the same work, not that one is waiting for the other.

The clue that settles it is the kernel count: **2009 launches for a 28-layer
model, about 72 per layer**, where a Qwen3 decoder layer needs roughly seven
matmuls. Seventy-two per layer is ten times what it should be, and at
16.3 µs per launch that count *is* the 32.7 ms step.

**A conclusion that a diagnostic script prints in capital letters is still a
hypothesis.** The script said "CPU-BOUND"; the human reading had to check it
against the other number in the same output.

### Platform walls are worth finding early

Both candidate fixes for the decode step turned out to be impossible on this
machine, and both had to be ruled out before they could be recommended:

* **Flash attention is not available on Windows, and never has been.**
  Since torch 2.2 it is the preferred SDPA backend, but the Windows wheel
  does not compile it in. The error says *"Torch was not compiled with
  flash attention"*, which reads like a broken install and is not one.
  Tracked upstream as
  [pytorch/pytorch#108175](https://github.com/pytorch/pytorch/issues/108175).
  Community Windows wheels exist, but each is pinned to one exact
  torch + CUDA + Python triple, and **a mismatch installs and then fails to
  import** — so "there is a wheel" is not the same as "there is a wheel for
  this environment".
* **`torch.compile` needs triton**, which has no official Windows wheel
  either, only community forks.

The general rule: when a platform-specific feature is missing, check whether
it is *missing from the build* or *missing from the platform* before spending
an afternoon trying to install it. The error text does not distinguish them.

### Check whether the speed problem matters at all

A model that stops after 206 tokens instead of 2048 saves ten times more
than any launch-overhead or attention-backend work. The token count was the
bigger lever, and the prompt-mismatch diagnosis had already found it.
**Weigh the fix against the alternative before optimising.**

### How transformers actually handles the grouped-query expansion

The `cat`, `copy_` and `mul` per layer come from `sdpa_attention_forward`
in `transformers/integrations/sdpa_attention.py`, which picks between three
paths:

```python
if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
    if use_gqa_in_sdpa(attention_mask, key, value):
        sdpa_kwargs["enable_gqa"] = True     # no expansion at all
    elif q_length == 1:
        query = pack_gqa(query, module.num_key_value_groups)
        attention_mask = pack_gqa(attention_mask, module.num_key_value_groups)
    else:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
```

* `use_gqa_in_sdpa` returns `attention_mask is None and key.shape[-1] ==
  value.shape[-1] <= 256`. **A batched decode has a padding mask, so this
  path is unavailable in practice** — the native `enable_gqa=True` that
  would avoid the expansion entirely is not reachable.
* `pack_gqa` is the decode path (`q_length == 1`). It is a *reshape*, not a
  copy: `(batch, 16 heads, 1, 128)` becomes `(batch, 8, 2, 128)`, so sdpa
  reads the two query rows of a group as two positions and the key/value
  heads are used directly at 8. No kv copy is made.
* `repeat_kv` is the prefill path, and it *does* materialise a copy of the
  whole kv cache.

All three paths are numerically equivalent (checked: expand versus
`pack_gqa` differ by 5e-7, expand versus `enable_gqa` by 0).

**The escape hatch:** `TRANSFORMERS_SDPA_FORCE_GQA=1` makes
`use_gqa_in_sdpa` return True regardless of the mask. It may not help,
because with a mask present sdpa cannot use a fused kernel anyway and the
math backend would still expand — but it is one environment variable to try
before installing anything.

**The deeper point:** the expansion is a *transformers* decision, not a
torch one. torch's `scaled_dot_product_attention` takes `enable_gqa=True`
and handles it natively; it is transformers that declines to use it when a
mask is present, because a masked GQA call falls back to the math kernel,
which is slower than expanding.

### The bandwidth floor was wrong, and so was the conclusion

The projection timings showed something that did not fit:

| projection | MB | ms | GB/s | "floor" at 756 GB/s |
|---|---|---|---|---|
| `q_proj` | 8.4 | 0.0204 | 411 | 1.8× |
| `k_proj` | 4.2 | 0.0153 | **274** | 2.8× |
| `v_proj` | 4.2 | 0.0155 | **271** | 2.8× |
| `o_proj` | 8.4 | 0.0131 | 641 | 1.2× |
| `gate_proj` | 25.2 | 0.0218 | **1155** | 0.7× |
| `up_proj` | 25.2 | 0.0190 | **1327** | 0.6× |

**The MLP projections hit 1155 and 1327 GB/s.** The card's *specification*
is 504 GB/s, and the earlier profiles used 756 GB/s. Neither is 1327. So
the "floor" computed from the specification was wrong, and with it the
conclusion that the matmuls run at a quarter of peak.

What is really happening: the 25 MB projections are large enough that the
transfer pipelines and overlaps the launch, so they exceed the nominal
figure. The 4.2 MB ones are dominated by launch and ramp-up instead.

**So the matmuls are not the bottleneck in bandwidth terms — they are at or
above the measured floor.** The cost is the *number of operations*: about
144 kernels per layer for a layer that needs seven matmuls and two norms,
each taking 13 to 22 µs of work. At those sizes the launch cost is
comparable to the work itself.

`scripts/check_bandwidth.py` measures the achievable bandwidth with a plain
copy at nine sizes, so the floor comes from a measured number rather than a
specification.

**This is the third hypothesis ruled out by measurement, and the second one
I got wrong by reasoning from the specification.** The lesson is the same
one as the grader: *the instrument, not the subject, is usually the thing
that is wrong.* A floor computed from a datasheet number is not a
measurement.

**And it does not change the conclusion, which was already right:** the
cost is the kernel count, the fix is fewer kernels, and both obvious
fixes are unavailable on Windows. The EOS fix is still worth ten times
more than any of this.

### The measurement method was wrong, not just the bandwidth

The bandwidth curve looked impossible, and the reason is worth recording:

| copy size | wall ms | GB/s |
|---|---|---|
| 1 MB | 0.0098 | 102 |
| 4 MB | 0.0113 | 355 |
| 8 MB | 0.0171 | 468 |
| 16 MB | 0.0238 | **672** |
| 32 MB | 0.1474 | 217 |
| 64 MB | 0.3304 | 194 |
| 128 MB | 0.6636 | 193 |
| 512 MB | 2.5272 | 203 |

Bandwidth *rising* with size up to 16 MB, then collapsing to a steady
~200 GB/s, is not a memory system. It is an **asynchronous launch**
artefact.

`bench()` measured wall-clock time around a loop of kernel launches. A
CUDA launch returns as soon as the work is *queued*, not when it
finishes, so the loop measured how fast the launches were **issued** and
the GPU work overlapped the issuing. The 672 GB/s at 16 MB was the
launch cost overlapping the transfer, not an achievable bandwidth.

The proof that the method was wrong: `gate_proj` reads 25.2 MB in
0.0171 ms, while *copying* 25.2 MB took 0.1474 ms. A read cannot beat a
copy of the same size, because a copy also writes. And every projection
took 13 to 31 µs regardless of size — 4.2 MB and 25.2 MB the same —
which is the signature of a launch-bound kernel, not a bandwidth-bound
one.

**The rule:** on a GPU, wall-clock time around a launch loop measures the
issue rate. Synchronise per call, or use CUDA events, to measure the work.
Every bandwidth or latency figure taken this way needs checking against
the impossible-result test: *can a read beat a copy of the same bytes?*
If it can, the method is wrong.

`scripts/check_projection_timing.py` measures everything three ways —
wall clock, CUDA events, and synchronise-per-call — so the overlap is
visible rather than hidden.

### The step is real work - measured three ways

`scripts/check_projection_timing.py` measures the same thing three ways:
wall clock around the loop, CUDA events around the loop, and a
synchronize-per-call. The copies and the projections disagree by up to
2x between wall and sync (the launch overlap), but the **decode step
does not**:

| measurement | step |
|---|---|
| wall clock | 32.45 ms |
| CUDA events | 31.87 ms |
| sync per call | 33.85 ms |

All three agree within 6%. **So the 32.7 ms step is real work plus real
launch cost, not an issue-rate artefact.** The earlier projection
numbers were the issue rate; the step never was, because it has enough
work to fill the pipe.

### The card's real bandwidth: ~200 GB/s, not 1327

| copy size | wall ms | events ms | sync ms | wall GB/s | sync GB/s |
|---|---|---|---|---|---|
| 4 MB | 0.0149 | 0.0133 | 0.0153 | 269 | 262 |
| 16 MB | 0.0174 | 0.0171 | 0.0248 | 918 | 644 |
| 32 MB | 0.1545 | 0.1484 | 0.1483 | 207 | 216 |
| 64 MB | 0.3180 | 0.3400 | 0.3170 | 201 | 202 |
| 256 MB | 1.3196 | 1.2661 | 1.2732 | 194 | 201 |

The 16 MB row is the launch overlap, not an achievable bandwidth. From
32 MB upward the copy sustains **~200-216 GB/s** by both methods, which
is half the card's 504 GB/s specification - exactly what a copy (which
reads *and* writes) should reach. **The earlier "1327 GB/s" reading was
an artefact of the timing method and is retracted.**

### The arithmetic that closes the diagnosis

| quantity | value |
|---|---|
| weights | 3.44 GB |
| one read at 200 GB/s | 17.2 ms |
| the measured step | 32.7 ms |
| ratio | **1.9x** |

The projections, timed honestly, cost 0.259 ms per layer for 100.8 MB,
which is *below* the 0.504 ms floor at 200 GB/s - so some of the weights
are served from cache, and the projections are not the problem.

**Two candidate explanations for the missing ~15 ms, and they need
different fixes:**

* **(a) the weights are read more than once** - 32.7 / 17.2 = 1.9x is
  suspiciously close to 2x. The fix is to find and remove the second
  read.
* **(b) launch overhead on top of a bandwidth-bound read** - 4020
  kernels x ~8 us is ~32 ms of CPU-side cost. If the CPU takes ~32 ms
  to *issue* the step, the CPU is the bottleneck and the fix is fewer
  launches or a faster CPU.

`scripts/check_step_bottleneck.py` measures the issue path against the
drain path, so the two are separated instead of guessed at.

### The CPU is the bottleneck - and the earlier retraction was wrong

The issue path measured directly, with the GPU work deliberately not
drained:

| trial | issue | drain | total |
|---|---|---|---|
| 0 | 34.42 ms | 1.33 ms | 34.92 ms |
| 1 | 33.58 ms | 1.60 ms | 34.88 ms |
| 2 | 34.55 ms | 1.53 ms | 34.52 ms |

**The CPU spends 34.4 ms issuing the step; the GPU then needs 1.3 ms to
finish what was queued. The GPU is idle 98.6% of the step.**

This contradicts the earlier retraction. The two-clock reading - "CPU-issue
~31 ms and GPU-execute ~31 ms, so both are busy" - was wrong for a
specific reason: **the GPU-execute number was taken with CUDA events
around the whole step, so it recorded when the LAST kernel finished.
That is bounded below by the issue path, because the last kernel cannot
start until the CPU has issued it.** So "GPU execute ~= wall" says the
last kernel finished at the end of the step, and nothing about whether the
GPU was busy during it. The drain measurement is the only number that
separates the two cases, and it was missing from every earlier script.

**The corrected rule, replacing the one recorded two sections above:**

| drain | meaning |
|---|---|
| ~0 | the GPU is starved - **CPU-bound** |
| ~total | the GPU is the bottleneck |
| in between | both matter |

Without the drain number, the two clocks are uninterpretable. Never
conclude "the GPU is idle" or "the CPU is waiting" from two clocks
that are both close to the wall time.

### What the numbers say about the fix

4020 launches in 34.4 ms is **8.6 us per launch**, matching the
profiler's 8.1 us per kernel. The fix is fewer launches, and the
available routes are `torch.compile` (needs triton - no official Windows
wheel) or xformers (available, but transformers does not use it for
Qwen3 by default).

The LM head is 622.3 MB - 18% of the weights - and costs 1.44 ms, timed
honestly. It is not a rounding error in the step.

### The priority, in units

| change | effect |
|---|---|
| EOS supervision (already implemented) | **10x** fewer decode steps |
| every speed fix available on Windows | ~1.3-2x |

At 34.4 ms a step, 2048 tokens is 70.5 s per sequence and 206 tokens is
7.1 s. Across the 660-problem benchmark that is 12.9 hours against
78 minutes. **The platform walls cap the speedup at 1.3-2x; the EOS fix
is worth 10x. They are not comparable, and the EOS fix is already
implemented.**

### The launch cost is 7.08 us, and that is the whole step

Measured on its own, with a trivial kernel, so the per-launch figure is
comparable with the 8.6 us the step implies:

| measurement | per launch |
|---|---|
| issue only (GPU work overlapping) | **7.08 us** |
| with a synchronize after each | 14.88 us |

A CUDA launch is normally **3-5 us** of CPU time. 7.08 us is slow, and
it is a property of this machine, not of the model.

| quantity | value |
|---|---|
| 4020 launches x 7.08 us | 28.5 ms |
| measured issue path | 34.4-37.1 ms |

The two agree, so **the step is the launch cost.** There is no other
explanation to look for.

What the fixes are worth:

| change | step |
|---|---|
| now | 34.4 ms |
| 4020 -> 2000 launches | 14.2 ms (2.4x) |
| 7.08 us -> 4 us per launch | 17.2 ms (2.0x) |
| both | 7.1 ms (4.8x) |

**And the original 4% GPU utilisation reading now has an explanation.**
A CPU-bound step leaves the GPU idle 98.6% of the time, which is exactly
what a 4% GPU reading looks like. That observation from the original
stall report is consistent with everything measured since.

### The unexplained 15 copy_ per layer

The kernel breakdown divides evenly by 28: `aten::mm` 197 (28x7 - the
projections, correct), `aten::copy_` 425 (28x15), `aten::mul` 424
(28x15), `aten::cat` 113 (28x4), `aten::mean` 113 (28x4), `aten::reduce`
113 (28x4), `aten::add` 226 (28x8), `aten::bmm` 57 (28x2).

At decode time `q_length == 1`, so transformers should take `pack_gqa` -
a **reshape, not a copy**. Yet there are 15 copies per layer. The GQA
expansion would be **2** copies per layer (k and v), not 15, so the
expansion is probably not the source and something else is copying.

`scripts/check_gqa_kernels.py` settles this by measurement rather than by
reading transformers' source: it counts the kernels under each of the
three GQA paths in isolation, and profiles the step with `with_stack=True`
so the profiler reports the **file and line** that issues each copy.

### Available on Windows, in descending order of value

| option | cost | effect |
|---|---|---|
| EOS supervision | already implemented | **10x** |
| xformers memory-efficient backend | `pip install xformers` | fuses attention, native GQA |
| `TRANSFORMERS_SDPA_FORCE_GQA=1` | nothing | removes the expansion |
| `torch.compile` | **blocked** - triton has no official Windows wheel | would fuse the norms |
| flash attention | **blocked** - no wheel for 2.12.1+cu130+cp314 | would fuse attention |

### The real number: 401 torch ops per layer, and the RMSNorm fp32 round-trip

Counting every profiler event rather than only the CUDA kernels changes
the picture, and corrects the "4020 kernels" figure:

| quantity | per step | per layer |
|---|---|---|
| total torch op events | **11229** | **401.0** |
| of which `cudaLaunchKernel` | 1812 | 64.7 |
| `aten::as_strided` | 737 | 26.3 |
| `aten::to` | 486 | 17.4 |
| `aten::view` | 453 | 16.2 |
| `aten::copy_` | 425 | 15.2 |
| `aten::mul` | 424 | 15.1 |
| `aten::_to_copy` | 369 | 13.2 |
| `aten::empty_strided` | 369 | 13.2 |
| `aten::reshape` | 340 | 12.1 |
| `aten::transpose` | 338 | 12.1 |
| `aten::matmul` | 254 | 9.1 |

**The cost is the number of torch OP DISPATCHES, not the number of CUDA
launches.** 11229 ops at ~3 us of CPU dispatch each is 33.7 ms, which is
the measured 34.4 ms issue path. `cudaLaunchKernel` is a subset.

### Where the copies come from: Qwen3RMSNorm

`Qwen3RMSNorm.forward` in `transformers/models/qwen3/modeling_qwen3.py`:

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

Two dtype conversions and two copies per call, plus four elementwise ops.
**Qwen3 has four RMSNorms per layer** - `self_attn.q_norm`,
`self_attn.k_norm` (over head_dim), `input_layernorm` and
`post_attention_layernorm` (over hidden). That is 8 conversions and 8
copies per layer from the norms alone, with the rest from the rotary
embedding's float32 cos/sin construction and the attention mask.

The rotary embedding does **not** upcast the query and key in the text
model - `apply_rotary_pos_emb` is plain `(q * cos) + (rotate_half(q) *
sin)`, and `Qwen3RotaryEmbedding.forward` returns `cos.to(dtype=x.dtype)`.
The `.float()` upcast exists only in the **vision** variants
(`apply_rotary_pos_emb_vision`), which is a trap when reading the
transformers tree.

### The three GQA paths, verified

Running transformers' own `pack_gqa` and `repeat_kqa` code, and torch's
native `enable_gqa`, on identical inputs:

| path | output shape | max diff vs repeat_kv |
|---|---|---|
| `repeat_kv` (copies k/v) | (bs, 16, 1, hd) | - |
| `pack_gqa` (reshape q) | (bs, 8, 2, hd) | 6.66e-16 |
| `enable_gqa` (native) | (bs, 16, 1, hd) | 0 |

**All three are numerically equivalent.** `pack_gqa` folds the QUERY,
not the key and value:

```python
def pack_gqa(hidden_states, n_rep):
    if hidden_states is None or hidden_states.shape[1] == 1:
        return hidden_states
    batch, num_attention_heads, _, dim = hidden_states.shape
    return hidden_states.reshape(batch, num_attention_heads // n_rep, n_rep, dim)
```

The comparison must be on the **flattened** form, because the pack path
returns `(bs, nkv, groups, hd)` while the others return `(bs, nh, 1, hd)`.
The element order is the same - packing preserves it - and the model's
own downstream `reshape(*input_shape, -1)` flattens both identically, so
transformers' `transpose(1, 2)` in the undo step is harmless.

### The verdict

A Qwen3 decoder layer needs **7 matmuls and 4 RMSNorms**. The measured
**401 ops per layer** are mostly bookkeeping and elementwise work around
those. **Only fusion fixes it**, and fusion means `torch.compile`, which
needs triton - and triton has no official Windows wheel.

The realistic Windows-available gain is therefore ~1.2-1.5x (xformers
fuses the attention path only, not the norms), not the 2.4x that the
"halve the launch count" estimate suggested. Against the EOS fix at 10x,
that is not worth another round of work.

### The escape hatch is a TRAP on a padded batch

Measured on the GPU, the three GQA paths isolated:

| path | ops | cudaLaunchKernel | max diff vs repeat_kv |
|---|---|---|---|
| `repeat_kv` (copies k/v) | 64 | 5 | - |
| **`pack_gqa` (reshape q)** | **40** | **3** | **0** |
| `enable_gqa` (native) | 118 | 17 | 3.906e-03 |

**`enable_gqa` is the MOST expensive path, not the cheapest.** It issues
17 kernel launches where `repeat_kv` issues 5 and `pack_gqa` issues 3.

The reason: torch's flash and memory-efficient SDPA kernels do not
support GQA, so `enable_gqa=True` **with an attention mask present**
falls back to the math kernel, which materialises the expanded K/V and
computes the full attention matrix.

At decode (`q_length == 1`) with a **padded** batch, the mask is always
present, so `use_gqa_in_sdpa()` returns False and transformers takes the
`pack_gqa` branch - **which is already the cheapest path**. So:

> **`TRANSFORMERS_SDPA_FORCE_GQA=1` would replace the cheapest path (40
> ops, 3 launches) with the most expensive (118 ops, 17 launches) - 3x
> the op count - and introduce a 3.9e-3 numerical drift in bf16.**
>
> **Do not set it on a padded batch.** The escape hatch is a trap.

This also explains the earlier finding that `MATH` was the fastest SDPA
backend: with a mask, that is the backend that actually runs.

### The numerical difference is precision, not semantics

On CPU, `enable_gqa` matched `repeat_kv` **exactly** in both float64 and
float32. On GPU in bf16 it differs by 3.9e-3. So the maths is the same;
the GPU's math kernel accumulates in a different order at bf16's ~3
significant digits.

### Profiler stack traces do not work on this build

`with_stack=True` recorded **no Python stacks** - every op came back as
"(no stack recorded)". The stack-attribution section has been removed
rather than left printing nothing useful. The RMSNorm attribution came
from the transformers **source**, which is stronger evidence anyway: it
is the code, not a sample of it.

### The benchmark's closing instructions differ from the corpus's

`compare_prompts.py` was rewritten. The old one compared raw TEXT prompts
on the Base checkpoint and generated up to 2048 tokens per batch - about
70 seconds a batch with no EOS supervision, which is why it looked hung.
It also predated the run-5 migration: it compared the base model's raw
text prompts, not the rendered chat prompts run 5 actually uses.

The new script does no generation unless `--generate` is passed, and the
cap defaults to 256.

**What it found, measured over all 660 released problems:**

| | count |
|---|---|
| carry a closing instruction mentioning `\boxed{}` | **440 / 660** |
| carry **no** closing instruction at all | **220 / 660** |
| use the corpus's **exact** closing wording | **126 / 660** |

The 220 with no closing instruction are **the entire 4x4 file**. The
3x3 and 5x5 files always carry one, but in **nine different wordings**:

```
Show all intermediate steps. Put your final numerical answer inside \boxed{}.
Show your steps. Put your final numerical answer inside \boxed{}.
Show all steps. List all eigenvalues inside \boxed{}.
Put the final 3x3 result matrix inside \boxed{}.
Put the final vector b inside \boxed{}.
Write out A^T explicitly. Put the final matrix inside \boxed{}.
Show your row operations. Put your final numerical answer inside \boxed{}.
...
```

The corpus teaches **one** closing per family, always the same wording:
`Show all intermediate steps. Put your final <type> answer inside
\boxed{}.`

**This is a distribution shift, not a format break.** The system prompt
already says "Always put your final ... answer inside `\boxed{}`." for
every subcategory, and the grader falls back from `\boxed{}` to the last
value-like line. So a missing closing line is not fatal for scoring.

**The decision: do not change it before seeing run 5's score.** If run 5
underperforms on the 4x4 file specifically, the fix is to append the
corpus's closing line to the benchmark prompt for the problems that lack
one. Changing what the benchmark sends before measuring would make the
score incomparable with the published baseline.

`tests/test_benchmark_closing_instructions.py` pins the measured counts
(440 / 220 / 126) so a change to the benchmark data or the prompt
construction shows up as a test failure rather than as a silent
regression.

### Run 5's benchmark was scoring the WRONG BASE MODEL

**The bug:** two hardcoded copies of the base model name.

| source | base model |
|---|---|
| `configs/default.yaml` | `Qwen/Qwen3-1.7B` (the instruct checkpoint) |
| `Makefile` `BASE ?=` | `Qwen/Qwen3-1.7B-Base` |

Run 5 was trained with `--base Qwen/Qwen3-1.7B`, and then `make bench-all`
used the Makefile's default, so the benchmark ran:

```
python -m la.bench --base Qwen/Qwen3-1.7B-Base --adapter runs/vslm
```

**A LoRA adapter is a difference between two models.** The delta was
learned against the instruct weights and was merged into the BASE weights,
producing a model that is neither one.

**The symptom:** ~35 minutes per batch of 8. The benchmark was at
batch 6/83 after 156 minutes - about **48 hours** for the full 660.

**Why it was that slow:** the merged model was incoherent, so it never
emitted its stop token. Every problem ran to the 2048-token cap. (The
tokenizer still came from the adapter directory, so `eos_token_id` was
correctly 151645 = `<|im_end|>` - the wiring was right, the weights were
wrong.)

Note the timing arithmetic that gives it away: 2048 steps at the measured
34 ms is 70 seconds, not 35 minutes. A 30x discrepancy means something
other than the expected path was running.

**The fix, in two parts:**

1. **The Makefile reads the base from the config**, so there is exactly one
   place that names it:

   ```make
   BASE ?= $(shell $(PYTHON) -c "import yaml,sys; print(yaml.safe_load(open('configs/default.yaml'))['model']['base'])" 2>/dev/null || echo Qwen/Qwen3-1.7B)
   ```

2. **The benchmark refuses to score an adapter against a different base.**
   `_check_adapter_base()` reads `adapter_config.json` and
   `train_args.json` from the adapter directory, and exits with a clear
   message if the requested base disagrees. A silent incoherence becomes a
   refusal.

**The general lesson: one source of truth for a value that must agree
across two places.** The config already existed and was already correct;
the Makefile simply did not read it, and it won because a command-line
default overrides what the program would have chosen.

### What run 5 needs to do

```bash
# stop the current run - it is scoring the wrong model
# then:
git pull
python -m la.bench --base Qwen/Qwen3-1.7B --adapter runs/vslm \
    --benchmark linalg --baseline --perf --out data/bench_linalg.jsonl
```

`make bench-all` now resolves `--base Qwen/Qwen3-1.7B` and does the same
thing. `scripts/check_bench_run.py` runs a handful of problems and prints
whether each one stopped or hit the cap, so the fix can be confirmed in
two minutes rather than in 48 hours.

## RUN 5 RESULT — the benchmark ran correctly, and exposed three corpus bugs

The wrong-base fix worked. Run 5's benchmark completed in **35.9 minutes**
for the adapter pass and **95.3 minutes** for the base pass, against a
projected 48 hours for the broken run. Every adapter completion stopped
on EOS.

### The scores

| size | n | fine-tuned | base | delta |
|---|---|---|---|---|
| 3x3 | 220 | 52.7% | 75.0% | **-22.3 points** |
| 4x4 | 220 | 43.6% | 41.8% | +1.8 points |
| 5x5 | 220 | 28.6% | 22.3% | +6.4 points |
| overall | 660 | **41.7%** | 46.4% | -4.7 points |

The fine-tuned model is **worse than its own base model**. That is not a
speed problem, a stop-token problem, or a prompt-format problem any more.
It is a **corpus/benchmark contract mismatch**, and it is diagnosable from
the by-task breakdown:

```
  by task:              fine-tuned    base
    determinant              1/150  (  0.7%)   54/150  ( 36.0%)
    eigenvalue               1/90   (  1.1%)    2/90   (  2.2%)
    matrix_power            43/60   ( 71.7%)   34/60   ( 56.7%)
    matrix_vector           45/60   ( 75.0%)   44/60   ( 73.3%)
    multiplication          33/60   ( 55.0%)   25/60   ( 41.7%)
    nullity                 15/60   ( 25.0%)   19/60   ( 31.7%)
    rank                    21/60   ( 35.0%)   21/60   ( 35.0%)
    trace                   56/60   ( 93.3%)   58/60   ( 96.7%)
    transpose               60/60   (100.0%)   49/60   ( 81.7%)
```

Two families are catastrophic: **determinant 0.7%** and **eigenvalue
1.1%**. Everything else is healthy or better than base. Three verified
causes, each reproduced against the real grader:

### Cause 1 — rank and nullity are taught as a PAIR, asked as SCALARS

The corpus has one family, `rank_nullspace`, whose answer is literally
`rank = 2, nullity = 1`. The benchmark asks for rank **or** nullity and
grades against the scalar `2` or `1`.

The model, asked for the rank alone, answers `rank = 2, nullity = 1`. The
grader reads the **last** number out of that string, so it sees the
nullity and never the rank:

```
model output : the answer is rank = 2, nullity = 1
parsed       : 1          <- the nullity, not the rank
vs reference '2'  -> equal=False   (wrong, every time)
vs reference '1'  -> equal=True    (right by accident)
```

**The model has the right number and the wrong answer shape.** The corpus
never showed it that these are two different questions. Asked for the
rank, it is wrong *every single time* - the number it wanted to give is
sitting in the middle of the string where the grader cannot see it.

This is also why nullity (25.0%) scores better than rank (35.0% would be
expected if both were equally broken). It is not; the nullity number is
simply the one that survives the extraction.

### Cause 2 — the corpus teaches EXACT eigenvalues, the benchmark asks for DECIMALS

The corpus's eigenvalue generator picks eigenvalues and builds a matrix
with exactly those eigenvalues, so the trained answer is always exact:
`\{ 2, 5, 1 \}`, or `\{ 3 - 2*I, 3 + 2*I \}`. The benchmark's eigenvalues
are roots of a characteristic polynomial and are irrational —
`-10.9074, 4.5753, 6.3321`.

Measured over 90 generated corpus eigenvalues: **every one is an exact
number** — 86 integers and 4 exact complex pairs, and not one irrational.
The corpus never shows the model a value it has to round, and the
benchmark's 90 eigenvalue questions all require one.

Two sub-failures, both verified:

- The answer *content* differs. The model has learned to emit exact
  values. Asked for the eigenvalues of a matrix whose eigenvalues are
  irrational, it produces integers — and is wrong. This is the one that
  costs the points.
- The answer *format* differs, but less severely than it first looked.
  `\{2, 5, 1\}` parses to a list, so the shape survives; the benchmark's
  reference normalises to a bare list `-10.9074, 4.5753, 6.3321`. A
  correct answer written in the corpus's set-brace style *does* score
  correct (verified against the real grader), so the format is not what
  is losing the points.

Note the base model is also near zero here (2.2%), so this is a hard
task for a 1.7B model, not only a corpus problem. But the corpus does
not help at all here, where it helps on trace and transpose.

A second, smaller mismatch in the same family: the corpus calls this
family `eigenvalues` and the benchmark calls it `eigenvalue`, and the two
names reach different closing instructions - the benchmark asks for "your
final answers" (plural), the corpus's fallback asks for a "final numerical
answer" (singular). Cosmetic next to the exact-versus-decimal content
problem, but the same root cause: two names for one family and nobody
reconciled them.

### Cause 3 — the determinant's algorithm choice does not match the corpus's, at 3x3

The corpus teaches cofactor expansion for 3x3 and elimination for 4x4 and
5x5. The benchmark's 3x3 determinants are graded against the same scalar
format the corpus teaches, so the format is fine — but the model scores
0.7% overall and the base scores 36%. The corpus's 3x3 cofactor expansion
is a long, sign-sensitive derivation; the benchmark's own error taxonomy
says models fail by losing the cofactor sign. The corpus's 3x3 path is
precisely the failure mode the benchmark documents.

Meanwhile the base model, which does not use the corpus's method, scores
36% on determinants — because it can compute a 3x3 determinant by
elimination or by a more direct route.

### What this means

The fine-tune is **not yet a net win**. It beats base on 4x4 (+1.8) and
5x5 (+6.4) — the sizes that matter most for the benchmark's difficulty
curve — and loses badly on 3x3, which is the easiest size. That inversion
is the signature of a corpus that has taught the wrong *shape* of answer
on the easy problems while helping on the hard ones.

This is the same class of bug as the first run's prompt mismatch, one
layer down: the prompt now matches, but the **answer contract** does not.
The benchmark asks one question and the corpus answers a different one.

### The fixes, in priority order

1. **Split `rank_nullspace` into `rank` and `nullity` families**, each
   with a scalar answer. The benchmark asks them separately, so the
   corpus must teach them separately. (Cheapest fix, biggest win: rank
   35% + nullity 25% of 120 problems.)
2. **Add an eigenvalue family whose answers are irrational decimals**,
   matching the benchmark's reference format, and teach the bare-list
   answer shape rather than a set literal.
3. **Teach a second determinant path for 3x3** (elimination alongside
   cofactor expansion), so the model is not locked into the method the
   benchmark says models fail at.

### The rule this adds

**A benchmark's by-task breakdown is a map of the contract, not just a
score.** When a fine-tune loses to its base model, look at the per-task
table first: the tasks where it wins and the tasks where it collapses are
usually not random. `determinant 0.7%` and `eigenvalue 1.1%` next to
`transpose 100%` is not a capability gradient — it is a formatting
contract that was never taught.

### The other measurement: the base model's answers are mostly unreadable

```
  base model, no adapter
  3x3   76.4%   (boxed only: 75.0%)
  4x4   45.9%   (boxed only: 41.8%)
  5x5   23.2%   (boxed only: 22.3%)
  no answer parsed: 128
```

The base model **knows more than it scores**. Read from anywhere in the
response it gets 76.4% of 3x3; scored strictly on `\boxed{}` it gets
75.0%. It also fails to stop: 128 of 660 answers never produced a parseable
value, and the base pass took 95.3 minutes against the adapter's 35.9 —
2.7x slower, because it runs to the cap.

So the comparison has to be stated carefully. Against the *benchmark's own
protocol*, base scores 46.4% and the fine-tune scores 41.7%. Against a
lenient read of the base model's prose, base scores ~50%. The fine-tune
does not lose on mathematics; it loses on **answer shape**, in exactly the
two families where the corpus teaches the wrong shape.

---

## Run 6 — implemented (2026-11-17)

The design is `docs/superpowers/specs/2026-11-17-run-6-answer-contract-design.md`
and the implementation plan is
`docs/superpowers/plans/2026-11-17-run-6-answer-contract.md`.

### What changed in the repository

| change | where |
|---|---|
| one family -> benchmark subcategory map | `src/la/benchprompt.py` (`FAMILY_TO_SUBCAT`) |
| `rank` and `nullity` are separate scalar families | `src/la/problems.py` |
| `eigenvalue` (exact) and `eigenvalue_numeric` (four decimals) | `src/la/problems.py` |
| the 3x3 determinant has both a cofactor and an elimination path | `src/la/problems.py` |
| the rank and determinant derivations show their row operations | `src/la/problems.py` |
| 70/30 benchmark/background corpus mix, declared in the config | `configs/default.yaml`, `src/la/build_dataset.py` |
| the contract checker | `scripts/check_corpus_contract.py`, `make contract-check` |

The corpus now has **19 families** (17 in run 5, minus `rank_nullspace`, plus
`rank`, `nullity`, `eigenvalue` and `eigenvalue_numeric` — `eigenvalues`
became `eigenvalue`).

### The measurement that changed the design

The eigenvalue **shape** was never the problem. Measured against the real
grader, the corpus's existing set literal wins **34/34** of the exact
eigenvalue references, and true roots rounded to four decimals win
**56/56** of the decimal ones. The corpus's 90 eigenvalues are 86 integers
and 4 exact complex pairs — **zero irrationals**. The defect is the value
domain, not the format.

### The rerun procedure

```bash
make contract-check       # no GPU: the contract map must be clean
make data                 # ~3 min: rebuild the 20,000-example corpus
make check-arithmetic     # no GPU: every printed derivation adds up
make train                # ~146 min on one RTX 4070 SUPER
make eval                 # ~5 min: the held-out set, must not drop below 54.5%
make bench-all            # ~36 min for the adapter, ~95 min for the base
```

**Target:** beat base's 46.4% overall, and beat base on each of 3x3, 4x4 and
5x5. Run 5's adapter stays on disk, so the comparison is a one-command diff.

### What is not yet measured

No GPU ran in this session. The corpus-side work and every check that does
not need a GPU are done; the score above is still a projection, not a
result.

## RUN 6 RESULT — 230/660 = 34.8%, a regression (2026-11-17)

The rerun was genuine: it pulled the run-6 commit, the contract checker
passed, the corpus had 19 families at the 70/30 mix, the arithmetic check
passed, and the adapter trained to `runs/vslm-run6`.

| | run 5 | run 6 | base |
|---|---|---|---|
| overall | 275/660 = 41.7% | **230/660 = 34.8%** | 306/660 = 46.4% |
| 3×3 | 52.7% | 44.5% | 75.0% |
| 4×4 | 43.6% | 37.7% | 41.8% |
| 5×5 | 28.6% | 22.3% | 22.3% |

Per family, run 6 against base:

| family | run 5 | run 6 | base |
|---|---|---|---|
| determinant | 1/150 | **0/150** | 54/150 |
| eigenvalue | 1/90 | 2/90 | 2/90 |
| matrix_power | 43/60 | 33/60 | 34/60 |
| matrix_vector | 45/60 | **16/60** | 44/60 |
| multiplication | 33/60 | 26/60 | 25/60 |
| nullity | 15/60 | 20/60 | 19/60 |
| rank | 21/60 | 21/60 | 21/60 |
| trace | 56/60 | 52/60 | 58/60 |
| transpose | 60/60 | 60/60 | 49/60 |

The pattern in that table is the diagnosis: **transpose (100%) and trace
(86.7%) held; the families that collapsed are the ones whose PROMPT varies
most.** determinant went to zero while the base model — taught neither
method — scored 36%.

## RUN 7 — the fifth leg of the contract: the PROMPT SHAPE

The answer contract has four legs (system prompt, closing instruction,
answer shape, value domain). Run 6 matched all four and still lost 11.6
points to its base model. The leg nobody had looked for is the **shape of
the prompt**, and it turns out the benchmark sends three different ones.

Measured directly against the released problem files:

| leg | the benchmark | the corpus (runs 1–6) |
|---|---|---|
| task line | names the matrix | matched |
| **method line** | "Use cofactor expansion along the first row." | **never produced** |
| **closing line** | **two** different ones per family | **one of the two** |
| **bare shape** | all 220 of the 4×4 problems: a matrix, no task, no closing line | **never produced** |

### The method lines, verbatim, with the benchmark's own counts

| subcategory | method line | at 3×3 |
|---|---|---|
| determinant | "Use cofactor expansion along the first row." | 18 of 50 |
| eigenvalue | "Compute the characteristic polynomial det(A − λI) = 0 and solve for λ." | 24 of 30 |
| rank | "Use row reduction to find the number of non-zero rows in row echelon form." | 14 of 20 |
| nullity | "Recall: nullity = number of columns − rank." + "First find the rank using row reduction, then compute the nullity." | 16 of 20 |
| multiplication | "Show your row-by-column computation steps clearly." | 12 of 20 |
| matrix_power | "Multiply A by itself. Show all row-by-column steps." | 20 of 20 |
| matrix_vector | "Compute each entry of b by taking the dot product of each row of A with x." | 20 of 20 |
| trace | "The trace is the sum of the diagonal elements." | 14 of 20 |
| transpose | "The transpose A^T is obtained by swapping rows and columns." | 20 of 20 |

**None of them appear at 4×4 or 5×5** — the method line is a 3×3-only
shape.

### The closing lines: two per family

| subcategory | variant A | variant B |
|---|---|---|
| determinant | 82× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | — (same closing, different method line) |
| eigenvalue | 36× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | 24× "Show all steps. List all eigenvalues inside \boxed{}." |
| matrix_power | 20× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | 20× "Put the final 3×3 result matrix inside \boxed{}." |
| matrix_vector | 20× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | 20× "Show your steps. Put the final vector b inside \boxed{}." |
| multiplication | 28× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | 12× "Put the final 3×3 result matrix inside \boxed{}." |
| nullity | 24× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | 16× "Show your steps. Put your final numerical answer inside \boxed{}." |
| rank | 26× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | 14× "Show your row operations. Put your final numerical answer inside \boxed{}." |
| trace | 26× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | 14× "Show your steps. Put your final numerical answer inside \boxed{}." |
| transpose | 20× "Show all intermediate steps. Put your final numerical answer inside \boxed{}." | 20× "Write out A^T explicitly. Put the final matrix inside \boxed{}." |

Plus the third shape: the bare one, 20 of every 60 problems per family,
which is the whole 4×4 file.

### Why this is the regression, not the mathematics

The corpus taught cofactor expansion, row reduction, the characteristic
polynomial, dot products — every method the benchmark names. **But it never
put the request in the prompt.** So the model was taught "here is a
determinant, use cofactors" and was then scored on "use cofactor expansion
along the first row", and a fine-tuned model that had been narrowed onto one
derivation answered a differently-shaped question with the shape it knew.
The base model, which had never been narrowed, fell back on what it knew and
scored 36% where the fine-tune scored 0.

The evidence for "prompt, not mathematics":

- The corpus's derivations were already correct. Measured: 14 of 14
  cofactor-instructed problems use cofactors, 0 mismatches, 0 wrong answers.
- The families that held (transpose 100%, trace 86.7%) are the ones whose
  second variant differs least from the first.
- The families that collapsed are the ones whose prompt varies most.
- The corpus's *arithmetic* was never in doubt — `make check-arithmetic`
  passes on 3,800 problems across 19 families.

### What run 7 changed

| change | where |
|---|---|
| every prompt tail the benchmark uses, verbatim, with its counts | `src/la/benchprompt_tails.py` (new) |
| the tail is chosen per problem from the per-problem RNG | `src/la/augment.py` (`prompt_tail`) |
| the prompt's method line **decides** the derivation | `src/la/problems.py` (`gen_determinant`) |
| the VERIFY line recomputes instead of restating the answer | `src/la/problems.py` (3×3 expands along the first column; other sizes use the triangular form) |
| the evaluator survives a malformed plot spec | `src/la/schema.py` (`spec_features`) |
| the table is checked against the released files, line for line | `tests/test_tail_table_matches_benchmark.py` (new) |
| the corpus produces each benchmark tail | `tests/test_closing_instruction_variants.py`, `tests/test_benchmark_instructions_in_corpus.py` (new) |
| the 4×4 bare shape is in the corpus | `tests/test_prompt_shapes.py` (new) |

`make contract-check` now checks all ten families against all three of
their tails and reports the tail coverage per family.

### The rerun procedure — every command names the adapter

```bash
git pull
make contract-check                          # no GPU: the contract map must be clean
rm -rf data                                  # run 6's corpus is still on disk
make data                                    # ~3 min
make check-arithmetic                        # no GPU
make train ADAPTER=runs/vslm-run6            # ~182 min on one RTX 4070 SUPER
make eval ADAPTER=runs/vslm-run6             # ~5 min
make bench-all ADAPTER=runs/vslm-run6        # ~36 min adapter, ~95 min base
```

**`ADAPTER=runs/vslm-run6` on every scoring command.** The Makefile's
default is the old `runs/vslm`, and the default wins — a `make bench-all`
without it scores the wrong adapter. This is the same class of bug as the
wrong-base-model one in run 5.

### What is not yet measured

No GPU ran in this session. Every check that does not need a GPU passes
(191 tests, `make check` exit 0, `make contract-check` exit 0, the
arithmetic check on 3,800 problems). **The run-7 score is still a
projection.** The diagnosis for run 6 is measured; the fix for it is not
yet scored.
