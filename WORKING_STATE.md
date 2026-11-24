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

## RUN 7 RESULT — 282/660 = 42.7%, up from 34.8% (2026-11-20)

The rerun was genuine: it pulled `2bfe297`, the contract checker reported
**3/3 tails for all ten families**, the corpus had 19 families at the 70/30
mix, the arithmetic check passed, and the adapter trained to
`runs/vslm-run6`.

| | run 5 | run 6 | **run 7** | base |
|---|---|---|---|---|
| overall | 275/660 = 41.7% | 230/660 = 34.8% | **282/660 = 42.7%** | 306/660 = 46.4% |
| 3×3 | 52.7% | 44.5% | **54.5%** | 75.0% |
| 4×4 | 43.6% | 37.7% | **48.6%** | 41.8% |
| 5×5 | 28.6% | 22.3% | **25.0%** | 22.3% |

Per family, run 7 against base:

| family | run 5 | run 6 | **run 7** | base | run 7 − base |
|---|---|---|---|---|---|
| determinant | 1 | 0 | **4** | 54 | **−50** |
| eigenvalue | 1 | 2 | 5 | 2 | +3 |
| matrix_power | 43 | 33 | 40 | 34 | +6 |
| matrix_vector | 45 | 16 | 37 | 44 | −7 |
| multiplication | 33 | 26 | 33 | 25 | +8 |
| nullity | 15 | 20 | 19 | 19 | 0 |
| rank | 21 | 21 | 29 | 21 | +8 |
| trace | 56 | 52 | 56 | 58 | −2 |
| transpose | 60 | 60 | 59 | 49 | +10 |

**The prompt-shape fix worked.** Every family except determinant improved,
4×4 went from 37.7% to 48.6% (**+6.8 over base**), 5×5 went from 22.3% to
25.0% (**+2.7 over base**), and the fine-tune now beats base on two of the
three sizes. The families whose prompt varied least are the ones that held
or improved, exactly as the diagnosis predicted.

**But the overall score is still 24 problems behind, and one family is the
whole of it.**

## RUN 8 — the arithmetic the benchmark asks for

### The decisive number

Remove the determinant family and the fine-tune **wins**:

    run 7 without determinant = 278/510 = 54.5%
    base  without determinant = 252/510 = 49.4%

One family — 4/150 against the base model's 54/150 — is the entire
24-problem deficit. Everything else is competitive or better.

### What was actually wrong

The prompt shape was fixed in run 7 and the corpus's determinant
derivations are arithmetically correct (`make check-arithmetic` passes on
3,800 problems, and 14 of 14 cofactor-instructed problems use cofactors
with 0 wrong answers). The problem was the **arithmetic load**:

    corpus 3x3 determinant matrix entries:   [-4, 5]
    benchmark 3x3 determinant matrix entries: [-9, 9]

    corpus 3x3:    cofactor term a_1j * M_1j, median  20
    benchmark 3x3:                              median  60, p90 360

    corpus 5x5: 82% of derivations contain a fraction whose numerator is
    >= 100, the largest seen at 4009; the benchmark's 5x5 determinants
    reach 9112.

The corpus taught the model to emit a long chain of exact arithmetic — nine
numbered steps at 3×3, and at 5×5 a fraction cascade like `807/118` — and
then asked for a bigger number than the chain was trained on. **One slip
loses the whole problem.** The base model was never taught that chain, so
it answers directly and lands the number more often.

The benchmark's entry distributions have a TAIL, and the corpus's old range
was the small end with none of it:

| family | size | median \|entry\| | p90 | max |
|---|---|---|---|---|
| rank | 3×3 | 3 | 8 | 18 |
| rank | 4×4 | 3 | 9 | 31 |
| nullity | 3×3 | 4 | 12 | 24 |
| nullity | 4×4 | 4 | 12 | 32 |
| eigenvalue | 3×3 | 6 | 80 | 240 |
| eigenvalue | 4×4 | 3 | 20 | 1120 |
| determinant | 3×3 | 8 | 9 | 9 |

### What run 8 changed

| change | where |
|---|---|
| the benchmark's entry SHAPE per family and size | `src/la/benchprompt_magnitudes.py` (new) |
| the corpus draws its matrices with that shape | `src/la/problems.py` (`draw_entry`, `_rand_int_matrix`) |
| rank and nullity generate 4×4 and 5×5 as well as 3×3 | `src/la/problems.py` (`_rank_problem`) |
| the rank plot changes with the size | `src/la/problems.py` (`_rank_plot`) |
| one eigenvalue problem in four is large-entry and rank-deficient | `src/la/problems.py` (`_large_entry_eigenvalue_matrix`) |
| the eigenvalues come from the rational root theorem, not `simplify()` | `src/la/problems.py` (`_exact_eigenvalues`) |
| the magnitudes are checked against the released files | `tests/test_entry_range_coverage.py` (new) |

Three details worth keeping:

1. **The large-entry eigenvalue matrix is built as `S D S⁻¹`** with `S` an
   integer unimodular matrix, so the entries are large and integral while
   the spectrum stays small and simple. A *random* large matrix produces an
   irreducible characteristic polynomial whose roots come out as
   cubic-root expressions — the benchmark never asks for those, and a
   corpus that teaches cube roots teaches the wrong arithmetic.

2. **`simplify()` on a large-entry 5×5's eigenvalues took 2.1 seconds.**
   The rational root theorem gives the same answer in milliseconds: the
   eigenvalue family went from **1394 ms per problem to 9 ms**, a 150×
   speedup that keeps the corpus build at about 1.8 minutes on four
   workers.

3. **A null space of four or five dimensions cannot be drawn in three.**
   The rank and nullity families now generate at all three sizes, and the
   plot changes: 3×3 draws the null space, 4×4 and 5×5 draw the echelon
   form — which is the thing the derivation reduces to, and whose pivot
   count *is* the rank.

### Verification

`make check` exit 0 · `make contract-check` exit 0 · **196 tests pass** ·
the arithmetic check passes on 3,800 problems across 19 families · the
eigenvalue answers were checked against the matrices' true spectra on 300
generated problems, 0 wrong.

### The rerun procedure — every command names the adapter

```bash
git pull
make contract-check                          # no GPU: the contract map must be clean
rm -rf data                                  # the previous corpus is still on disk
make data                                    # ~2 min
make check-arithmetic                        # no GPU
make train ADAPTER=runs/vslm-run6            # ~180 min on one RTX 4070 SUPER
make eval ADAPTER=runs/vslm-run6             # ~31 min
make bench-all ADAPTER=runs/vslm-run6        # ~41 min adapter, ~94 min base
```

**`ADAPTER=runs/vslm-run6` on every scoring command.** The Makefile's
default is the old `runs/vslm`, and the default wins — a `make bench-all`
without it scores the wrong adapter. This is the same class of bug as the
wrong-base-model one in run 5.

### What is not yet measured

No GPU ran in this session. Every check that does not need a GPU passes.
**The run-8 score is still a projection.** The run-7 diagnosis is
measured; the fix for it is not yet scored.

## RUN 8 RESULT — 250/660 = 37.9%, a regression from 42.7% (2026-11-21)

| | run 5 | run 6 | run 7 | **run 8** | base |
|---|---|---|---|---|---|
| overall | 275 = 41.7% | 230 = 34.8% | 282 = 42.7% | **250 = 37.9%** | 306 = 46.4% |
| 3×3 | 52.7% | 44.5% | 54.5% | 46.8% | 75.0% |
| 4×4 | 43.6% | 37.7% | 48.6% | 43.6% | 41.8% |
| 5×5 | 28.6% | 22.3% | 25.0% | 23.2% | 22.3% |

Per family:

| family | run 7 | **run 8** | base | Δ |
|---|---|---|---|---|
| determinant | 4 | **1** | 54 | −3 |
| eigenvalue | 5 | 4 | 2 | −1 |
| matrix_power | 40 | 39 | 34 | −1 |
| matrix_vector | 37 | **17** | 44 | **−20** |
| multiplication | 33 | 28 | 25 | −5 |
| nullity | 19 | 17 | 19 | −2 |
| rank | 29 | 27 | 21 | −2 |
| trace | 56 | **59** | 58 | **+3** |
| transpose | 59 | 58 | 49 | −1 |

### The harness is deterministic — so this is real

The base model was scored in all three runs and gave **306/660 with
byte-identical per-family numbers every time**. There is no measurement
noise: every difference between runs is a property of the adapter.

### The magnitude fix DID work, and it still lost

The determinant's cofactor term `a₁ⱼ · M₁ⱼ`:

| | median | p90 | max |
|---|---|---|---|
| run 7 corpus | 18 | 56 | 165 |
| **run 8 corpus** | **70** | **195** | **288** |
| benchmark | 60 | 360 | 1170 |

The corpus's determinant arithmetic moved onto the benchmark's scale, and
the determinant score went **4 → 1** anyway. So "match the benchmark's
magnitudes" was necessary and **not sufficient**.

### What actually went wrong: the corpus got HARDER to fit

| | loss @ step 200 | loss @ step 400 | score |
|---|---|---|---|
| run 7 | 0.0542 | 0.0536 | 42.7% |
| **run 8** | **0.0692** | **0.0784** | **37.9%** |

The loss is 1.28–1.46× higher at every step after 40, and **the loss and
the score move together across all four runs**. The model fits run 8's
corpus worse, and scores worse on it.

The corpus-side cause, measured on the generated data:

| family | run 7 | run 8 | why |
|---|---|---|---|
| rank | 28 ops (3×3 only) | 32 (3×3), 55 (4×4), 63 (5×5) | 4×4 and 5×5 are now generated, and a 5×5 reduction is more than twice the work |
| nullity | 33 ops (3×3 only) | 36, 47, 67 | the same |
| eigenvalue | 0 big numbers per problem | 1.4 | the large-entry matrices |
| determinant | 4.1 big numbers | 5.6 | the wider entries |
| trace | — | — | unchanged, and it is the **only family that improved** |

**The families whose corpus got harder lost points. The family whose corpus
did not change, trace, gained them.** Widening the distribution spends the
model's capacity on the hard tail, and that costs more on the easy
majority than it gains on the tail.

### A second defect: the change perturbed the ENTIRE corpus

`draw_entry()` consumes **two** RNG draws per entry — one `random()` for
the heavy/light decision and one `randint()` — where the old
`rng.randint()` consumed **one**. The builder seeds per `(seed, index)`,
so every matrix after the first one in **every family** changed:

- only **24%** of rows were identical at the same index;
- `matrix_vector`, whose generator was not deliberately changed, ended up
  with a completely different set of matrices — and it lost 20 points,
  the largest single move in the run.

**This is the lesson: a change to how randomness is consumed silently
invalidates every A/B comparison built on "the other family's data was
unchanged".** Comparing per-family *distributions* would have said
"matrix_vector is the same" — comparing the actual values says it is
completely different.

### What run 8 should have been

Run 8 changed three things at once — the magnitudes, the rank/nullity
sizes, and the RNG consumption — and the corpus got harder on every
axis. The magnitude fix was right *in isolation*: the determinant's
arithmetic moved onto the benchmark's scale. It was wrong as a *package*,
because the same commit made rank and nullity roughly twice as long.

The next run has to change **one** thing, and it has to be measured on the
**loss curve** — which arrives about 180 minutes before the benchmark does.

### The rules this adds

1. **The base model is the control.** Score it in every run. If its
   numbers move, the harness is not deterministic and no adapter
   comparison means anything.

2. **The loss curve is the early warning.** It moves with the score and
   it arrives three hours earlier. A corpus change that raises the loss
   will lower the score.

3. **Matching the benchmark's magnitudes is necessary, not sufficient.**
   The question is not "does the corpus contain the benchmark's range"
   but "does the model get MORE RIGHT overall".

4. **One change per run.** Run 8 changed the magnitudes, the rank/nullity
   sizes and the RNG consumption together, so its result cannot be
   attributed to any one of them.

### The rule this adds

**A corpus has to match the benchmark's MAGNITUDES, not just its format.**
The prompt shape was the fifth leg of the contract; the arithmetic load is
the sixth. A corpus that teaches "here is a determinant, now compute it"
has to teach it at the size of number the benchmark will ask for, or it
has taught a chain that cannot reach the answer.

And the diagnostic that found it: **when one family is far below its
neighbours, compute the score WITHOUT that family.** If the fine-tune
beats the base model on everything else, the deficit is that family and
nothing else, and the whole problem reduces to one family. That single
subtraction turned a 24-problem mystery into a 50-problem question about
the determinant.

---

## RUN 9 — the size mix was never checked (2026-11-21)

### The comparison that had to be redone first

The first corpus comparison of run 7 against run 8 was **invalid**. The
`la` package is installed in the virtualenv as an editable install
pointing at the MAIN repository, so running `python -m la.build_dataset`
from a git worktree at `2bfe297` silently used run 8's code. The
comparison said "94.7% of the rows are identical" and concluded that
run 8's change was a near no-op.

Rebuilding with `PYTHONPATH=<worktree>/src` gives the real picture:

| family | run 7 sizes | run 8 sizes |
|---|---|---|
| rank | 3×3 only (85) | 3×3 53, 4×4 15, 5×5 16 |
| nullity | 3×3 only (85) | 3×3 47, 4×4 19, 5×5 18 |
| eigenvalue | 2×2 38, 3×3 32, 4×4 10, 5×5 5 | 2×2 32, 3×3 28, 4×4 14, 5×5 10 |

Only **24.2%** of rows were identical at the same index. Run 8's corpus
was a completely different sample, not a shifted one.

**The lesson:** an editable install in the environment makes a worktree
comparison meaningless unless the import path is pinned. Check where the
module was actually loaded from before believing any A/B result.

### The harness is deterministic

The base model scored **306/660 with byte-identical per-family numbers**
in runs 6, 7 and 8. There is no measurement noise, so every difference
between the runs is a property of the adapter.

### What run 8 actually did

Three changes in one commit, and the corpus got harder on every axis:

| | loss @ step 200 | score |
|---|---|---|
| run 7 | 0.0542 | 42.7% |
| run 8 | 0.0692 | 37.9% |

| family | run 7 ops | run 8 ops |
|---|---|---|
| rank | 28 (3×3 only) | 32 (3×3), 55 (4×4), 63 (5×5) |
| nullity | 33 (3×3 only) | 36, 47, 67 |

The families whose corpus got harder lost points. `trace`, whose corpus
did not change, gained three. **The training loss is the early warning,
and it arrives three hours before the benchmark does.**

### The magnitude fix worked — and the corpus overshot

The determinant's cofactor term `a₁ⱼ · M₁ⱼ`:

| | median | p90 | max |
|---|---|---|---|
| run 7 corpus | 18 | 56 | 165 |
| run 8 corpus | 70 | 195 | 288 |
| benchmark | 60 | 360 | 1170 |

The corpus's determinant arithmetic moved onto the benchmark's scale and
the determinant score went **4 → 1** anyway. Matching the magnitudes was
necessary and not sufficient.

And the eigenvalue family's corpus **overshot the benchmark by 23×**:

| | median | p90 | max |
|---|---|---|---|
| benchmark | 3 | 12 | 1120 |
| run 8 corpus | 2 | **276** | **11,080** |

The cause: the large-entry eigenvalue matrix is built as `S D S⁻¹` and
then scaled by a blind factor `k ∈ {6, 8, 12, 20}`. `|S D S⁻¹|` is
unbounded — `S⁻¹`'s entries grow with `n` — so the product reached
11,080 against the benchmark's 1,120.

### THE CONTRACT LEG THAT WAS NEVER CHECKED: THE SIZE MIX

Measuring every matrix in the released benchmark files against the
corpus's generated data turned up a leg that had never been checked.

**The benchmark's size mix is EXACTLY UNIFORM, for every family, and
there is not one 2×2 matrix anywhere in all 660 problems:**

| subcategory | benchmark | corpus (run 8) |
|---|---|---|
| determinant | 3×3:50 4×4:50 5×5:50 | 2×2:34 3×3:49 4×4:53 5×5:64 |
| eigenvalue | 3×3:30 4×4:30 5×5:30 | 2×2:80 3×3:75 4×4:27 5×5:18 |
| matrix_power | 3×3:20 4×4:20 5×5:20 | 2×2:43 3×3:102 4×4:55 |
| matrix_vector | 3×3:20 4×4:20 5×5:20 | 2×2:33 3×3:65 4×4:69 5×5:33 |
| multiplication | 3×3:40 4×4:40 5×5:40 | 2×2:43 3×3:81 4×4:76 |
| nullity | 3×3:20 4×4:20 5×5:20 | 3×3:94 4×4:53 5×5:53 |
| rank | 3×3:20 4×4:20 5×5:20 | 3×3:89 4×4:46 5×5:65 |
| trace | 3×3:20 4×4:20 5×5:20 | 2×2:19 3×3:58 4×4:67 5×5:56 |
| transpose | 3×3:20 4×4:20 5×5:20 | 2×2:36 3×3:79 4×4:50 5×5:35 |

**Between 10 and 40 per cent of the corpus taught 2×2 problems the
benchmark never scores.** Those are rows NOT spent on 5×5 — and a 5×5
derivation is a different KIND of task: more row operations, more
arithmetic, more chances to slip. The corpus under-taught exactly the
size where the benchmark's arithmetic is hardest, and the determinant
family — where run 7 scored 4 against the base's 54 — is the family
whose 5×5 cofactor products reach 9,112.

### Run 9's changes

1. **The size mix is the benchmark's.** A single `_bench_size()` helper
   draws 3, 4 or 5, and every benchmark family uses it. No 2×2.
2. **The eigenvalue corpus stops overshooting.** The large-entry matrix
   is scaled so its largest entry LANDS ON a target drawn from the
   benchmark's own per-size shape — and when the base matrix is already
   too large it is scaled DOWN, which run 8's `max(1, target // peak)`
   silently refused to do.
3. **`CRootOf` answers are gone.** A matrix whose characteristic
   polynomial is irreducible is REJECTED rather than answered: sympy
   returns a `CRootOf` for it, the answer grammar cannot read it, and the
   grader would fail on it. The benchmark's answers are all plain
   decimals.
4. **The numeric eigenvalue family is built symmetric**, because a
   random integer matrix has a real spectrum only 2% of the time at 5×5
   (98% have a complex conjugate pair) while the benchmark's are all
   real. A symmetric matrix has all-real eigenvalues, always.
5. **Generation is 20× faster.** `sp.nsimplify()` ran mpmath's PSLQ on
   expressions that were already exact — 13.8 s of a 21.7 s run. The
   exactification now skips the search for an already-exact value, and
   the eigenvalue generator went from 1,096 ms to 54 ms per problem.
6. **Each generator owns its random stream**, pinned by
   `_rng_for(seed, index)`. Run 8's change to `draw_entry` consumed two
   draws per entry where the old code consumed one, and that rewrote
   76% of the corpus — including every matrix in `matrix_vector`, the
   family that lost 20 points.

### The tests that now guard all of this

- `tests/test_size_mix.py` — the corpus's size mix against the released
  benchmark files, in both directions.
- `tests/test_corpus_stability.py` — the corpus must not overshoot the
  benchmark's magnitudes; the arithmetic load must stay in a band; each
  generator's random stream must be independent.

Both fail on run 8's code, with the measured numbers in the message.

### The size mix, measured after run 9's changes

| family | benchmark | corpus before | corpus after |
|---|---|---|---|
| determinant | 3:50 4:50 5:50 | 2:34 3:49 4:53 5:64 | 3:42 4:37 5:41 |
| eigenvalue | 3:30 4:30 5:30 | 2:80 3:75 4:27 5:18 | 3:40 4:49 5:31 |
| eigenvalue_numeric | 3:30 4:30 5:30 | — | 3:40 4:35 5:45 |
| matrix_power | 3:20 4:20 5:20 | 2:43 3:102 4:55 | 3:39 4:38 5:43 |
| matrix_vector | 3:20 4:20 5:20 | 2:33 3:65 4:69 5:33 | 3:34 4:45 5:41 |
| rank | 3:20 4:20 5:20 | 3:89 4:46 5:65 | 3:39 4:38 5:43 |
| nullity | 3:20 4:20 5:20 | 3:94 4:53 5:53 | 3:42 4:37 5:41 |
| trace | 3:20 4:20 5:20 | 2:19 3:58 4:67 5:56 | 3:46 4:36 5:38 |
| transpose | 3:20 4:20 5:20 | 2:36 3:79 4:50 5:35 | 3:40 4:39 5:41 |
| multiplication | 3:40 4:40 5:40 | 2:43 3:81 4:76 | 3:40 4:33 5:47 |

No 2×2 anywhere, and every size the benchmark scores is covered.

### The eigenvalue magnitudes, measured

| | median | p90 | max |
|---|---|---|---|
| benchmark (pooled) | 3 | 12 | 1120 |
| run 8 corpus | 2 | **276** | **11,080** |
| run 9 corpus | 1 | **9** | **1120** |

Per size, against the benchmark's own per-size figures:

| | benchmark med / p90 / max | run 9 corpus |
|---|---|---|
| 3×3 | 6 / 80 / 240 | 2 / 96 / 240 |
| 4×4 | 3 / 20 / 1120 | 1 / 138 / 1120 |
| 5×5 | 3 / 6 / 8 | 1 / 5 / 9 |

### Three more defects run 9 fixed

1. **`CRootOf` answers.** 2 per cent of the corpus's eigenvalue answers
   were `CRootOf(x**5 + 2*x**4 + ...)`, which the answer grammar cannot
   read and the grader cannot score. The cause was `_exact_eigenvalues`
   FALLING BACK to sympy's `eigenvals()` when the rational root theorem
   could not find every root. A matrix whose characteristic polynomial is
   irreducible is now rejected, not answered.

2. **The decimal eigenvalue family had almost no 5×5.** A random integer
   matrix has a real spectrum only 2 per cent of the time at 5×5 — 98 per
   cent have a complex conjugate pair — while all 90 of the benchmark's
   eigenvalue answers are real decimals. The family is now built
   symmetric, which has all-real eigenvalues always.

3. **The corpus exceeded the sequence cap.** The 5×5 matrix products and
   powers ran to 1,818 tokens against the 1,536 cap, and a truncated
   example loses the end of its completion, which is where the plot spec
   is. The fix keeps the partial sums — the run with the sums removed
   scored 27 per cent on multiplication and 0 on powers — and drops the
   product list and the repeated result matrix at the large sizes only.
   The longest example is now 1,242 tokens.

### Generation is 20× faster

| | before | after |
|---|---|---|
| eigenvalue | 1,096 ms | 54 ms |
| eigenvalue_numeric | 122 ms | 13 ms |
| the whole unit suite | 337 s | 182 s |

The cause was `sp.nsimplify()` running mpmath's PSLQ on expressions that
were already exact — 13.8 seconds of a 21.7 second generation run. The
exactification now skips the search for an already-exact value.

### The verification, all of it without a GPU

```
209 tests                              OK
make contract-check                    CONTRACT OK
make check-arithmetic  (3,800 problems)  every printed expression evaluates
```

**The run-9 score is still a projection.** The base model is the control
and it scored 306/660 with byte-identical per-family numbers in runs 6,
7 and 8, so the harness is deterministic and the comparison is sound —
but the score needs the user's GPU.

---

## THE RUN THAT WAS SUBMITTED — 259/660 = 39.2% (2026-11-21)

**This run did NOT use the run-9 code.** The run-9 commit `10d0ab5` was
never pushed, so `git pull` reported "Already up to date" and the run
trained on commit `0f64ed5` — run 8's code.

Three independent pieces of evidence, all from the log itself:

1. **The contract-check output is the old one.** It ends "…system prompt,
   closing instruction and answer shape". The run-9 version ends "…and
   size mix" and prints a size table. There is no size table.
2. **Corpus generation ran at 412/s** against run 8's 373/s — 10%, not the
   20× (1,096 ms → 54 ms per eigenvalue problem) the run-9 code gives.
3. **The size mix is run 8's.** Run 8's `_rank_problem`/`gen_determinant`
   draw from `[2, 2, 3, 3, 4, 4, 5, 5]`, so its corpus carries 2×2 and
   under-teaches 5×5 — measured directly from `0f64ed5`:
   determinant {2:24, 3:33, 4:42, 5:51}, eigenvalue {2:53, 3:59, 4:22, 5:16}.

### What it measured

| | run 8 (250) | this run (259) | base (306) |
|---|---|---|---|
| overall | 37.9% | **39.2%** | 46.4% |
| 3×3 | 103/220 (46.8%) | 97/220 (44.1%) | 165/220 (75.0%) |
| 4×4 | 96/220 (43.6%) | 95/220 (43.2%) | 92/220 (41.8%) |
| 5×5 | 51/220 (23.2%) | **67/220 (30.5%)** | 49/220 (22.3%) |

Per family:

| family | run 7 | run 8 | this run | base |
|---|---|---|---|---|
| determinant | 4/150 | 1/150 | **4/150** | 54/150 |
| eigenvalue | 5/90 | 4/90 | 4/90 | 2/90 |
| matrix_power | 40/60 | 39/60 | **41/60** | 34/60 |
| matrix_vector | 37/60 | 17/60 | 18/60 | 44/60 |
| multiplication | 33/60 | 28/60 | **33/60** | 25/60 |
| nullity | 19/60 | 17/60 | **14/60** | 19/60 |
| rank | 29/60 | 27/60 | 26/60 | 21/60 |
| trace | 56/60 | 59/60 | 59/60 | 58/60 |
| transpose | 59/60 | 58/60 | **60/60** | 49/60 |

Without the determinant, the subtraction diagnostic:

    this run  255/510 = 50.0%
    base      252/510 = 49.4%

**The fine-tune now beats the base model on everything except the
determinant**, and by one problem. The determinant is still the entire
deficit: 4/150 against the base's 54/150.

### The loss curve is IDENTICAL to run 8's

| step | run 7 | run 8 | this run |
|---|---|---|---|
| 200 | 0.0542 | 0.0692 | 0.0687 |
| 400 | 0.0536 | 0.0784 | 0.0790 |
| 600 | 0.0425 | 0.0553 | 0.0543 |
| 800 | 0.0448 | 0.0591 | 0.0593 |
| 1000 | 0.0374 | 0.0597 | 0.0598 |

Same code, same corpus, so the loss should match — and it does, to within
0.0005 at every step. **This confirms the run really did use run 8's
code.**

### The 5×5 gain is the interesting part

5×5 went **51 → 67 (+16 problems, 23.2% → 30.5%)** and the fine-tune
now beats the base model on 5×5 by 8.2 points — the first time any
fine-tune has beaten the base on that column. 4×4 also beats the base
(+1.4).

So the two columns where the corpus under-taught 5×5 are exactly the two
that improved, and the 3×3 — where the corpus spent 2×2 rows — lost 6.

### The lessons

1. **Verify that the push landed BEFORE the run.** `git pull` saying
   "Already up to date" means the local branch matched the remote — it
   does not mean the remote has your work. Check
   `git log --oneline -1` after the pull, or put the commit SHA in the
   log banner.
2. **The corpus's size mix belongs in the contract, and the contract
   check should print it.** The check that would have caught this is
   exactly the size-mix table run 9 added.
3. **A run's own log can tell you which code ran.** The contract-check
   banner, the generation rate, and the corpus's size mix are all
   fingerprints. Three of them agreed.
4. **The 3×3 and 5×5 columns move in opposite directions when the size
   mix changes.** The next run must decide which size it is buying, and
   say so.

### The run-10 target

The determinant is still 4/150 against the base's 54/150, and removing it
gives 255/510 = 50.0% against the base's 252/510 = 49.4%. **The fine-tune
is one problem ahead of the base on every other family.** Everything now
turns on one family, and run 10 should touch the determinant and nothing
else — including the size mix, which must now stay uniform.

---

## RUN 9 RESULT — 225/660 = 34.1%, a regression (2026-11-22)

The run used the right code this time: the pull fast-forwarded
`0f64ed5..36d1bbc`, the contract check printed the size-mix table, and
corpus generation ran at **1,268/s** against run 8's 373/s — the
`_exactify` fast path, 3.4× faster.

| | run 7 | run 8 | the 259-run | **run 9** | base |
|---|---|---|---|---|---|
| overall | 282 = 42.7% | 250 = 37.9% | 259 = 39.2% | **225 = 34.1%** | 306 = 46.4% |
| 3×3 | 58/220 | 103/220 | 97/220 | 100/220 | 165/220 |
| 4×4 | 107/220 | 96/220 | 95/220 | **70/220** | 92/220 |
| 5×5 | 55/220 | 51/220 | 67/220 | 55/220 | 49/220 |

### The loss curve predicted it

| step | run 7 | run 8 | the 259-run | **run 9** |
|---|---|---|---|---|
| 200 | 0.0542 | 0.0692 | 0.0687 | **0.0761** |
| 400 | 0.0536 | 0.0784 | 0.0790 | 0.0812 |
| 600 | 0.0425 | 0.0553 | 0.0543 | 0.0578 |
| 800 | 0.0448 | 0.0591 | 0.0593 | 0.0592 |
| 1000 | 0.0374 | 0.0597 | 0.0598 | 0.0602 |

**Run 9 has the highest loss of any run and the worst score.** The
early-warning rule held: the loss arrives three hours before the
benchmark, and it said run 9 would be worse than run 8.

### The cause: run 9 stopped teaching the multiplications

| family | 259-run | **run 9** | base | Δ |
|---|---|---|---|---|
| **matrix_power** | 41 | **15** | 34 | **−26** |
| matrix_vector | 18 | 8 | 44 | −10 |
| multiplication | 33 | 29 | 25 | −4 |
| determinant | 4 | 3 | 54 | −1 |
| eigenvalue | 4 | 2 | 2 | −2 |
| nullity | 14 | 20 | 19 | **+6** |
| rank | 26 | 29 | 21 | **+3** |
| trace | 59 | 59 | 58 | 0 |
| transpose | 60 | 60 | 49 | 0 |

`matrix_power` lost 26 problems — more than the whole 34-problem
regression.

**The mechanism.** Run 9 changed the dot product's format at 4×4 and 5×5
to fit the 1,536-token sequence cap:

    full:     4*4 + (-4)*(-4) + 3*3 + 0*0 = 16 + 16 + 9 + 0 = 41
    compact:                        16 + 16 + 9 + 0 = 41

The compact form drops the **products**. The model is shown the partial
sums and never the multiplication that produced each term, so it is
taught to emit a number it cannot compute. `matrix_power` went from
41/60 to 15/60 and the 4×4 column from 95 to 70.

**The arithmetic checker could not see it.** It validates the expression
on the LEFT of each `=`, and a compact line is a perfectly well formed,
perfectly well valued expression. What it stopped TEACHING is invisible
to a checker that only asks whether what is present is right.

### The fix

**The 4×4 form fits the cap in full.** Measured, not assumed:

| | full form | tokens | cap |
|---|---|---|---|
| matrix_power 3×3 | 1,206 chars | 591 | ok |
| matrix_power 4×4 | 2,005 chars | **983** | ok |
| matrix_power 5×5 | 3,267 chars | 1,601 | over |
| matrix_multiply 4×4 | 2,008 chars | **984** | ok |
| matrix_multiply 5×5 | 3,237 chars | 1,587 | over |

So run 10 restores the FULL dot product at 3×3 and 4×4 and keeps the
reduced form at **5×5 only** — the one size that genuinely does not fit.
The longest example is now 1,242 tokens.

`tests/test_derivation_shows_products.py` pins it, and it fails on run
9's code with the defect quoted in the message.

### The lesson

**A checker that asks "is what is present correct?" cannot see what has
stopped being present.** The arithmetic checker proved every printed
expression evaluated to the value it claimed — 3,800 problems, all
correct — while the corpus was quietly teaching less.

Run 9 traded away the thing the model most needed to learn in order to
fit a sequence-length cap, and the cap did not actually require the
trade at 4×4. The measurement that would have prevented it: what does
the FULL form cost at each size? One step per entry at 4×4 is 983
tokens against a 1,536 cap.

**Corollary:** when a change is made to fit a limit, measure the limit at
EVERY size the change touches, not at the size that failed. Run 9
measured 5×5, applied the fix to 4×4 as well, and lost 26 problems.

---

## Run 11 — the families that suck, and why (2026-11-23)

**The request.** "Only trace and transpose work relatively well all other
tasks suck. Fix them." Run 10 measured `276/660 = 41.8%` against the base
model's `306/660 = 46.4%`, and the per-family table said exactly this:

| family | run 10 | base | gap |
|---|---|---|---|
| determinant | **1** / 150 | 54 / 150 | **−53** |
| eigenvalue | **1** / 30 | 2 / 30 | −1 |
| nullity | 14 / 60 | 19 / 60 | −5 |
| rank | 27 / 60 | 21 / 60 | +6 |
| multiplication | 35 / 60 | 25 / 60 | +10 |
| matrix_power | 38 / 60 | 34 / 60 | +4 |
| matrix_vector | 43 / 60 | 44 / 60 | −1 |
| trace | 58 / 60 | 58 / 60 | 0 |
| transpose | 59 / 60 | 49 / 60 | +10 |

Without the determinant the fine-tune was **275/510 = 53.9%** against the
base model's **252/510 = 49.4%** — so the determinant was the entire
remaining deficit, and it was a deficit of 53 problems in one family.

Run 11 fixes nine separate causes. Every one of them was found by
measuring, not by guessing.

### 1. The task statement is part of the contract, per (family, size)

The corpus invented its own task statements: 66–78% of them named no size
at all, and the ones that did wrote the size with an ASCII "x" where the
benchmark writes U+00D7. The benchmark's statement is deterministic per
(family, size), and the corpus must use the benchmark's own words.

`src/la/benchprompt_tasks.py` holds them, extracted byte-exactly from the
released files, and `scripts/extract_prompt_parts.py --check` verifies them
against those files in both directions — it is wired into
`make contract-check`.

**The 4×4 shape is the scorer's, not the file's.** The 4×4 problem file is
a bare matrix with no task statement, but the scorer reconstructs the task
from the subcategory label and sends *"Find the determinant of the 4×4
matrix A."* So the corpus must teach the reconstructed statement. Both
sides call `benchprompt.task_for()`, so they cannot drift apart.

### 2. The prompt tail is size-specific, and the method and the closing
### instruction are ONE paragraph

The corpus's tail table mixed the sizes: the transpose's 3×3 tail could
turn up in a 5×5 problem. And the method instruction and the closing
instruction were in two paragraphs, where the benchmark has them in one,
separated by a single newline — so a prompt with a method instruction had
five paragraphs where the benchmark's has four.

`src/la/benchprompt_tails.py` now carries the tails per size, with the
line breaks the files actually have.

### 3. The prompt decoration must not be able to move the matrix

The task statement and the tail were drawn from the SAME random stream as
the matrix they decorate. Every change to the prompt contract consumed a
different number of draws and silently rewrote every matrix after it —
the run 8 defect, in a form that does not change the corpus's shape and
so cannot be seen in the output.

Run 10's corpus was itself a reshuffle because of this. The fix is
`benchprompt_tails.decoration_rng`: the decoration takes exactly ONE draw
from the problem's stream — its seed — and its own stream after that, so
the count cannot depend on the contract. `tests/test_decoration_stream.py`
pins it, and it fails on run 10's code.

### 4. The determinant's cofactor minors were ASSERTED, not computed

    the first minor is \begin{bmatrix} 4 & -1 \\ 5 & 1 \end{bmatrix} = 9

The model was shown the minor's value and never the arithmetic that
produced it — taught to emit a number it could not compute. This is run
9's lesson in a different family, and the determinant is the family that
lost 53 problems.

The minor is now computed: `= 4*1 - (-1)*5 = 4 - -5 = 9`.

### 5. The determinant's elimination used FRACTIONS

Every 4×4 and 5×5 determinant ran through `\frac{-281}{32}` and
`\frac{-688}{281}`. The benchmark uses **no fractions at all** — 0 of its
660 problems and 0 of its 660 answers contain one, and all 150 of its
determinant answers are integers.

The elimination is now the fraction-free (Bareiss) form, whose divisions
are exact, and the determinant is the last diagonal entry of the Bareiss
form. The verification step is an independent cofactor expansion, so it
is a different computation rather than a repeat of the answer.

### 6. The eigenvalue's characteristic polynomial was PYTHON, not LaTeX

    the characteristic polynomial is p(λ) = lambda**3 + 7*lambda**2 - ...

`lambda**3` is Python's syntax. It rendered as italic "lambda**3" and it
was the FIRST line of every eigenvalue derivation in the corpus. The
characteristic polynomial is now `\lambda^{3} + 7 \lambda^{2} - 38 \lambda
- 240`, and the roots are FOUND rather than asserted: the factoring is
shown, and the polynomial factors were verified against sympy on 60
samples.

### 7. The rank and nullity corpora could not produce a full-rank matrix

The rank was drawn from 1 to n−1, so a full-rank matrix was **impossible**.
The benchmark's matrices span the whole range and the full-rank case is
the common one:

    benchmark rank (size, rank)     corpus rank (size, rank), before
      3x3  1: 2  2: 8  3: 10         3x3  1: 51  2: 43
      4x4  2: 4  3: 7  4: 9          4x4  1: 36  2: 32  3: 34
      5x5  2: 3  3: 4  4: 6  5: 7    5x5  1: 26  2: 26  3: 24  4: 28

The corpus's distribution was the benchmark's shifted one rank down, and
the benchmark's single most common 3×3 rank answer is 3 — which the
corpus could not produce. The rank and nullity families were the two
that LOST to the base model, and they are the two whose corpus was
missing a whole rank.

`src/la/benchprompt_ranks.py` carries the benchmark's own weights per
size, so the corpus's mix of RANKS matches the mix of rank ANSWERS.

### 8. The rank reduction scaled the pivot rows, and stated its operations
### without their results

`R1 <- R1 / (-14)` divided a whole row and manufactured fractions, and the
echelon form's entries then appeared from nowhere. The reduction no longer
scales — the rank is the number of non-zero rows in the echelon form,
which is what the benchmark asks for — and each operation is followed by
the row it produced.

### 9. The 5×5 dot products dropped their PRODUCTS — and the cap that
### justified it was never real

Run 9 taught the 5×5 dot products in a compact form that dropped the
products:

    full:     4*4 + (-4)*(-4) + 3*3 + 0*0 = 16 + 16 + 9 + 0 = 41
    compact:                        16 + 16 + 9 + 0 = 41

because "the full form runs to 1,601 tokens against a 1,536 cap". **Both
numbers were wrong.** The 1,601 was measured against a longer derivation
that no longer exists, and the 1,536 was `MAX_SEQ_LEN`, a stale constant
enforced NOWHERE — the real cap is `max_seq_len: 2048` in
`configs/default.yaml`, and the trainer truncates at that.

So the 5×5 examples were measured in REAL tokens for the first time, with
the model's own tokenizer (`scripts/qwen_bpe.py`, a faithful BPE over the
model's `tokenizer.json`):

| | full form | real tokens | vs 2048 |
|---|---|---|---|
| matrix_power 5×5 | per entry | 1,892 | fits |
| matrix_multiply 5×5 | per entry | 2,042 | fits |

with **zero** of 800 sampled examples over the cap. Getting there took
four changes, each measured:

- the 5×5 entries are grouped by row, so the label is paid five times
  instead of twenty-five;
- the per-entry label is `(1,1) =` rather than `(AB)_{11} =`;
- the products carry no parentheses: `3*-2` is unambiguous and is three
  tokens shorter per entry;
- the plot spec stores whole numbers as whole numbers — `6` not `6.0` —
  which is 157 tokens down to 107 for a 5×5.

**A cap measured in characters divided by a heuristic is not a cap.** The
5×5 exception cost the matrix_power family 26 problems, and it was never
needed.

### What run 11 does NOT change

`enable_thinking=False`, the EOS supervision, the shared `row_prompt()`
rendering, the grader and the released benchmark files are all untouched.
`configs/default.yaml` is still the single source of truth for the base
model and the corpus weights.

### The verification

| | |
|---|---|
| unit suite | 238 tests |
| `make contract-check` | CONTRACT OK, and the prompt tables match the released files |
| `make check-arithmetic` | 3,800 problems, every printed expression evaluates to the value it claims |
| the corpus's length | measured in REAL tokens against the real 2,048 cap |


---

## Run 12 — the trainer's cap was 1,536 and the config said 2,048 (2026-11-23)

### What the log showed

The right commit ran (`491ac2a`), the contract check passed, the prompt
tables matched the released files, the corpus built at 957 problems a
second and the arithmetic was clean on 3,800 problems. Then:

```
  step      1/1,250  loss 1.6386  lr 2.70e-06  10.12 s/step  ...
                                                     eta 210.6 min  vram 5.8/8.5 GB
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling
`cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```

and then two more errors that were both consequences of the crash.

### THE CAP WAS NEVER THE CONFIG'S CAP

The banner printed `max seq len 1536`. The config says `max_seq_len:
2048`. The Makefile read `configs/default.yaml` for the base model and
nothing else; `la.train` never opened the file and carried its own
`default=1536`. **The config was inert.**

Measured with the model's own tokenizer over 1,500 sampled examples, the
full training text (prompt + completion):

| cap | truncated |
|---|---|
| 1,536 | 222 / 1,500 = **14.8%** |
| 2,048 | 45 / 1,500 = 3.0% |
| **2,304** | **0 / 1,500 = 0%** |

The longest example is **2,132 tokens** — a 5×5 matrix product with its
products shown — and the cap is now **2,304**, read from the config.

The truncation lands at the END of the completion, which is where the
answer and the plot spec are. So 14.8% of the corpus was being taught
without its answer, in exactly the families run 11 had just fixed.

Memory check: the logits cost about 1.5 MB per token over a vocabulary of
151,936, so a 2,304 cap needs about **7.2 GB** of live tensors against the
card's 12.88 GB. It fits.

**A value in a configuration file is not a value the code uses unless the
code reads it.** The trainer now reads it, and `tests/
test_config_is_the_source_of_truth.py` fails if the two ever disagree.

### THE TOKENIZER CAME FROM THE ADAPTER

`load_model` did `AutoTokenizer.from_pretrained(adapter or base)`. The
trainer creates the output directory BEFORE the training loop and writes
the adapter into it at the VERY END, so the crash left an **empty
directory** behind, and every later command died with:

```
ValueError: Couldn't instantiate the backend tokenizer from one of:
(1) a `tokenizers` library serialization file,
(2) a slow tokenizer instance to convert or
(3) an equivalent slow tokenizer class to instantiate and convert.
```

which reads like a missing dependency rather than a missing training run.
The tokenizer is the base model's; the adapter holds weights. It now
comes from the base, always.

### AND THE EMPTY DIRECTORY WOULD HAVE SCOREDED AS THE BASE MODEL

The benchmark's adapter guard skipped anything with no
`adapter_config.json` — which is exactly the empty shell — and fell
through to the base model. It would have produced a full 660-problem
score and reported it as the fine-tune's. **The number would have been
indistinguishable from a real one.**

There is now a guard that refuses an adapter directory with no adapter in
it, and the trainer writes `.incomplete` into the output directory before
the loop and removes it when the run succeeds.

### THE GPU FAULT IS NOT A BUG IN THE LOOP

`CUBLAS_STATUS_EXECUTION_FAILED` has two documented causes — the card is
out of memory, or the Windows TDR watchdog gave up on a long kernel — and
neither is in the training code. **The same code trained the whole of run
10 on the same card, at the same cap, with the same allocator.** Run 11's
banner and run 10's differ only in the corpus's length.

So the fault is retried, with `torch.cuda.empty_cache()` between
attempts, and a batch whose retries are exhausted is skipped rather than
fatal: a batch of 20,000 is worth more than one batch of 16.

### What run 12 changes

| | |
|---|---|
| the cap | 2,304, read from `configs/default.yaml` by `la.train` |
| the trainer's other defaults | read from the config too: lr, batch size, grad accum, warmup ratio, base |
| the tokenizer | loaded from the base, never from the adapter |
| an empty adapter directory | refused by the benchmark, with a message that says what to do |
| an unfinished run | marked with `.incomplete`, unmarked on success |
| a GPU fault | retried with the cache emptied; the batch is skipped if it does not clear |

`enable_thinking=False`, the EOS supervision, the shared `row_prompt()`
rendering, the grader and the released benchmark files are all untouched.
The corpus itself is **unchanged** — run 12 is about the trainer and the
benchmark harness, not about the mathematics.

---

## RUN 12 RESULT — 273/660 = 41.4%, and the grader was reading seven of the nine families with one eye shut (2026-11-23)

### The run itself was clean

The right commit ran. The contract check passed, the prompt tables matched
the released files, the corpus built at 957 problems a second and the
arithmetic was clean on 3,800 problems. Training ran the full 1,250 steps
with no GPU fault:

```
  max seq len        2304
  step      1/1,250  loss 1.6386 ... vram 5.8/8.5 GB
  step    1,250/1,250  loss 0.0658 ... 181.9 min
peak vram: 6.40 GB allocated, 12.10 GB reserved
saved adapter to runs/vslm-run12
```

So the cap fix took effect (2,304 in the banner, not 1,536) and the crash
did not come back.

### The score

**273/660 = 41.4%.** The base model scored **311/660 = 47.1%.**

| | run 10 | run 12 | base |
|---|---|---|---|
| overall | 276 = 41.8% | **273 = 41.4%** | 311 = 47.1% |
| determinant | 1/150 | 7/150 | 60/150 |
| eigenvalue | 1/30 | 1/30 | 1/30 |
| matrix_power | 38/60 | 22/60 | 33/60 |
| matrix_vector | 43/60 | 36/60 | 43/60 |
| multiplication | 35/60 | 32/60 | 28/60 |
| nullity | 14/60 | 22/60 | 19/60 |
| rank | 27/60 | 35/60 | 20/60 |
| trace | 58/60 | 58/60 | 57/60 |
| transpose | 59/60 | **60/60** | 50/60 |
| 3x3 | 113 | 118 | 165 |
| 4x4 | 106 | 92 | 97 |
| 5x5 | 57 | **63** | 49 |

loss@200 was 0.0789, against run 10's 0.0774 — the same, and the scores
are the same, which is what the loss has been good at predicting.

### THE GRADER COULD NOT READ THE BENCHMARK'S OWN ANSWERS

The eval log has the reason, in a single line:

```
reason: unparseable: cannot tokenise at '² = \begin{bmatrix} '
```

`bench.score` parses the REFERENCE with

```python
parse_answer(normalise_reference(p["answer"]))
```

and the MODEL'S boxed answer with

```python
parse_answer(got)          # <- no normalisation
```

`normalise_reference` exists precisely because a benchmark answer is prose
around a value — `\det(A) = -41`, `A^2 = \begin{bmatrix} ...`, `rank = 3`
— and the label has to come off before the answer grammar sees it. The
reference got that treatment and the model's answer did not.

**Measured against the released files, the old path reads 120 of the 660
reference answers and the new path reads all 660.** Every one of the 660
released answers carries a label:

| family | the label | old path reads |
|---|---|---|
| determinant | `\det(A)` | 0 of 150 |
| eigenvalue | `\lambda` | 0 of 90 |
| matrix_power | `A^2` | 0 of 60 |
| nullity | `\text{nullity}(A)` | 0 of 60 |
| rank | `\text{rank}(A)` | 0 of 60 |
| trace | `\text{tr}(A)` | 0 of 60 |
| transpose | `A^T` | 0 of 60 |
| matrix_vector | `Ax` | 60 of 60 |
| multiplication | `AB` | 60 of 60 |

Two forms fail, for two different reasons:

- `A^2 = <matrix>` — **the benchmark's own answer form.** The lexer reads
  `^` as a bitwise XOR and then fails on the `2`.
- `A² = <matrix>` — what the model writes, because the benchmark's own
  PROBLEM STATEMENT says "Compute A² = A × A" and the model copies it.
  U+00B2 is not in the lexer's name pattern at all.

Both are correct mathematics. Both scored zero.

### WHY THE FINE-TUNE LOST AND THE BASE WON

The base model answers with a bare number in the box, so the grader read
it. The fine-tune learned the benchmark's answer STYLE and put the label
on, because the problem statement gave it the label to copy — and the
grader misread the label.

**The grader punished the fine-tune for matching the benchmark's own
conventions.** The base model's 128 "no answer parsed" were answers it
put in prose instead of a box; the fine-tune's 0 unparsed were answers the
grader read but MISread.

That is why the two families where the fine-tune won big — rank +15 and
transpose +10 — are the ones where its style happened to survive: its rank
answer was a bare number and its transpose answer was a bare matrix.

### The fix

`bench.score` now normalises the model's answer the way it normalises the
reference, and `normalise_reference` maps the superscript the problem
statement uses onto the caret the answer grammar can read:

```python
a = a.replace("²", "^2").replace("³", "^3")
```

`tests/test_the_answer_is_read_like_the_reference.py` pins it, and the
same normalisation applies to the base model, so the comparison stays
symmetric.

### What this does NOT change

The corpus. It teaches a bare value with no label — `the answer is -534`
— which is exactly what `normalise_reference` produces. The corpus was
never wrong here; the grader was reading the two sides of the comparison
differently.

### And the remaining real problem

With the label read correctly, the determinant is still the family's weak
point. The held-out eval showed `determinant 0/13` and `eigenvalue 0/14`,
and those go through `extract_model_answer`, which ALREADY normalised.
So those are genuine mathematics failures, not formatting. The Bareiss
elimination and the computed cofactor minors are in the corpus and the
model is still not producing them.

**So the label was a real bug worth fixing, and it is not the whole
story.** The determinant remains the family's problem to solve.

---

## The paging file is too small — and it is not a corrupt checkpoint (2026-11-23)

The re-score of run 12 never started:

```
OSError: The paging file is too small for this operation to complete. (os error 1455)
  File ".../transformers/modeling_utils.py", line 4444, in _load_pretrained_model
    file_pointer = safe_open(file, framework="pt", device=device, backend=backend)
```

### WHAT IT IS

`os error 1455` is `ERROR_COMMITMENT_LIMIT`. The machine could not commit
enough **virtual** memory. It is not a disk error — the checkpoint is on
`G:` with room, and `C:` has about 26 MB free, which is why the HF cache
lives on `G:` in the first place — and it is not a network error.

### WHY IT HAPPENS

`safetensors` opens a shard with a copy-on-write memory map, and Windows
charges the **ENTIRE VIEW** to the commit limit for as long as the shard
is open. The `MapViewOfFile` documentation says so outright:

> When copy-on-write access is specified, the system and process commit
> charge taken is for the entire view

So a 3.4 GB checkpoint costs 3.4 GB of commit for the whole load, on top
of the RAM, and the commit limit is RAM plus the page file. A machine
whose page file is small or system-managed runs out — and the error names
the page file rather than the real cause.

### IT WAS NOT A REGRESSION

The identical command loaded the same model **four times** in the previous
session, including a 124-minute generation run:

```
Loading weights: 100%|...| 311/311 [00:04<00:00,  68.74it/s]
```

and the run 13 diff is 22 lines, all in the scoring path, nowhere near
`from_pretrained`.

### THE FIX IS UPSTREAM, AND IT IS BACKPORTED HERE

`transformers` 5.17 switched Windows to `safetensors`' `pread` backend,
which reads each tensor into its own buffer and maps nothing, so nothing
is charged to commit for the life of the file handle. `safetensors` has
had the `backend` keyword since 0.8.

`src/la/pread.py` backports that behaviour to the versions before 5.17. It
patches `transformers.modeling_utils.safe_open` so a **CPU** shard open
defaults to `backend="pread"`, and it is installed by both paths that
load a checkpoint (`la.inference.load_model` and `la.train.build_model`).

It is a no-op off Windows, on a `safetensors` older than 0.8, and on a
`transformers` that has already been fixed. `make check-memory` prints the
commit limit from `GlobalMemoryStatusEx` and warns when there is under
about 6 GB free.

### THE BUG IN MY OWN FIRST ATTEMPT

I checked whether `safetensors` supports the `backend` keyword by looking
for it in `inspect.signature(safe_open).parameters`. **`safe_open` takes
`**kwargs`, so `backend` never appears as a named parameter** — the check
rejected every real installation, and the patch silently never installed,
on every machine.

**A patch that silently does not install is worse than no patch.** The only
reliable test is to CALL it:

```python
_real_safe_open(os.devnull, framework="pt", device="cpu", backend="pread")
```

`TypeError` means safetensors is too old and the patch must give up; any
other exception means the keyword was accepted, which is the answer we
needed.

And the test that catches this class of thing builds a **fake**
`safetensors` and a **fake** `transformers` in `sys.modules` and asserts
that the patch really installs and really changes what is passed to the
open. **A patch must be tested by what it does to a call, not by what its
source contains.**

### The rule

**A test that asserts on a patch's SOURCE proves the source was written.
Only a test that exercises the patch proves it works.**
