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
