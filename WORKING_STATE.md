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
