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
| decode | ~38 ms/step at batch 8 | **7.8× the ~5 ms memory-bandwidth floor** |

**The CUDA build is not the problem.** That was worth ruling out, because the
first hypothesis — a bad build — would have sent the user off to reinstall
torch for no gain. The remaining suspect is the attention backend: PyTorch's
own documentation says the SDPA *math* backend upcasts FP16/BF16 inputs to
FP32, which "may cause the performance regressions in the math backend".
`scripts/check_attention.py` measures it.

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

### The Hugging Face cache lives on G:

The cache defaults to `%USERPROFILE%\.cache\huggingface`, which on this
machine is on C: — where there was about **26 MB free**. A 1.7B checkpoint
needs about **4 GB**, and the download died part way through with:

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
