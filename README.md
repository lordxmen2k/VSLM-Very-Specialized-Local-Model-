# VSLM — a linear algebra model that runs on one 12 GB card

**Very Specialised Local Model.** A fine-tune of `Qwen3-1.7B-Base` that answers
linear algebra questions, shows its working, and draws the figure.

Everything the model was trained on is generated from scratch by this
repository. There is no scraped corpus, no borrowed dataset and no third-party
model in the loop: sympy computes every answer, and the training data is
therefore correct by construction. The corpus is reproducible from a seed.

```bash
git clone https://github.com/lordxmen2k/VSLM-Very-Specialized-Local-Model-
cd VSLM-Very-Specialized-Local-Model-
pip install -e ".[train,dev]"

make preflight # check the GPU, CUDA and bitsandbytes first
make data      # build the corpus            (~30 min, 4 workers)
make train     # QLoRA fine-tune            (~30 min for 2 epochs)
make eval      # score it on the held-out set
make chat      # ask it something
```

---

## What it does

```
### Problem
For v = \begin{pmatrix} 3 \\ 5 \end{pmatrix} and w = \begin{pmatrix} 3 \\ 1 \end{pmatrix},
compute the orthogonal projection of the first vector onto the second, and the residual.

### Scratchpad
GIVEN: v = \begin{pmatrix} 3 \\ 5 \end{pmatrix}, w = \begin{pmatrix} 3 \\ 1 \end{pmatrix}
STEP 1 <v, w> = 14 and <w, w> = 10
STEP 2 proj = (<v,w> / <w,w>) w = \begin{pmatrix} \frac{21}{5} \\ \frac{7}{5} \end{pmatrix}
STEP 3 residual = v - proj = \begin{pmatrix} \frac{-6}{5} \\ \frac{18}{5} \end{pmatrix}
STEP 4 VERIFY proj + residual = v and proj is parallel to w. OK

### Answer
p = \begin{pmatrix} \frac{21}{5} \\ \frac{7}{5} \end{pmatrix}, r = \begin{pmatrix} \frac{-6}{5} \\ \frac{18}{5} \end{pmatrix}

### Plot
{"kind":"projection","data":{...},"labels":{...},"annotate":["orthogonal projection"]}
```

Three things follow from that shape.

**The scratchpad is computed, not written.** Every step in it is what sympy
actually did, and every answer ends with a VERIFY line that re-checks the
result by substitution. The model learns the habit of computing and then
checking, which is the habit that makes its arithmetic checkable.

**The model never draws.** It emits a small declarative JSON object describing
the figure, and a deterministic renderer turns that into a PNG. A drawing
instruction is a far smaller competence to learn than a drawing, and it means
the numbers in the picture are the numbers in the answer.

**The answer is checked, not trusted.** `la.answer` defines a restricted LaTeX
grammar for answers and parses it back into sympy. The evaluation harness
compares the model's parse against the ground truth. Nothing about the
evaluation depends on the model being right about itself.

---

## The seventeen problem families

Every family is generated, verified by sympy, and paired with a figure.

| Family | What it asks | Figure |
|---|---|---|
| `determinant` | det of an n×n integer matrix | matrix heatmap |
| `solve_system` | solve Ax = b uniquely | A and its reduced row echelon form |
| `eigenvalues` | the spectrum, real or complex | eigenvalues on the complex plane |
| `diagonalize` | the diagonal form D in A = P D P⁻¹ | spectrum of D |
| `orthonormal_basis` | Gram–Schmidt to an orthonormal set | the basis in 2D or 3D |
| `projection` | projection of v onto span(w), and the residual | vector, projection, residual |
| `least_squares` | the least-squares solution of an overdetermined system | data and fitted line |
| `lu_factor` | L and U with LU = A | the upper triangular factor |
| `svd` | the singular values | singular value spectrum |
| `condition_number` | cond₂(A) | matrix heatmap |
| `rank_nullspace` | rank and nullity | a basis of the null space |
| `linear_transform` | T(v) and the image of the unit square | the unit square and its image |
| `matrix_trace` | tr(A) | matrix heatmap |
| `matrix_transpose` | Aᵀ | matrix heatmap |
| `matrix_multiply` | AB | matrix heatmap |
| `matrix_power` | Aᵏ | matrix heatmap |
| `matrix_vector` | Av | matrix heatmap |

The last five exist because they are the operation types a published benchmark
uses. Without them a score on that benchmark would not be a score on the same
axis as anyone else's.

---

## Why the corpus is different from a normal one

Three deliberate choices, each of which matters more than it looks.

**1. The ground truth is mechanical.** There is no human-written answer
anywhere in this repository. sympy computes the answer, the scratchpad and the
figure values, and the training data inherits them. The usual failure of a
domain corpus — the answer quietly drifting from the question — cannot happen.

**2. Every problem is augmented.** A model trained on one phrasing of one
shape of problem learns to recognise that shape. So each problem is asked
several ways (three to five templates per family), and matrices are relabelled:
rows permuted, or put through a similarity transform, which preserves the
spectrum exactly. The answer is recomputed from the matrix as it appears in
the question, so augmentation can never introduce an inconsistency.

**3. The scratchpad ends with a check.** Every generated derivation closes with
a VERIFY line. The model learns to state the check, which is the one thing that
distinguishes a model that can be audited from one that merely sounds certain.

---

## Why it fits on a 12 GB card

`Qwen3-1.7B-Base` is 1.72B parameters: 28 layers, hidden 2048, 16 query
heads, 8 KV heads, head dimension 128, tied embeddings, 32,768-token native
context.

| | |
|---|---|
| Weights, bf16 | 3.44 GB |
| Weights, NF4 (QLoRA) | ~0.9 GB |
| KV cache per token | 112 KB (bf16) · 56 KB (q8_0) |
| KV cache at 8K context | 0.88 GB (bf16) · 0.44 GB (q8_0) |

Inference is comfortable. **Q4_K_M weights (1.1 GB) plus an 8K context plus
runtime overhead is about 2.2 GB**, and at 32K context in bf16 it is about
7.6 GB. A 131K context fits only if the KV cache is quantised, and only just.

Training is the part that needs care, and QLoRA is what makes it work:

| Method | VRAM | Host RAM | Fits 12 GB? |
|---|---|---|---|
| Full fine-tune, AdamW | 27.5 GB | — | no |
| Full fine-tune, 8-bit Adam | 10.3 GB | ~6 GB | tight; no room for activations |
| Full fine-tune, offloaded optimiser | 6.9 GB | ~21 GB | yes |
| **LoRA, bf16 base** | **~7 GB** | ~2 GB | yes, comfortable |
| **QLoRA, NF4 base** | **~2 GB** | ~2 GB | yes, very comfortable |

QLoRA saves about 2.5 GB and is somewhat faster. LoRA on a bf16 base is the
fallback when bitsandbytes will not load, and it fits on its own.

The default is QLoRA: 4-bit NF4 with double quantisation, rank 32, alpha 64,
gradient checkpointing, sequence length 512, batch size 4 with 4 gradient
accumulation steps (an effective batch of 16), `paged_adamw_8bit`, bf16
compute, lr 1e-4 cosine with 3 % warmup, one epoch.

One epoch is the default because it is the cheap half and the corpus is
large enough to fill it. For a second pass, `make train TRAIN_ARGS="--epochs 2"`.

**Measured cost.** 1,250 optimiser steps per epoch over 20,000 examples, at
an effective batch of 16.

The first configuration tried — batch 1, 16-way accumulation, 2048-token
sequences — measured **9.0 s per optimiser step** on an RTX 4070 SUPER, which
is **six hours** for two epochs. The default is now batch 4 with 4-way
accumulation: the same effective batch, the same number of optimiser steps,
but four times fewer forward and backward passes, each four times larger.
That is expected to be faster because a 250-token sequence at batch 1 leaves
the GPU badly under-occupied. **It has not been measured yet.** Measure it
before committing to a long run:

```bash
make train TRAIN_ARGS="--max-steps 20 --limit 200 --log-every 1"
```

The log line reports the time per step, the peak VRAM, the time spent
waiting for the data loader, and the tokens per second, so a slow GPU and a
starved loader are distinguishable.

**Measured, not estimated.** Batch 1 with 16-way accumulation and batch 4
with 4-way accumulation both measured about 9-10 s per optimiser step, on the
same effective batch of 16 and the same tokens. Batching alone therefore did
not help, which rules out the obvious explanation — an under-occupied GPU —
and points at the cost being per token. `make speed-test` measures that
directly: the same data through five micro-batch shapes, with the token
count of the corpus printed alongside.

```
  corpus: 2000 examples, 612,441 tokens, mean 306 tokens, median 288, p95 431, max 449
  sequence cap: 512 tokens

  batch  accum  eff   s/step     tok/s     ex/s   vram GB
      1     16   16      9.02        532     1.8      3.91
```

Read it as: if `s/step` is flat across the row, the cost is per token and
the levers are fewer tokens or fewer examples, not a different batch shape.
If it falls steeply, the micro-batch shape was the problem.

The corpus build measures about 594 problems a second on four workers — the
20,000-problem training set takes about 34 seconds, and the 40 sample
figures add about half a minute at 0.77 s each.

---

## Install

### Linux

Install torch **first**. If it is missing, `pip install -e ".[train,dev]"`
pulls it in from PyPI as a dependency of peft and accelerate, and PyPI's
torch is the CUDA 13.2 build that bitsandbytes cannot load.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install --no-deps bitsandbytes==0.50.2
pip install -e ".[train,dev]"
```

### Windows

```bat
scripts\setup_windows_gpu.bat
```

### Why torch is not in the requirements

The `train` extra deliberately lists **neither torch nor bitsandbytes**.
Both resolve from PyPI. PyPI's torch is built against the newest CUDA
release, and an older bitsandbytes has no binary for it — so if pip is
allowed to choose, `pip install -e ".[train,dev]"` will quietly replace a
working CUDA-matched torch with one that breaks the run. Install those two
yourself, from the index that matches your GPU, and pip will leave them
alone.

`make setup` therefore never touches torch. If torch is missing,
`make preflight` says so in one line.

---

## The commands

### Build the corpus

```bash
make data                       # 20,000 train + 200 held-out
# or
python -m la.build_dataset --n-train 20000 --n-test 200 --seed 7 --out data --workers 4
```

Writes `data/train.jsonl`, `data/test.jsonl` and a sample of rendered figures
in `data/figures/`. The build is deterministic: the same seed gives the same
corpus, whatever the worker count.

### Render a sample of figures first

```bash
make sample
```

Cheap sanity check: 240 problems and 24 rendered figures, about a minute. Look
at `data-sample/figures/` before spending an hour on the full corpus.

### Train

```bash
make train
# or
python -m la.train --data data/train.jsonl --out runs/vslm --epochs 2 --lr 1e-4
```

Smoke test first — twenty steps on five hundred examples, about a minute:

```bash
make train TRAIN_ARGS="--max-steps 20 --limit 500"
```

Then the real run:

```bash
make train
```

Useful flags: `--max-steps 50` for a smoke test, `--limit 2000` for a short
run, `--max-seq-len 4096` if you have the memory and want longer problems,
`--precision bf16` to train a LoRA on an unquantised bf16 base (about 7 GB
in total) instead of the default NF4 quantised one (about 2 GB). The 24 GB
figure in the table above is for a *full* fine-tune, which this project never
does — every run is a LoRA.

### Measure the speed

```bash
python -m la.bench --perf --adapter runs/vslm
```

Time to first token, decode throughput at batch 1, 4 and 8, peak VRAM and
model load time.

### One thing to know about the LaTeX

LinAlg-Bench writes every matrix as `\begin{bmatrix}`. This corpus writes
`\begin{bmatrix}` or `\begin{pmatrix}`, chosen from the contents of the
matrix, so the same matrix is written the same way within an example while
different matrices in the same example may use different environments. A
model that has only ever seen one spelling meets the other for the first
time on benchmark day, and the answer grammar accepts both.

### Score it against published results

```bash
make bench              # LinAlg-Bench, with the published table printed
make bench-all          # the above, plus the base model as a baseline, plus --perf
```

`la.bench` downloads **LinAlg-Bench** — SymPy-verified linear algebra problems
across 3×3, 4×4 and 5×5 matrices and nine operation types, with ten frontier
models scored on it (NeurIPS 2026 Datasets & Benchmarks; CC-BY-4.0). The
report prints your score next to theirs:

| Model | 3×3 | 4×4 | 5×5 |
|---|---|---|---|
| OpenAI-o1 | 94.6 % | 92.3 % | 81.4 % |
| Gemini-3.1-Pro | 93.7 % | 94.6 % | 88.0 % |
| DeepSeek-V3 | 89.3 % | 79.9 % | 62.5 % |
| GPT-5.2 | 85.8 % | 82.1 % | 69.2 % |
| Qwen3-235B | 79.8 % | 80.5 % | 71.5 % |
| Mistral-Large | 81.4 % | 75.6 % | 45.1 % |
| Claude-4.5-Sonnet | 80.1 % | 69.7 % | 33.1 % |
| Llama-3.3-70B | 63.1 % | 60.6 % | 28.3 % |
| Qwen2.5-72B | 69.4 % | 39.8 % | 24.6 % |
| GPT-4o | 66.6 % | 36.0 % | 28.7 % |

Add `--baseline` and the base model is scored on the same problems, so the
delta attributable to the fine-tune is printed rather than asserted.

**What that table says, and why the corpus is built the way it is.** Accuracy
falls off a cliff between 4×4 and 5×5 — 32 percentage points on determinants —
and the models that fail there do not compute badly, they *stop computing* and
fabricate an answer. The benchmark calls them "cofactor-locked": models that
expand a 5×5 determinant by cofactors lose the sign and the terms. So:

- **5×5 is in the training mix.** The corpus would be useless if the model had
  only ever seen 2×2 and 3×3.
- **Above 3×3 the determinant is taught by elimination, not cofactors.** Row
  addition leaves the determinant alone and the only sign is the number of row
  interchanges — a route that does not collapse at 5×5. The scratchpad shows
  the eliminations and the diagonal product.
- **Every derivation ends with a check,** and the answer is graded by sympy, so
  a fabricated answer is visible rather than plausible.

### If the step times swing

A training step that takes 11 seconds one time and 32 seconds the next, with
no trend, is not a slow GPU. It is the GPU being interrupted.

Windows time-slices the GPU between processes through the display driver. A
browser with a video playing, a video call, a container runtime or an
overlay all hold a context on the card, and the scheduler will hand them a
slice in the middle of your training step. The interruption length is not
predictable, which is exactly what the variance looks like.

`make preflight` now asks nvidia-smi what the GPU is doing and warns if it
is more than 5% busy. Close the browsers, the video call and the container
runtime before a long run, and watch the step times settle.

One counter is not comparable across platforms. PyTorch's caching allocator
reports *reserved* memory, and on Windows the driver can map host memory
into the device address space, so the reserved figure can exceed the card's
dedicated VRAM. The *allocated* figure is the one to watch: 7.8 GB of 12.9 GB
is comfortable.

### Check the machine first

```bash
make preflight
```

Prints the Python and torch versions, the CUDA build, the GPU name and its
VRAM, whether 4-bit quantisation actually works on *this* machine, whether
the corpus is present, and which precision the training run will use. It is
the difference between a dozen lines and a thousand-line traceback.

Make does not search upward for a makefile, so `scripts/Makefile` forwards
every target to the root one. `make preflight` works from anywhere in the
tree, including from inside `scripts/`.

### If bitsandbytes fails to load

`bitsandbytes` ships pre-compiled binaries for specific CUDA versions — 11.8
through 13.0 at the time of writing. If your torch is built against a CUDA
release it has no wheel for, the import **succeeds** and the failure arrives
later, inside model loading:

```
RuntimeError: CUDA VERSION MISMATCH
Requested CUDA version:          13.2
Detected PyTorch CUDA version:   13.2
Available pre-compiled versions:
  - 11.8 ... 13.0
```

There are two ways out.

**Let the script handle it (the default).** The training script probes
bitsandbytes with a real 4-bit quantisation on the GPU *before* it commits to
the path. If the probe fails it prints why, falls back to a LoRA on a bf16
base, and carries on:

```
4-bit quantisation is unavailable: CUDA VERSION MISMATCH ...
falling back to a LoRA on a bf16 base. It fits in about 7 GB on a 12 GB card,
so the run proceeds; only the memory headroom is smaller.
```

A LoRA on a bf16 base needs roughly **7 GB** in total — the frozen weights are
3.44 GB, the adapters and their optimiser state about 0.5 GB, and the
activations and logits the rest — so it fits a 12 GB card comfortably. The only
thing you give up is about 2.5 GB of headroom. Force either path with
`--precision 4bit` or `--precision bf16`.

**Upgrade bitsandbytes (simplest — keeps your torch).** The current
bitsandbytes, 0.50.2, bundles native libraries for CUDA **11.8, 12.1, 12.4,
12.6, 12.8, 13.0 and 13.2**. The mismatch only happens with an *older*
bitsandbytes, whose newest bundled library is 13.0, meeting a torch built
against CUDA 13.2:

```bash
pip install -U bitsandbytes==0.50.2
```

`make preflight` prints the CUDA builds the installed bitsandbytes actually
ships, read out of the package directory, and compares them against your
torch. If it says `cuda match YES`, the problem is gone.

**Or move torch down** to a CUDA version every bitsandbytes has. This is what
`scripts\setup_windows_gpu.bat` does on Windows:

```bat
scripts\setup_windows_gpu.bat
```

It installs torch 2.12.1+cu130 and torchvision 0.27.1+cu130 from the PyTorch
index, installs bitsandbytes with `--no-deps` so pip cannot drag torch back
to PyPI, and restores the packages a previous attempt bumped. By hand:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.12.1+cu130 torchvision==0.27.1+cu130
pip install --no-deps bitsandbytes==0.50.2
```

Going this way has one advantage: torchvision 0.27.1 pins `torch==2.12.1`,
so it clears that conflict on the way past.

**Do not run `pip install --force-reinstall bitsandbytes`.** It reinstalls
every dependency at its newest version, which drags torch in from PyPI and
bumps `fsspec` and `filelock` past the versions `datasets` and `virtualenv`
accept. Use `-U` and `--no-deps` instead. If you have already run it, put
the two back:

```bash
pip install "fsspec[http]==2026.7.0" "filelock>=3.24.2,<4"
```

`constraints.txt` pins them for future installs. `python -m bitsandbytes`
prints detailed diagnostics about what it found and what it expected.

One more thing worth checking: **Python 3.14 is very new** and the torch and
bitsandbytes wheel coverage for it is still catching up. If anything else
misbehaves after the CUDA issue is resolved, a 3.12 or 3.13 environment is
the safe choice.

### Evaluate

```bash
make eval
```

Prints the four numbers that matter, and the per-family and per-difficulty
breakdown:

```
  answer correct        142/200  (71.0%)
  plot kind correct     168/200  (84.0%)
  plot values correct   121/200  (60.5%)
  abstained              14/200  (7.0%)
```

**The abstention rate is not a footnote.** A model that always answers has a
100 % rate on the problems it guessed and 0 % on the ones it did not, and the
headline number cannot tell you which is which. A model that abstains correctly
on a fifth of the problems is more useful than one that confabulates the same
fraction.

### Ask it something

```bash
python -m la.chat --adapter runs/vslm
python -m la.chat --adapter runs/vslm --question "What is the determinant of [[2,1],[1,2]]?"
```

The front end parses the answer and the plot spec, and tells you if either
failed to check out.

---

## Layout

```
src/la/
  problems.py      the twelve generators; sympy computes every answer
  augment.py       question templates and matrix relabelling
  answer.py        the answer grammar: printer, parser, comparator
  schema.py        the plot-spec schema, validator and feature extraction
  render.py        plot spec -> PNG, ten figure kinds
  build_dataset.py the corpus builder
  train.py         the QLoRA training loop
  evaluate.py      the evaluation harness
  chat.py          the front end
  inference.py     loading, generation, parsing a completion
tests/              the unit tests
configs/default.yaml
```

## Checking the corpus

The scratchpads are the training signal, so a derivation that does not add up
is worse than no training at all. `make check-arithmetic` walks every family,
finds every line that states an expression and a result, evaluates the
expression with sympy and compares it with the result it claims:

```
$ make check-arithmetic
  ...
checked 3400 problems across 17 families
every printed expression evaluates to the value it claims
```

It caught a real one. The 3x3 cofactor expansion printed the signed minors
and left out the matrix element, so

```
STEP 1 expand along the first row: det(A) = (+1)*(-4) + (-1)*(-10) + (+1)*(2) = 4
```

evaluated to 8. The ground truth was right, so no test that compared the
answer with the truth would ever have noticed. It is fixed, and
`tests/test_scratchpad.py` now runs the check over every family on every
test run, along with a check that the step numbers are contiguous.

## The tests

```bash
make test
# or
python -m unittest discover -s tests -v
```

They are written on `unittest`, so they run with the standard library alone;
`pytest` runs them unchanged if it happens to be installed. They cover the
parts that can silently rot: the answer grammar round trip (print, parse,
compare), the plot-spec validator, the renderer on every figure kind, the
problem generators, and the augmentation.

Twenty-five tests, about twenty seconds.

---

## Notes and limits

- **Scope.** Introductory through applied numerical linear algebra. Functional
  analysis and infinite-dimensional theory are out. If you widen it, widen the
  families to match.
- **The graded answer is the mathematics.** For `orthonormal_basis` a basis is
  only determined up to the sign of each vector, and for `diagonalize` up to the
  order of the diagonal entries, so those two are compared with the
  corresponding tolerance rather than exactly.
- **A plot is not a proof.** The harness compares the numbers in the plot
  against the numbers in the reference, not the pixels. That is deliberate:
  it tests the mathematics, and a rendering difference should not be able to
  fail a correct answer.
- **The scratchpad is a training target, not a guarantee.** The model will
  sometimes write arithmetic that does not check out. The host-side verifier
  is what makes that visible.
- **Copyright.** This repository generates its own data and downloads nothing.
  If you extend it with human-written textbooks, check the licence before you
  redistribute the resulting weights.

Apache-2.0.
