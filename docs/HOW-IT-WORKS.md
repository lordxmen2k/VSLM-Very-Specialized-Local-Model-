# How VSLM works

The README says what the project does. This document says how, and why each
piece is the way it is. It is written for someone who wants to build
something like this, or wants to understand why a small fine-tune of a
1.7B model can beat a 235B model on linear algebra.

The mistakes the project made are in here too, in
[Bugs this project found](#bugs-this-project-found). That section is the most
useful part of the document, because every one of them was invisible to the
obvious test.

---

## The pipeline

```
sympy                          src/la/problems.py
  │  generates a matrix, computes the answer,
  │  writes the derivation, builds the plot spec
  ▼
augmentation                   src/la/augment.py
  │  asks the question three to five ways,
  │  relabels the matrix, recomputes the answer
  ▼
the corpus                     src/la/build_dataset.py
  │  data/train.jsonl, one JSON object per line
  ▼
tokenisation                   src/la/train.py
  │  prompt tokens masked, completion tokens supervised
  ▼
QLoRA                          src/la/train.py
  │  4-bit NF4 base, rank-32 LoRA, paged 8-bit Adam
  ▼
the adapter                    runs/vslm/
  │  34,865,152 trained parameters, 2 GB
  ▼
evaluation                     src/la/evaluate.py, src/la/bench.py
   │  parse the answer back into sympy, compare with the truth
   ▼
the score
```

Nothing in that diagram is allowed to be an approximation. Every arrow is a
function that either works or fails loudly.

---

## 1. The corpus

### The oracle

sympy is the oracle. For a random matrix `A`, the determinant is
`sp.simplify(A.det())`, the spectrum is `A.eigenvals()`, the singular values
are `A.svd()`, and so on. The generator writes the same values into three
places: the answer, the derivation and the figure.

That is the whole reason the ground truth cannot drift. There is no second
copy of the answer, maintained by hand, to fall out of step with the question.
The usual failure mode of a hand-built corpus — someone changes the matrix and
forgets the answer — has no mechanism to occur.

### Why the derivation is generated rather than written

A human-written derivation for a random matrix is a research project. A
generated one is a function of the matrix. `gen_determinant` for a 3×3:

```python
cofactors = []
for j in range(3):
    minor = A.minor_submatrix(0, j)
    sign = 1 if j % 2 == 0 else -1
    coef = sp.simplify(A[0, j] * sign)
    cofactors.append(f"({fmt_scalar(coef)})*({fmt_scalar(sp.simplify(minor.det()))})")
steps.append(step("expand along the first row: det(A) = "
                  + " + ".join(cofactors) + f" = {fmt_scalar(det)}"))
```

Every number in that string comes from sympy. The cofactor sign is folded into
the element, so the printed expression is the determinant and not the sum of
the signed minors. That detail is not cosmetic — see
[Bugs this project found](#bugs-this-project-found).

### Augmentation

Three things are varied:

**The wording.** Each family has three to five templates:

```python
TEMPLATES: dict[str, list[str]] = {
    "determinant": [
        "Find the value of the determinant of {stem}.",
        "What is det({stem})?",
        ...
    ],
```

A model trained on one phrasing learns to match the phrasing. Three to five
ways teaches it to match the *question*.

**The matrix.** Rows are permuted (`_permute_rows_only`), which changes the
determinant's sign and therefore forces the model to actually compute rather
than recall. Similarity transforms (`A → S A S⁻¹`) preserve the spectrum
exactly, which is the point: the eigenvalues must come out the same, so a
model cannot cheat by remembering.

**The scale.** The size mix is deliberate:

```python
n = rng.choice([2, 2, 3, 3, 4, 4, 5, 5])
```

Two fifths of the determinants are 5×5, which is where the frontier models
break. See below.

### The elimination decision

LinAlg-Bench reports that the failures at 5×5 are not arithmetic slips, and
its error taxonomy says what they are. Every failure is classified into a
taxonomy with thirteen categories, and for determinants the sign-error
subtypes are:

```
Product_Sign_Error   Operation_Direction   Rule_Interference
Parity_Sign_Error    Double_Negative_Trap   Alternating_Drift
Cofactor_Neglect     Silent_Sign_Flip
```

Two of them describe a failure that only a cofactor expansion can produce.

**`Alternating_Drift`**, from the benchmark's own judge prompt: *"Model
correctly applied (−1)^{1+j} for the first few cofactors but then lost the
+−+−+ checkerboard pattern. The first wrong value is a cofactor with correct
magnitude but wrong sign because the pattern decayed after 2–3 terms. DETECT:
Model gets C_11=+M_11, C_12=−M_12, C_13=+M_13 correct, then writes C_14=+M_14
instead of −M_14. Or inside a nested 4×4 minor expansion, loses the pattern
midway."*

**`Cofactor_Neglect`**: *"The model computed the minor value correctly but
used it directly in the final assembly without applying the (−1)^{i+j} parity
factor. The parity step was simply skipped entirely. DETECT: Model computes
M_12 = 220 correctly, then writes det(A) = ... + a_12 · M_12 instead of ...
+ a_12 · (−1) · M_12."*

A 5×5 cofactor expansion *is* a nested 4×4 minor expansion — five of them,
each expanded into four 3×3 minors, each of those into three 2×2 minors.
The full expansion has 5! = 120 scalar terms and the alternating ± pattern
has to survive all of them. That is what `Alternating_Drift` describes, and
the benchmark says it happens.

The size arithmetic is not the main point, but it is worth having: a full
cofactor expansion of an n×n determinant has n! scalar terms — 6 at 3×3, 24
at 4×4, 120 at 5×5. A model that has learned "expand along the first row" as
a *shape* has to hold all 120 terms and their signs, and it fails.

Gaussian elimination by row addition is different. Each row operation is
`row_r -= k · row_c`, the determinant is unchanged, and the only sign in the
whole procedure is the parity of the row interchanges. The scratchpad shows
the operations explicitly:

```
STEP 1 eliminate below each pivot by adding a multiple of one row to another,
       which leaves the determinant alone
  row 2 -= (-1) * row 1
  row 3 -= (4) * row 1
STEP 2 the upper triangular form has diagonal -1, -2, 18
STEP 3 det(A) = -1 * -2 * 18 = 36
```

The procedure does not collapse with size. It is longer on the page and much
easier to follow.

So the corpus teaches cofactors below 4×4 — where they are the natural thing
and the expansion is short enough to hold — and elimination above it. A model
that has only ever seen 2×2 and 3×3 would be useless at exactly the size the
benchmark cares about.

**And the benchmark confirms the choice at 3×3.** Of the fifty 3×3
determinant problems in LinAlg-Bench, eighteen say *"Use cofactor expansion
along the first row"*. The 4×4 and 5×5 problems specify no method at all. So
the corpus teaches the method the benchmark asks for at each size: cofactors
where the benchmark asks for them, elimination where it leaves the choice
open and the cofactor route collapses.

That is also a hard constraint on the corpus. The taxonomy is judging models
on exactly the step this project once got wrong. Our 3×3 expansion printed
the signed minors and left out the matrix element — the mirror image of
`Cofactor_Neglect`, which skips the parity factor instead — and it produced
an expression that was not the determinant. A corpus that exhibited either
error would be teaching the failure the benchmark is built to measure.

### The VERIFY line

Every derivation ends with a check that the generator has actually performed:

```
STEP 4 VERIFY the determinant of the same matrix is 36. OK
```

Three reasons this is in the data rather than bolted on afterwards:

1. **It teaches a habit.** A model that writes the check has to state what it
   would check, which is a different act from stating the result.
2. **It makes the training target self-consistent.** If a derivation and its
   answer disagree, the model is being taught to contradict itself. The
   VERIFY line is where that shows up.
3. **It makes the output auditable.** A reader — or a program — can check the
   last line of every answer without trusting the middle.

The verification is not a formality. It is computed: for `determinant` the
generator re-evaluates the determinant of the same matrix and compares.

### The arithmetic checker

`make check-arithmetic` walks every family, finds every line of every
scratchpad that states an expression and a result, evaluates the expression
with sympy, and compares it with the result claimed:

```python
m = ENDS_IN_NUMBER.search(line)
rhs = m.group(1)
head = line[:m.start()]
cut = head.rfind("=")
lhs = head[cut + 1:].strip()
expr = _to_expr(lhs)
...
if sp.simplify(got - want) != 0:
    complaints.append(...)
```

The checker only looks at lines whose left-hand side is plain arithmetic —
digits, `+ - * / ( )` — so a line containing LaTeX is skipped rather than
misparsed. It checks 3,400 problems across all seventeen families in about
forty seconds.

This exists because of a specific failure: the ground truth was right, the
answer was right, and the *derivation printed in between* was wrong. No test
comparing the answer with the truth can see that. Only reading the derivation
and doing the arithmetic can.

---

## 2. The answer grammar

The model writes its answer in LaTeX. To grade it, that LaTeX has to become
something sympy can compare. A general-purpose LaTeX parser would be the
obvious choice and a bad one: `sympy.parse_latex` needs antlr, is slow, and
mis-parses a lot. The answer grammar here is small and written by hand:

```
answer   := pairs | set | matrix | scalar
pairs    := label '=' value (',' label '=' value)*
set      := '{' value (',' value)*
matrix   := '\begin{pmatrix}'|'\begin{bmatrix}' row ('\\' row)*
            ('\end{pmatrix}'|'\end{bmatrix}')
value    := scalar | matrix | set | pairs
scalar   := term (('+' | '-') term)*
term     := factor (('*' | '/') factor)*
factor   := '-' factor
         | number
         | '\frac' '{' scalar '}' '{' scalar '}'
```

`la.answer` implements it three ways — a printer, a parser and a comparator —
and the round trip is unit tested: print a sympy object, parse it back, check
the two are equal. `answers_equal` then handles the two families where exact
equality is the wrong test: an orthonormal basis is determined up to the sign
of each vector, and a diagonalisation up to the order of the diagonal entries.

The comparator is what makes the evaluation honest. The model is never asked
whether it agrees with itself. Its answer is parsed into a mathematical object
and compared with sympy's.

### Why the grammar accepts two matrix environments

LinAlg-Bench writes every matrix as `\begin{bmatrix}`. This corpus writes
both spellings, chosen from the contents of the matrix:

```python
def matrix_environment(body: str) -> str:
    return "bmatrix" if (zlib.crc32(body.encode("utf-8")) >> 16) & 1 else "pmatrix"
```

Deriving the choice from the contents rather than from a random number keeps
the same matrix written the same way within an example — the question and the
scratchpad agree — while different matrices in the same example may use
different environments. That variety is the point: the model should be
indifferent to the spelling.

The grammar has to accept both, and it did not at first. See
[Bugs this project found](#bugs-this-project-found).

---

## 3. The plot spec

The model does not draw. It emits a small JSON object, and `la.render` turns
that into a PNG deterministically.

```json
{"kind":"projection",
 "data":{"vectors":[[0.6,1.0],[1.4,0.2]]},
 "labels":{"title":"projection of v onto w"},
 "annotate":["orthogonal projection"]}
```

There are ten kinds — `basis_2d`, `basis_3d`, `eigenvalue_spectrum`,
`row_echelon`, `subspace_2d`, `projection`, `least_squares`,
`singular_spectrum`, `matrix_heatmap`, `transform_unit_square` — each with a
schema, a validator and a renderer.

Three reasons for this arrangement:

- **A description is a smaller competence than a drawing.** Learning to
  specify a scatter plot is a matter of choosing a kind and filling in a
  couple of arrays.
- **The numbers in the picture are the numbers in the answer.** Both come
  from the same generator, so a figure cannot disagree with the result.
- **The evaluation compares numbers, not pixels.** `spec_features` extracts
  the numbers from the model's spec and from the reference, and compares
  them. A rendering difference cannot fail a mathematically correct answer.

---

## 4. The training recipe

| | |
|---|---|
| Base model | `Qwen3-1.7B-Base` — 28 layers, hidden 2048, 16 query heads, 8 KV heads, head dim 128, tied embeddings |
| Quantisation | 4-bit NF4, double quantisation, bf16 compute |
| Adapters | LoRA rank 32, alpha 64, on all seven projections |
| Trainable | 34,865,152 of 1,755,440,128 parameters — 1.99 % |
| Sequence length | 768 (the corpus has a median of 344 tokens and a tail past 512) |
| Batch | 2, with 8-way accumulation — an effective batch of 16 |
| Gradient checkpointing | on |
| Optimiser | `paged_adamw_8bit`, lr 1e-4, cosine, 3 % warmup, weight decay 0.01 |
| Epochs | 1 |

### Why each of those

**4-bit NF4** is what makes it fit. The base weights go from 3.44 GB to about
0.9 GB, and with gradient checkpointing the whole run sits at 7.8 GB
allocated — comfortably inside 12 GB, with room for the logits, which are the
memory-hungry part at a vocabulary of 151,936.

**LoRA rather than a full fine-tune** is not a compromise for this project; it
is the point. A full fine-tune needs 27.5 GB for the optimiser state alone.
LoRA needs 2 % of the parameters trained, which means the model has to learn
a narrow, structured behaviour — emit a derivation, an answer and a plot spec
in a fixed format — rather than relearn linear algebra. Both the format and
the arithmetic are in the prompt and the corpus; the adapter only has to learn
to use them.

**Gradient checkpointing** trades compute for memory: the activations of each
layer are recomputed during the backward pass instead of being kept. On a
12 GB card that is the difference between fitting and not fitting.

**An effective batch of 16** at a learning rate of 1e-4 is a conservative
setting, which is the right default for a run you might repeat.

**One epoch** over 20,000 procedurally generated examples is 7.2 M tokens.
Because the corpus is generated, the supply of distinct examples is not the
constraint — the constraint is what one card can get through. A second pass
costs twice as much and is usually worth less than twice as much, since the
model sees the same seventeen shapes either way.

### The measurement that set the batch size

`make speed-test` runs the same data through five micro-batch shapes at the
same effective batch of 16, on an RTX 4070 SUPER:

```
   batch  accum  eff   s/step     tok/s    ex/s  alloc GB  resv GB   frag
       1     16   16     6.87       814     0.1      3.68     4.33   0.66
       2      8   16     3.58     1,633     0.6      5.04     7.45   2.41
       4      4   16    24.40       231     0.2      7.78    13.68   5.90
       8      2   16    36.36       169     0.2    13.24    26.12  12.88
      16      1   16    29.04       191     0.6    24.03    27.97   3.93
```

Read the `resv GB` column against the card's 12.88 GB. Batch 1 and batch 2
stay under it and are fast. Batch 4 crosses it and is **seven times slower**.
Batch 8 and 16 cross it by a wide margin and are slower still.

The cause is memory, not arithmetic. The logits dominate: a bf16 copy, an
fp32 copy for the loss, and the gradient of that copy, over a vocabulary of
151,936. That term scales with micro-batch × sequence length, and once it
passes the card's physical VRAM the caching allocator spills into host
memory mapped into the device address space. Every access to it crosses PCIe.

**The rule that follows: take the fastest row whose reserved memory stays
under the card.** Batch 2 with 8-way accumulation, at 1,633 tokens per second,
twice as fast as batch 1 and seven times faster than batch 4 — on the same
effective batch and the same tokens, so the training dynamics are identical.

The budget follows directly. The corpus is 7.2 M tokens per epoch, so one
epoch is **73 minutes**.

### The token budget, measured

The supervised part of an average example:

| part | characters | share of supervised text |
|---|---|---|
| scratchpad steps | ~470 | 73 % |
| plot spec | 196 | (25 % of the whole example) |
| GIVEN line | 105 | 17 % |
| VERIFY line | 62 | 10 % |

The GIVEN line restates the matrix, which is already in the question. It is
17 % of the supervised text spent copying. The VERIFY line is a deliberate
teaching device and stays. The plot spec is a quarter of the example, but only
58 of its 196 characters are the matrix being copied — the rest is JSON
scaffolding — so redesigning the spec format to reference the matrix would
save about 8 % and would break the contract between the model and the
renderer. Not worth it.

---

## 5. Reading the benchmark score

`make bench-all` scores the adapter on all 660 LinAlg-Bench problems and
prints the result next to the published table, with the base model scored on
the same problems as a baseline.

The comparison worth making is **against the base model**, not against the
frontier models. The frontier table is there to show where the difficulty
lies; the base model is there to show what the fine-tune bought. Everything
else — the grader, the questions, the hardware — is held fixed.

Two things to watch for:

- **Abstention.** A model that answers everything has a 100 % rate on what it
  guessed and 0 % on what it did not. The abstention count separates those.
- **The 5×5 column.** It is where the frontier models fall apart and where
  this corpus was aimed. If the gap between the 4×4 and 5×5 columns is much
  smaller than the published gap, the elimination training did what it was
  supposed to.

---

## Bugs this project found

Every one of these was invisible to the test that existed at the time. They
are written down because the pattern is the point: **the obvious test does
not catch the interesting bug.**

### The derivation that did not add up

The 3×3 cofactor expansion printed the signed minors and left out the matrix
element:

```
STEP 1 expand along the first row: det(A) = (+1)*(-4) + (-1)*(-10) + (+1)*(2) = 4
```

That expression evaluates to **8**. The ground truth was 4 and was always
right, so every test that compared the answer with the truth passed. The
model was being taught to write a derivation whose own arithmetic does not
work, and then to state the right answer anyway — which is precisely the
fabrication failure the corpus was built to counter.

Found by writing `la.check_arithmetic`, which evaluates the printed
expression rather than trusting the answer.

### The benchmark's LaTeX that the corpus had never seen

LinAlg-Bench writes `\begin{bmatrix}` for all 340 matrices in each file. The
corpus wrote `\begin{pmatrix}` for all of its own. The model would have met
the benchmark's spelling for the first time on benchmark day.

Fixed by randomising the environment. But the answer grammar's lexer knew
only `\begin{pmatrix}`, so a `bmatrix` answer would have failed to parse and
scored **wrong** — for every family whose answer is a matrix. Fixing the
corpus without fixing the grammar would have turned a distribution shift into
a scoring bug.

### Six hours for a training run

The first configuration — batch 1, 16-way accumulation, 2048-token sequences —
measured 9.0 s per optimiser step, which is six hours for two epochs. The
README had claimed thirty minutes for the whole run.

That claim had never been measured. It was a guess with false precision, and
it was in the README, which is the document people trust. Changing the batch
size to 4 did not help: 21.6 s per step, on the same effective batch and the
same tokens. The explanation offered for that — an under-occupied GPU — was
also wrong. `make speed-test` now measures five micro-batch shapes against
each other so the question can be answered with a number instead of a theory.

### The override that did nothing

The Makefile documented

```bash
make train TRAIN_ARGS="--epochs 3 --lr 1.5e-4"
```

and the recipe never referenced `$(TRAIN_ARGS)`. The override was silently
discarded, so a smoke test asked for with it would have run the full training
instead. It is the kind of bug that only shows up when someone actually
follows the documentation.

### The counter that could not be true

A training run reported 7.78 GB allocated and **16.65 GB reserved** on a card
with 12.9 GB of memory. Reserved memory cannot exceed physical memory on one
device, so the number had to mean something other than what it appeared to
mean: on Windows the driver can map host memory into the device address
space, so the caching allocator's reserved figure is not comparable across
platforms. The *allocated* figure is the one to watch.

### Seven times slower, for a reason that was not the reason

For most of a day the belief was that a 250-token sequence at batch 1 left
the GPU under-occupied, and that batching the micro-batches four times larger
would fix it. It did not. Batch 4 measured *worse* than batch 1 — 24.40 s
per step against 6.87 — and the obvious explanation was ruled out by the
measurement.

The real cause was in a column nobody was reading. The reserved-memory figure
crossed the card's physical VRAM between batch 2 and batch 4, and every
configuration on the far side of that line was about seven times slower. The
logits are the reason: at a vocabulary of 151,936, a bf16 copy, an fp32 copy
and a gradient over micro-batch × sequence length is the dominant term, and
past the card's limit the allocator spills to host memory over PCIe.

Two things were wrong with the earlier diagnosis and both were mine. I
proposed a cause that the measurement did not support, and I attributed a
step-time variance to "a slow GPU" when the real cause was a shared GPU. The
speed test exists now because of that, and it prints the reserved figure and
flags any row that crosses the card.

### The step times that would not settle

Step times swinging between 11 and 32 seconds with no trend looked like a
broken benchmark, and two of the explanations offered for it were wrong. The
real cause was that the GPU was being shared: Windows time-slices it between
processes, and a browser, a video call or a container runtime will take a
slice in the middle of a training step for an unpredictable length of time.
`make preflight` now asks nvidia-smi whether the GPU is busy.

---

## Extending it

### A new problem family

1. Write `gen_<name>(rng)` in `src/la/problems.py`. It returns a `Problem`
   with the question, the scratchpad, the answer and the plot spec. Compute
   every number with sympy.
2. Register it in `GENERATORS`.
3. Add it to `TEMPLATES` in `src/la/augment.py` so it gets asked several ways.
4. Add it to `la.check_arithmetic`'s reach and run `make check-arithmetic`.
5. Add a test to `tests/test_problems.py`.

If the family's answer is a matrix, the answer grammar already handles it. If
the answer is something new — a polynomial, a set of values, a labelled
quantity — extend the grammar in `src/la/answer.py` and add the round-trip
test.

### A new figure kind

1. Add the kind to `SPEC_KINDS` in `src/la/schema.py`, with its required
   data keys.
2. Add the renderer branch in `src/la/render.py`.
3. Add it to `spec_features` so it can be compared numerically.
4. Add a case to `tests/test_schema_render.py`.

### A different recipe

The defaults are in `src/la/train.py` and are asserted by
`tests/test_training_config.py`, so changing one means changing the test —
deliberately. The memory table in the README is a guide, not a promise: run
`make preflight` and read the allocated figure on your own card.

---

## Licence and provenance

Apache-2.0. Everything here is written for this project.

The base model is `Qwen3-1.7B-Base`, under its own licence. The benchmark is
LinAlg-Bench (NeurIPS 2026 Datasets & Benchmarks), CC-BY-4.0, downloaded at
run time and not redistributed here. The corpus is generated locally and is
not redistributed here either.
