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
| Batch | 1, with 16-way accumulation — an effective batch of 16 |
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
same effective batch of 16, on an RTX 4070 SUPER, each shape in a process of
its own:

```
   batch  accum  eff   s/step     tok/s    ex/s  alloc GB  resv GB   frag  free GB
       1     16   16     4.19     1,256     0.2     10.01    12.02   2.00     8.78
       2      8   16    24.08       218     0.1     16.09    22.42   6.33     8.78   OVER THE CARD
       4      4   16    out of memory
       8      2   16    out of memory
      16      1   16    out of memory
```

**One shape fits.** A batch of two needs 16.09 GB of live tensors on a card
with 12.88 GB, so it cannot run at 768 tokens however much of the card is
free.

The figure to read is `alloc`, not `resv`. The finished training run
reserved 13.35 GB on a 12.88 GB card and ran perfectly well, because on
Windows the driver maps host memory into the device address space and the
caching allocator counts that against its reserved pool. Live tensors are
the constraint; the reserved pool is not.

The cause is the vocabulary, not the model. The base weights are under 1 GB.
The logits are the whole problem: a bf16 copy, the fp32 copy the loss is
computed in, the log-softmax, and the gradients of all of it come to about
**6 GB per example** over 151,936 tokens. That term scales with micro-batch
× sequence length, and once it passes the card's physical VRAM the caching
allocator spills into host memory mapped into the device address space.
Every access to it crosses PCIe, which is why batch 2 measures 218 tokens a
second instead of 1,256.

Batch 1 is also the *right* answer rather than merely the only one. At
batch 1 a batch's width is that example's own length, so there is no padding
waste and the peak is set by the longest example rather than by the cap. At
batch 2 the width is the longest of the two, so the wasted padding is real.

**The budget follows from the real run, not from this.** The speed test runs
seven steps on a card that has not warmed up. A real run is two and a half
hours of sustained load and the clocks come down.

The finished run: 1,250 steps in 153.4 minutes, 7.36 s per step against the
speed test's 4.19, loss 1.3465 down to 0.0661, 4.36 GB allocated, 704 tokens
per second sustained. **Plan on about two and a half hours per epoch.**

The speed test is still worth having — it is what told us that batch 2 was
impossible — but it measures a cold card and seven steps, and it should be
read as a comparison between shapes rather than as a prediction of wall-clock
time.

### The token budget, measured

The supervised part of an average example:

| part | characters | share of supervised text |
|---|---|---|
| scratchpad steps | ~410 | 75 % |
| plot spec | 196 | (25 % of the whole example) |
| GIVEN line | 0 | 0 % |
| VERIFY line | 62 | 11 % |

Two of those numbers used to be much worse, and both fixes were worth making.

**The GIVEN line used to reprint the matrix.** The question already contains
it — the corpus builds the question from the same matrix — so the line was a
pure copy task, 17 % of the supervised text spent transcribing. It now names
the objects instead:

```
GIVEN: A, b
STEP 1 det(A) = -48 != 0, so A is invertible and the solution is unique.
```

Removing it took the corpus from 781 to 672 characters per example, fourteen
per cent, and it is a better training signal: the model is no longer being
asked to copy.

**The orthonormalisation step used to print the same matrix twice.** Once as
the result of the step, and once again in the normalise line, identically. It
also mislabelled the quantity: the code computed `w · w`, which is the
*squared* norm, and printed it as `||w||`. It now says what it means:

```
STEP 1 w1 = v1 (there is nothing to subtract yet) gives w1 = <matrix>
STEP 1b the squared norm of w1 is 1, so w1 is already a unit vector
```

That family was the longest in the corpus at 1,441 tokens for the largest
instance; it is now 992.

The VERIFY line is a deliberate teaching device and stays. The plot spec is a
quarter of the example, but only 58 of its 196 characters are the matrix being
copied — the rest is JSON scaffolding — so redesigning the spec format to
reference the matrix would save about 8 % and would break the contract
between the model and the renderer. Not worth it.

---

## 5. What the first trained model showed

The first full run — 1,250 steps, 153 minutes, loss 1.3465 down to 0.0661 —
scored **46 %** on the held-out set. The per-family breakdown was the
interesting part:

| answer type | families | score |
|---|---|---|
| a scalar | eigenvalues, matrix_trace, matrix_transpose, rank_nullspace, svd | **100 %** |
| a scalar, but sign-sensitive | determinant | 38 % |
| a matrix or a vector | condition_number, least_squares, lu_factor, matrix_multiply, matrix_power, matrix_vector, solve_system | **0–15 %** |

Every family whose answer is a matrix or a vector failed. Every family whose
answer is a scalar succeeded. That is not noise; it is the shape of the
corpus.

### The corpus stated results instead of deriving them

Here is what the corpus asked the model to produce for a matrix product:

```
GIVEN: A, B
STEP 1 the entry (i, j) of AB is the dot product of row i of A with column j of B
STEP 2 AB = egin{bmatrix} 4 & 0 & 2 & 16 \ ... \end{bmatrix}
STEP 3 VERIFY det(AB) = det(A) det(B). OK
```

The sixteen dot products are never shown. The model is asked to emit a
sixteen-term sum in a single forward pass, and it gets it wrong. The same
was true of `matrix_vector` (three dot products, none shown), `matrix_power`
(the multiplication, not shown), `solve_system` (an inverse matrix, not
derived), `least_squares` (AᵀA and Aᵀb, not shown), `lu_factor` (the
elimination, not shown) and `condition_number` (the singular values, not
computed).

The two families that scored well despite the same shape are the two where
the result is exact: `svd` at 100 % and `condition_number` at 0 %, on the
same structure. The difference is whether the singular values can be
recalled exactly. They could not.

This is the failure the corpus was built to prevent, and it was in the
corpus. A model trained on derivations that assert results learns to assert
results.

### The fix: show the arithmetic

Every derivation now computes its own intermediate values.

```
GIVEN: A, B
STEP 1 the entry (i, j) of AB is the dot product of row i of A with column j of B
STEP 2 (AB)_{11} = -2*4 + (-3)*3 + 1*(-3) = -20
STEP 3 (AB)_{12} = -2*1 + (-3)*4 + 1*1 = -13
...
STEP 11 AB = egin{bmatrix} -20 & -13 & 12 \ 11 & 6 & -6 \ 4 & 4 & -4 \end{bmatrix}
STEP 12 VERIFY det(AB) = det(A) det(B). OK
```

Each line needs one row of A and one column of B, and the final matrix is a
transcription. The partial sums are left out — the terms are the hard part
of the line, and writing every partial sum out doubled the length of every
matrix example in the corpus. It also made the arithmetic checker validate
the *terms* rather than a sum of them, which is a stronger check.

The same treatment went into the determinant, with the cofactor sign
written as its own factor:

```
STEP 1 expand along the first row, with the cofactor signs +1, -1, +1:
       det(A) = (+1)*(-1)*(-4) + (-1)*(-1)*(-10) + (+1)*(5)*(2) = 4
```

The alternating pattern is now something the model can copy. Folding the
sign into the element — which is what the corpus used to do — hides the
pattern and leaves the model to compute it, and that is exactly where the
benchmark's "cofactor-locked" failures come from. The elimination path now
shows how each multiplier was found:

```
  the multiplier for row 2 is (-3)/(-2) = 3/2, so row 2 -= (3/2) * row 1
```

### The second run

The corpus with the four benchmark-critical families fixed was trained on an
RTX 4070 SUPER, one epoch, 1,250 steps, effective batch 16, sequence cap
1,024:

| | run 1 | run 2 | change |
|---|---|---|---|
| time | 153.4 min | 142.7 min | −7 % |
| seconds per step | 7.36 | 6.85 | −7 % |
| final loss | 0.0661 | **0.0462** | **−30 %** |
| peak allocated | 4.36 GB | 5.04 GB | +16 % |
| sustained throughput | 704 tok/s | 885 tok/s | +26 % |

The loss is lower at every matched step past the initial drop: −27 % at step
100, −32 % at step 580, −30 % at the end. The model fits the corpus better
when the corpus shows its arithmetic, which is the first evidence that the
change did what it was supposed to.

The throughput gain is probably not the corpus. Run 1 was sharing the card
with a benchmark run that was still going; run 2 had the GPU to itself.

The loss plateaus from about step 700 — it sits between 0.037 and 0.050 for
the last 550 steps while the learning rate decays to zero. The model has
converged on the training distribution, so a second epoch is unlikely to buy
much.

### The other six families

The same defect was in the six families that are not benchmark tasks, and
they were fixed the same way:

| family | before | after |
|---|---|---|
| `solve_system` | `x = A^{-1} b = <the inverse> <b> = <x>` | the elimination on the augmented matrix, then the back substitution, one line each |
| `least_squares` | `x = (A^T A)^{-1} A^T b = <x>` | AᵀA, Aᵀb, and the two dot products of the final solve |
| `lu_factor` | `L = <L> and U = <U>` | the elimination, with the multiplier and the entries it came from |
| `condition_number` | `the singular values of A are ...` | AᵀA, its eigenvalues, and the square roots that are the singular values |
| `orthonormal_basis` | `w1 = v1 gives w1 = <a unit vector>` | the unnormalised wᵢ, the coefficient of each earlier qⱼ, and the rescaling |
| `projection` | `proj = (<v,w>/<w,w>) w = <v>` | the ratio, then each component's product |

The orthonormalisation fix also corrected a labelling error. The corpus
printed the *normalised* vector and called it wᵢ, which taught the model that
v₁ is already a unit vector. It now shows the unnormalised wᵢ, the
coefficient of each earlier qⱼ, and the division that produces qᵢ.

Two more defects came out of this work, both caught by the arithmetic
checker rather than by any test:

- the `condition_number` line divided two singular values rounded to six
  significant figures and stated the quotient rounded to six significant
  figures, which are not equal. The checker now compares with a relative
  tolerance, because the corpus deliberately states rounded values.
- the `solve_system` back substitution subtracted the known terms twice: the
  right-hand side it printed had already been reduced. The stated answer was
  right and the line that derived it was wrong, which is the exact failure
  the whole project is about.

### What it costs

The corpus went from 672 to 826 characters per example, and the sequence cap
from 768 to 1,024 tokens, because the derivations now have a long tail. One
example in fifteen hundred is still over the cap.

| | before | after |
|---|---|---|
| characters per example | 672 | 826 |
| tokens per example | 329 | 404 |
| corpus, 20,000 examples | 6.5 M tokens | 8.1 M tokens |
| time per epoch at 704 tok/s | 2.5 h | 3.2 h |

`matrix_multiply` and `matrix_power` no longer generate 5×5 instances, and
neither does `solve_system`: a full product at 5×5 is 25 dot products, and
a 5×5 system runs its elimination and back substitution past the cap. The
determinant keeps its 5×5, because its elimination is a handful of lines
rather than a matrix of them — and the determinant is a quarter of the
benchmark, so it matters most.

`lu_factor` falls back to the terse form for the 17 % of matrices whose
elimination needs a pivot swap, because the derivation shown here does not
pivot. Those examples still have the correct answer; they just do not show
the working.

The evaluation now compares with a tolerance of 1e-4 rather than exactly.
The corpus states some values rounded to six significant figures — the
singular values, and the condition number built from them — so a model that
reproduces the same six figures was being marked wrong. An answer that is
off by one in an integer is still rejected.

## 6. What the benchmark said

The trained model was scored on all 660 LinAlg-Bench problems, with the base
model scored on the same problems for the delta.

| | fine-tuned | base | delta |
|---|---|---|---|
| 3×3 | 29.5 % | 65.0 % | **−35.5 points** |
| 4×4 | 0.0 % | 0.0 % | 0.0 |
| 5×5 | 21.4 % | 23.6 % | −2.3 |
| overall | 17.0 % | 29.5 % | −12.5 |

Two things in that table are about the data, and one is about the model.

### The 4×4 column measures the benchmark, not the model

Every model — the fine-tuned one and the base one — scores exactly zero on
the 4×4 problems. That is not a model failing; it is the benchmark failing.

The 3×3 and 5×5 questions say what they want:

```
Find the determinant of the 3×3 matrix A.

A = \begin{bmatrix} -3 & 5 & 1 \\ -2 & -3 & 9 \\ 1 & 1 & -6 \end{bmatrix}

Show all intermediate steps. Put your final numerical answer inside \boxed{}.
```

The 4×4 questions are a bare matrix with no instruction at all:

```
A = \begin{bmatrix} -4 & 1 & 5 & 3 \\ -1 & -2 & 0 & -3 \\ 1 & -5 & 4 & 2 \\ -5 & 2 & -1 & -4 \end{bmatrix}
```

Nothing in that text says whether to compute the determinant, the trace, the
rank or the eigenvalues. The task is in the CSV's `Subcat` column, which the
model never sees. **All 220 of the 4×4 problems are unanswerable as
written**, and a score on that column is a score on the data.

`la.grading.is_instructed` detects this, the benchmark report says how many
problems are affected, and
`tests/test_benchmark_grading.py::test_the_four_by_four_problems_state_no_task`
checks it against the real data. If the benchmark is ever fixed, that test
fails and the note in the report comes out with it.

### The 3×3 regression is about the question, not the mathematics

The fine-tuned model is 35.5 points worse than the base model at 3×3, and the
per-task breakdown says exactly where:

| task | fine-tuned | base | our corpus trains this with |
|---|---|---|---|
| determinant | 1/150 | 25/150 | a derivation |
| matrix_vector | 3/60 | 31/60 | a derivation |
| multiplication | 3/60 | 13/60 | a derivation |
| matrix_power | 1/60 | 14/60 | a derivation |
| eigenvalue | 1/90 | 7/90 | a derivation |
| trace | 35/60 | 38/60 | one line |
| transpose | 40/60 | 37/60 | one line |
| rank | 14/60 | 16/60 | one line |
| nullity | 14/60 | 14/60 | one line |

**The regression is on precisely the tasks whose derivations were expanded.**
The tasks with a one-line answer are flat or slightly better.

And the held-out set — our own questions, our own phrasings — went the other
way: `matrix_vector` went from 15 % to 54 % when the derivations were
expanded. So the derivation training helps on the phrasing it was trained on
and hurts on the benchmark's phrasing.

The benchmark asks for the answer inside `\boxed{}` and adds "Show all
intermediate steps". Our corpus asks for a `### Answer` block and shows the
work in a particular shape. The model has learned our question format, and
the benchmark's is out of distribution for it.

The raw completions settle which of the two mechanisms it is — a format
mismatch in the answer block, or a derivation that runs long enough to crowd
the answer out:

```bash
make explain RESULTS=data/bench_linalg.jsonl
```

## 7. Reading the benchmark score

### The grader is the part that was quietly wrong

Three separate bugs in the grading harness would have made correct answers
score zero. All three are fixed, and all three are now covered by a test
that downloads the real benchmark and checks it.

**The grammar had no rule for a bare list.** LinAlg-Bench writes the
eigenvalues as `\lambda = -1, 0, 3`. Strip the label and you have `-1, 0, 3`
— a comma-separated list with no braces. The grammar had `set` (braces) and
`pairs` (labels), and nothing for that. All ninety eigenvalue problems
failed to parse, which is 14 % of the benchmark.

**The benchmark rounds its eigenvalues.** The references are rounded to four
decimal places, and the largest gap between a reference and the true
eigenvalue is **4.991e-05**. The comparison used a tolerance of 1e-9, so a
correct eigenvalue was rejected. The benchmark comparison now uses 1e-3,
which is twenty times the measured error and still requires the model to be
right to three decimal places. Every other task in the benchmark is exact —
the measured gap for determinant, trace, rank and nullity is zero.

**`\boxed{...}` was extracted with a non-greedy regex.** It stopped at the
first closing brace, which for a matrix answer is the one inside
`\begin{bmatrix}`:

```
\boxed{A^T = \begin{bmatrix} 1 & 2 \end{bmatrix}}
   ->  "\begin{bmatrix"
```

That was 420 of the 660 problems — matrix-valued answers, and the ones
labelled `\text{rank}(A) = 3` where the first brace belongs to `\text{`.
Sixty-four per cent of the benchmark was unscoreable, whatever the model did.

**And the label came off the reference but not off the model's answer.** A
base model answers `\det(A) = 97` inside the box. The reference had its
label stripped; the model's answer did not, so the answer grammar never saw
a value. The baseline would have scored zero on every determinant.

Both the reference and the model's answer now go through the same
normalisation, and `\boxed{...}` is extracted with balanced braces.

The check that matters is the one in
`tests/test_benchmark_grading.py`: it downloads the 660 problems and
asserts that every reference answer parses, that a correct answer scores
correct in all three shapes a model might write it in, and that a wrong
answer scores wrong. Invented test data would not have caught any of these
three bugs.

### The numbers it produces

`make bench-all` scores the adapter on all 660 LinAlg-Bench problems and
prints the result next to the published table, with the base model scored on
the same problems as a baseline.

The base model is prompted in this project's format, which is not the format
the published numbers were obtained in — the published table asked for the
answer inside `\boxed{}` in the benchmark's own phrasing. Treat the base
score as the strict weights-only comparison and the published table as the
reference point, and do not read the base score as "what the base model would
score on the benchmark".

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

### The instrument that was measuring the wrong thing

The speed test ran all five micro-batch shapes in a single process, and
reported the result. The result changed every time it was run, and it
changed in a way that should not have been possible: after the corpus got
*smaller*, one configuration measured *slower* and its reserved memory went
*up*.

The allocated-memory column was stable across runs. The reserved column was
not. The difference is fragmentation: the caching allocator does not give
back memory it has cached, and it cannot always reuse a block for a request
of a different size, so a process that has already run a wider configuration
measures the next one through its own mess. Measuring several shapes in one
process measures the sum of them, not each one.

The tool now runs each shape in a process of its own and reads one
machine-readable row back over stdout. The row format is a contract between
the two processes, so it lives in `la/speed.py` and is unit tested without a
GPU.

**And then the fix introduced a second contamination.** The parent process
loaded the model onto the GPU before it spawned the children, because that
was where the model-loading code already was. So the card was shared with
the parent's copy of the model while the child measured, and every row was
measured against a card that was already partly full. The tell was a batch-1
row reporting 9.76 GB allocated where the same configuration had measured
4.29 GB in the previous, in-process version.

The parent now loads only the tokenizer and returns before the model is
built. There is a test that asserts the ordering, because it is exactly the
kind of thing that gets undone by a well-meaning refactor.

Two more things the tool had to get right:

- **The free VRAM at the start of each measurement is reported.** If the card
  is not free when a row begins, the row is not measuring what it claims to.
  The parent's own free memory is printed before it spawns anything, with a
  warning if the card is already more than 10 % in use.
- **The shuffle is fixed.** The width of a batch decides how wide its logits
  are, and the logits are the dominant memory term, so an unlucky draw can
  change a row's peak memory by more than half. Every configuration now sees
  the same examples in the same order, so the rows are comparable.

Three separate times on this project a measurement was confounded — by a
shared GPU, by a wrong model of the hardware, and now twice by the measuring
apparatus itself. The common thread is that a number is not a measurement
until you know what else was running.

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

## A note on the triton warning

Every run prints:

```
W torch/utils/flop_counter.py:29] triton not found; flop counting will not
work for triton kernels
```

It is worth saying what it is not. It is not a compute path failing, and it
is not this project asking for something it cannot have. It is a logging
warning raised at import time of `torch.utils.flop_counter`, on any CUDA
build without triton — and triton has no official Windows wheel, so on
Windows it appears unavoidably.

What it guards is `FlopCounterMode`, a diagnostic that intercepts operators
and sums shape-based FLOP formulas. Nothing here uses it, and PyTorch's own
documentation for it notes that "the workload still executes normally" and
that the count is theoretical rather than a hardware measurement.

`la/__init__.py` sets the level on that single logger, which removes the
noise and leaves every other warning intact. Installing `triton-windows`
would silence it too, and would do nothing else.

## Licence and provenance

Apache-2.0. Everything here is written for this project.

The base model is `Qwen3-1.7B-Base`, under its own licence. The benchmark is
LinAlg-Bench (NeurIPS 2026 Datasets & Benchmarks), CC-BY-4.0, downloaded at
run time and not redistributed here. The corpus is generated locally and is
not redistributed here either.
