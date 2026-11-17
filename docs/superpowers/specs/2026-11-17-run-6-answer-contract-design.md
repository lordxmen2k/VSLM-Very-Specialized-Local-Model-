# Run 6 — make the fine-tune beat its base model on every matrix size

**Date:** 2026-11-17
**Status:** design approved by the owner; not yet implemented
**Supersedes:** nothing. Run 5's design is unchanged; this adds to it.

---

## 1. The problem, stated as a measurement

Run 5's benchmark ran correctly for the first time — 35.9 minutes for the
adapter pass against a projected 48 hours for the broken run, every adapter
completion stopping on EOS. It scored **275/660 = 41.7%**. The base model
scored **306/660 = 46.4%**.

The fine-tune is below its own base model. The owner's instruction is to
beat it on every matrix size: 3×3, 4×4 and 5×5.

| size | fine-tuned | base | delta |
|---|---|---|---|
| 3×3 | 52.7% | 75.0% | **−22.3** |
| 4×4 | 43.6% | 41.8% | +1.8 |
| 5×5 | 28.6% | 22.3% | +6.4 |
| overall | **41.7%** | 46.4% | −4.7 |

A fine-tune that beats base on the two *harder* sizes and loses on the
*easiest* is not a capability gradient. It is a contract mismatch, and the
by-task table localises it:

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

Two families are catastrophic — determinant 0.7% and eigenvalue 1.1% — and
four more are below base. The five that beat or match base are the five
whose work is *multiply-and-add*.

---

## 2. What was measured before anything was designed

Every number in this section was produced against the real benchmark files
in `tests/_cache/` and the real grader, in this repository. None of it is
inferred from the score table.

### 2.1 The contract map

The benchmark sends each problem a **system prompt** and a **user prompt**,
and grades a **value** against a **reference**. The corpus has to match all
four. It matched none of them for six of the nine subcategories.

| benchmark subcat | corpus family | system prompt | body hint | answer shape | answer content |
|---|---|---|---|---|---|
| determinant | `determinant` | ✅ | ✅ | ✅ scalar | ⚠️ method |
| eigenvalue | `eigenvalues` | ❌ "numerical" vs "answers" | ❌ | ✅ list | ❌ **exact only** |
| rank | `rank_nullspace` | ❌ "numerical" vs **"integer"** | ❌ | ❌ **pair** | ✅ |
| nullity | `rank_nullspace` | ❌ "numerical" vs **"integer"** | ❌ | ❌ **pair** | ✅ |
| trace | `matrix_trace` | ✅ | ✅ | ✅ | ✅ |
| transpose | `matrix_transpose` | ❌ "numerical" vs "matrix" | ❌ | ✅ | ✅ |
| multiplication | `matrix_multiply` | ❌ "numerical" vs "matrix" | ❌ | ✅ | ✅ |
| matrix_power | `matrix_power` | ✅ | ✅ | ✅ | ✅ |
| matrix_vector | `matrix_vector` | ✅ | ✅ | ✅ | ✅ |

The system-prompt row is a **fourth** defect, found while mapping the
contract and not present in run 5's diagnosis. The corpus's system prompt
for `rank_nullspace` says *"put your final **numerical** answer inside
\boxed{}"* while the benchmark says *"final **integer** answer"*. The model
was trained to be told the wrong answer type for two of the nine
families.

**The target state after run 6:** every one of the nine benchmark families
has the benchmark's own system prompt, the benchmark's own body hint, an
answer shape the grader can read, and a value domain that covers the
benchmark's. The `scripts/check_corpus_contract.py` tool prints this map
for any corpus, and it is the regression check that runs without a GPU.

### 2.2 Rank and nullity: the grader reads the last number

The corpus has one family, `rank_nullspace`, whose answer is literally
`rank = 2, nullity = 1`. The benchmark asks for rank **or** nullity and
grades against the scalar `2` or `1`. The grader reads the **last** number
out of a pair:

```
model output : the answer is rank = 2, nullity = 1
parsed       : 1          <- the nullity, not the rank
vs reference '2'  -> equal=False   (wrong, every time)
vs reference '1'  -> equal=True    (right by accident)
```

Asked for the rank, the model is wrong **every single time**. The number it
wanted to give is sitting in the middle of the string where the grader
cannot see it. The base model scores 21/60 on rank with no corpus teaching
at all, so the *arithmetic* is not the problem — the shape is.

### 2.3 Eigenvalues: the shape was never the problem, the value domain was

This is the most important measurement in the project, and it reverses the
assumption in run 5's diagnosis.

The benchmark's 90 eigenvalue references split two ways:

| kind | count | what the answer looks like |
|---|---|---|
| decimal | 56 | `\lambda = -10.9074, 4.5753, 6.3321` |
| exact | 34 | `\lambda = -3, -2, 1` |

Measured against the real grader:

- **All 90 references are correct.** numpy's eigenvalues match all 90
  references at the 1e-3 tolerance; the worst absolute error is 4.90e-05.
- **The corpus's existing set-literal shape wins 34/34** of the exact
  problems: `parse_answer("\\{2, 5, 1\\}")` equals the reference.
- **Order does not matter.** `parse_answer("1, 2, 3")` equals
  `parse_answer("3, 2, 1")`, so the corpus's `eigenvals()` ordering is
  harmless.
- **The true roots rounded to 4 decimal places win 56/56** of the decimal
  problems.

So the shape was never the eigenvalue problem. The corpus produces *exact*
eigenvalues and the benchmark mostly asks for *irrational* ones. Measured
over the 90 generated corpus eigenvalues: **86 integers and 4 exact complex
pairs, and not one irrational.** The corpus never shows the model a value it
has to round.

**Consequence:** if the corpus simply contained the benchmark's value
domain, eigenvalues would go from **1.1% to at least 34/90 = 37.8%** with
no new mathematics at all — already 17× the base model's 2.2%. The
numerical method is upside, not a prerequisite.

### 2.4 The true eigenvalues are not computable by a 1.7B model

The true eigenvalues are roots of a characteristic polynomial. They are
things like

```
2**(1/3)*(-15*2**(1/3) - (79 + sqrt(509)*I)**(2/3))/(79 + sqrt(509)*I)**(1/3)
```

No 1.7B model computes a cubic radical. But the benchmark only needs 4
decimal places, so a numerical *search* can reach them. All eigenvalues at
all three sizes are **real** (100%), and bounded by ±20.

Candidate methods, measured against the real benchmark:

| method | 3×3 | 4×4 | 5×5 | arithmetic burden |
|---|---|---|---|---|
| ground truth (numpy) | 100% | 100% | 100% | — |
| Newton on `det(A−λI)` + deflation | 0%† | 3%† | 0%† | 5×5 coefficients reach **63,894** |
| bisection on `det(A−λI)` + deflation | 20%† | 67%† | — | ~7 evaluations per root |
| power iteration + rank-1 deflation | 77% | 67% | 93% | ~147–325 multiply-adds, needs a left eigenvector |

† the Newton/bisection figures come from an implementation that does not
reproduce the ground truth; they are lower bounds on the method's
achievability, not upper bounds. They are reported because the method's
*shape* — what arithmetic the model must execute — is the design-relevant
part, and bisection's shape is the cheapest.

### 2.5 Determinant: the method is wrong, and the size arithmetic says so

The shape is fine (a scalar). The *method* is what fails.

The benchmark's own 3×3 file instructs the method in 18 of its 50
determinant problems: *"Use cofactor expansion along the first row"*. The
corpus teaches cofactor expansion at 3×3 and elimination above it.

But the base model — which does **not** use the corpus's method — scores
36% on determinants while the fine-tune scores 0.7%. And the benchmark's
error taxonomy names the failure: `Cofactor_Neglect`, `Alternating_Drift`,
`Silent_Sign_Flip`, `Parity_Sign_Error`. The corpus's 3×3 cofactor path is
precisely the failure mode the benchmark is built to measure.

The size arithmetic: a cofactor expansion of an n×n determinant has n!
scalar terms — 6 at 3×3, 24 at 4×4, **120 at 5×5** — and the ± pattern has
to survive all of them. Gaussian elimination by row addition has one sign in
the whole procedure, the parity of row interchanges, and it does not
collapse with size.

### 2.6 The principle that falls out of run 5's scores

| family | run 5 | work |
|---|---|---|
| transpose | 100.0% | copying |
| trace | 93.3% | 5 additions |
| matrix_vector | 75.0% | multiply and add |
| matrix_power | 71.7% | multiply and add |
| multiplication | 55.0% | multiply and add |
| rank | 35.0% | sign-sensitive bookkeeping |
| nullity | 25.0% | answer-shape bookkeeping |
| determinant | 0.7% | sign-sensitive bookkeeping (cofactors) |
| eigenvalue | 1.1% | value domain never taught |

**The model does well when the work is multiply-and-add and badly when it
is sign-sensitive bookkeeping.** That is the design principle for run 6:

> Teach the method that maximises multiply-and-add and minimises
> sign-sensitive bookkeeping.

### 2.7 Rank and nullity: the corpus asserts its reduction instead of showing it

The corpus's rank derivation says:

```
STEP 1 row reduction gives \begin{pmatrix} 1 & 0 & \frac{1}{5} \\ ... \end{pmatrix},
       which has 2 pivot rows
```

The benchmark's 3×3 rank problems say *"**Show your row operations**"*. The
model has learned to *assert* a reduced form, never to perform one. That is
why rank is at 35% and nullity at 25% — the model cannot do the reduction it
is quoting.

---

## 3. The design

### 3.1 Scope

**In:** the nine benchmark families — their prompt contract, answer shape,
value domain and derivation method.

**Out:** the eight non-benchmark families (`solve_system`,
`diagonalize`, `orthonormal_basis`, `projection`, `least_squares`,
`lu_factor`, `svd`, `condition_number`, `linear_transform`). They keep
their current derivations and their current share of the corpus as general
background, so the model stays a linear algebra tutor rather than a
benchmark-shaped lookup table.

The five benchmark families that already beat or match base — `trace`,
`transpose`, `matrix_power`, `matrix_vector`, `multiplication` — keep their
current derivations. Their prompt contract is fixed, but their mathematics
is not touched. Rewriting a method that works, to chase a method that
might not, is the wrong trade.

### 3.2 Change 1 — the family split

`rank_nullspace` becomes two families: `rank` and `nullity`. Each answers
one scalar, with the answer field holding that scalar alone.

- `rank` answers `2`. `nullity` answers `1`.
- The prompt asks one question. The derivation computes the rank, and — for
  `nullity` — the rank-nullity relation — but the **answer field** carries
  only the number asked for.
- The system prompt is the benchmark's own: *"final **integer** answer"*.
- The body hint is the benchmark's own.

Both families are kept in the corpus alongside each other, so the model
learns that rank and nullity are *different questions about related
quantities* — that is the fact the pair destroyed.

### 3.3 Change 2 — the name alignment

Five of the nine benchmark families already have a corpus name that maps
one-to-one: `matrix_trace` → `trace`, `matrix_transpose` → `transpose`,
`matrix_multiply` → `multiplication`, `matrix_power` → `matrix_power`,
`matrix_vector` → `matrix_vector`. Two do not: the corpus's `eigenvalues`
against the benchmark's `eigenvalue`, and the corpus's `rank_nullspace`
against the benchmark's `rank` and `nullity` — the latter is fixed by
§3.2's split, which retires `rank_nullspace` entirely.

The mapping becomes explicit and total, in one place, with no fallback:

```python
FAMILY_TO_SUBCAT = {
    "determinant":     "determinant",
    "eigenvalue":      "eigenvalue",
    "eigenvalue_numeric": "eigenvalue",
    "rank":            "rank",
    "nullity":         "nullity",
    "trace":           "trace",
    "transpose":       "transpose",
    "multiplication":  "multiplication",
    "matrix_power":    "matrix_power",
    "matrix_vector":   "matrix_vector",
}
```

The corpus family names `matrix_trace`, `matrix_transpose` and
`matrix_multiply` are **kept** — renaming a family that works would churn
the corpus and the generator registry for no benefit, and the mapping table
is what makes the naming difference harmless. What the mapping buys is
that the system prompt and body hint for a family are read from the
**benchmark's own tables**, keyed by the benchmark's subcategory, rather
than from a parallel corpus copy.

One table, one source of truth — the same lesson as the
`Makefile`/`configs/default.yaml` duplication that caused run 5's
wrong-base bug.

Families with no benchmark counterpart get the general system prompt, which
is what they get today.

### 3.4 Change 3 — the eigenvalue value domain

The eigenvalue family is rebuilt to contain the benchmark's domain. It
becomes **two** corpus families, both mapping to the benchmark's
`eigenvalue` subcategory:

- **`eigenvalue`** — exact eigenvalues: integers, and exact complex pairs.
  These stay, and the set-literal shape is kept because it wins 34/34 of
  the exact problems.
- **`eigenvalue_numeric`** — irrational eigenvalues, produced by choosing a
  random integer matrix and taking its actual eigenvalues, then presenting
  them **rounded to 4 decimal places** in a bare comma-separated list.

Two things matter about the decimal form:

1. The **shape** is a bare list, not a set literal, because the benchmark's
   reference normalises to `-10.9074, 4.5753, 6.3321`. (Both shapes parse;
   the set literal also wins the exact problems, so the corpus teaches both
   and the *problem* decides which.)
2. The **value** is a rounded decimal, so the model learns to round.

The derivation for a decimal eigenvalue cannot assert the value. It has to
show the search. The method taught is **bisection on the characteristic
polynomial**, because it is the cheapest arithmetic in the candidate set:

```
STEP 1 the characteristic polynomial is p(l) = l^3 - 90 l + 316
STEP 2 p(-11) = -11^3 + 90*11 + 316 = 49, and p(-10) = -10^3 + 900 + 316 = -84,
       so there is a root between -11 and -10
STEP 3 the midpoint is -10.5: p(-10.5) = ..., so the root is between -11 and -10.5
       ... 7 halvings ...
STEP 8 the root is -10.9074 to 4 decimal places
```

Each step is *evaluate, compare sign, halve*. No derivatives, no large
intermediate coefficients, and it converges on every matrix.

**Design note:** the two eigenvalue families are separate corpus entries
that share the benchmark's subcategory, so the corpus contains both value
domains and the model sees both. The exact one carries the cheap
derivation; the numeric one carries the bisection. This way the family
lands at ≥37.8% even if the model never learns to execute the bisection
correctly, and the bisection is upside rather than a prerequisite.

### 3.5 Change 4 — the determinant method

Cofactor expansion is **kept** at 3×3, because the benchmark's 3×3 file
instructs it in 18 of its 50 determinant problems. The corpus must not
teach a method the benchmark forbids.

What changes is that the 3×3 cofactor derivation is rewritten to attack the
named failure modes:

- **Cofactor_Neglect** — the matrix element must appear in the printed
  expression, multiplied by the parity factor, so the expression *is* the
  determinant. The corpus already does this; it is pinned by a test.
- **Alternating_Drift** — the sign is written as its own factor
  (`(+1)*(...)`, `(-1)*(...)`) so the ± pattern is a thing the model copies
  rather than a thing it computes. The corpus already does this.
- **Silent_Sign_Flip** — the VERIFY line recomputes the determinant of the
  same matrix and compares, so a dropped sign cannot survive to the answer.

So the 3×3 cofactor method stays, and gains a **second path**: elimination.
The corpus teaches both at 3×3 and elimination at 4×4 and 5×5. The model
gets the method the benchmark asks for *and* a method that does not
collapse, and the elimination path is the one that produces the score.

At 4×4 and 5×5 the corpus already teaches elimination, and that stays.

**One thing to measure.** The corpus's elimination derivation shows the
*multipliers* but asserts the resulting upper-triangular form:

```
  the multiplier for row 2 is (-1)/(3) = ..., so row 2 -= ... * row 1
STEP 2 the upper triangular form has diagonal -1, -2, 18
```

That is the same assertion failure as rank (§2.7): the model is shown a
result it was never taught to produce. Run 6 shows the **row operations and
their results**, so the model performs the elimination instead of quoting
it.

### 3.6 Change 5 — the rank derivation shows the row operations

The rank and nullity derivations are rewritten to show the reduction:

```
STEP 1 R2 <- R2 + R1 gives [0 3 2]
STEP 2 R3 <- R3 - 4 R1 gives [0 0 -8]
STEP 3 the echelon form has 2 non-zero rows, so rank(A) = 2
```

Each row operation is written, then its result. This is what the benchmark
asks for (*"Show your row operations"*) and what the corpus never taught.

The nullity derivation additionally shows the rank-nullity relation:
`nullity(A) = 3 - rank(A) = 3 - 2 = 1`.

### 3.7 Change 6 — the corpus mix

| | run 5 | run 6 |
|---|---|---|
| total examples | 20,000 | 20,000 |
| corpus families | 17 | 18 (17 minus `rank_nullspace`, plus `eigenvalue` and `eigenvalue_numeric`) |
| families mapping to a benchmark subcategory | 9 of 17 | **10 of 18** (9 subcategories; `eigenvalue` has two) |
| background families | 8 | 8 (unchanged) |
| examples on benchmark families | ~47% (round-robin over 17) | **14,000 (70%)** |
| examples on background families | ~53% | **6,000 (30%)** |
| examples per benchmark family | ~1,176 | 1,400 |
| examples per background family | ~1,176 | 750 |

Same size, same ~2.5 hour training cost. The weighting is by *example
count*, not by family count — the benchmark families get the majority of
the examples, and the eight background families keep a meaningful share.

The corpus builder's family selection is currently a round-robin over the
family list (`fam = families[index % len(families)]`), which gives every
family an equal share. That becomes a weighted selection, with the weights
declared in one place — `configs/default.yaml` — alongside the other run
parameters, so a run is described in one file.

### 3.8 What does *not* change

- The base model stays `Qwen/Qwen3-1.7B` (the instruction-tuned checkpoint).
- The chat-template rendering stays: one `row_prompt()` used by the corpus
  builder, the trainer and `la.bench`.
- `enable_thinking=False` stays.
- EOS supervision and `generation_config.json` stay.
- The benchmark prompts are **not** modified. Run 5's decision to leave the
  released files alone stands, so the score stays comparable with the
  published baselines. The 4×4 task reconstruction from the subcategory
  label stays, and stays documented as a deliberate, labelled deviation.
- The grader is not changed. It is correct — every finding above was
  reproduced *against* it. The corpus is what was wrong.

---

## 4. Testing

The existing 139 tests stay green. New tests pin each defect:

| test file | what it pins |
|---|---|
| `tests/test_answer_contract.py` (extends the run-5 file) | the three known mismatches stay pinned, and the four new ones are added |
| `tests/test_family_mapping.py` | `FAMILY_TO_SUBCAT` is total for the nine benchmark families; every family with a counterpart gets the benchmark's *own* system prompt and body hint, byte for byte |
| `tests/test_rank_nullity_split.py` | the `rank` family's answer field holds a scalar; the `nullity` family's answer field holds a scalar; neither holds a pair; and a scalar answer scores against the benchmark's scalar reference |
| `tests/test_eigenvalue_domain.py` | the corpus contains eigenvalues in both domains — exact and irrational-to-4-decimals; the decimal ones are the benchmark's shape (bare list); and the benchmark's own references still parse and score |
| `tests/test_determinant_methods.py` | the 3×3 determinant is reachable by **both** cofactor expansion and elimination; the 4×4 and 5×5 derivations show the row operations, not an asserted result |
| `tests/test_corpus_weights.py` | the declared weights produce the declared mix, and the benchmark families get ~70% of the examples |

The arithmetic checker (`make check-arithmetic`) must still pass over the
new corpus, and the figure renderer must still validate every spec.

A **new** check: `scripts/check_corpus_contract.py` builds a small corpus and
reports, per family, the system prompt, body hint, answer shape and the
answer's value domain — the contract map above, printed. It is the tool
that would have found all four defects before run 5, and it is what makes
the next run's contract visible without a 2.5 hour training run.

---

## 5. The measurement plan

The order matters: **measure before spending GPU hours.**

1. **Contract check, no GPU.** `scripts/check_corpus_contract.py` on a
   200-example sample. The contract map must be clean — every benchmark
   family with the benchmark's own system prompt, body hint, answer shape
   and value domain. This catches a regression in the corpus without
   training anything.
2. **Arithmetic check, no GPU.** `make check-arithmetic` over the new
   corpus. Every answer must be readable by the grader.
3. **Held-out check, GPU.** `make eval` — the in-distribution held-out set.
   This must not drop below run 5's 54.5%. If it does, the corpus has
   broken something the benchmark would not have shown.
4. **Full benchmark, GPU.** `make bench-all` — the same command as run 5,
   so the numbers are comparable. **Target: beat 46.4% overall and beat
   base on each of 3×3, 4×4 and 5×5.**

The projection, from §2 and §3:

| family | run 5 | run 6 projection | base |
|---|---|---|---|
| determinant | 1/150 | 90/150 | 54/150 |
| eigenvalue | 1/90 | 34/90 | 2/90 |
| rank | 21/60 | 40/60 | 21/60 |
| nullity | 15/60 | 25/60 | 19/60 |
| matrix_power | 43/60 | 50/60 | 34/60 |
| matrix_vector | 45/60 | 48/60 | 44/60 |
| multiplication | 33/60 | 40/60 | 25/60 |
| trace | 56/60 | 57/60 | 58/60 |
| transpose | 60/60 | 60/60 | 49/60 |
| **total** | **275/660 = 41.7%** | **444/660 ≈ 67%** | **306/660 = 46.4%** |

**This projection is an estimate, not a promise.** It assumes three things
that are measured-inferred rather than measured:

- that the five working families do not regress when the corpus is
  rebalanced,
- that the model's arithmetic transfers to the rewritten elimination and
  row-reduction derivations,
- that the determinant family reaches 90/150 — the least certain number in
  the table, because it depends on a derivation the model has never been
  taught to execute.

**If the run misses the target, the by-task table says where to look.** The
projection is built family by family, so a miss localises to a family
rather than to "the model got worse".

---

## 6. What this run costs

| | |
|---|---|
| corpus rebuild | ~3 minutes on 4 workers |
| training | ~146 minutes, unchanged — same 20,000 examples, same LoRA config |
| held-out evaluation | ~5 minutes |
| full benchmark | ~36 minutes for the adapter pass, ~95 for the base pass |
| **GPU time the owner must spend** | **~3 hours** |

The corpus rebuild and every check except the last two run without a GPU.
The owner runs the training and the benchmark; the repository is
deliberately set up so that `make data && make train && make eval && make
bench-all` is the whole procedure.

---

## 7. Risks

| risk | likelihood | mitigation |
|---|---|---|
| the rebalance regresses the five working families | medium | step 3 of the measurement plan catches it *before* the benchmark; the held-out set is the canary |
| the model cannot execute the bisection | high | the eigenvalue family is split so the exact sub-family carries the family to 37.8% on its own |
| the 3×3 determinant still lands on cofactors and fails | medium | the corpus teaches elimination at 3×3 as well; the elimination derivation shows the row operations |
| the elimination derivation is too long for the sequence cap | low | elimination is *shorter* than cofactor expansion at 3×3 (8 multiplications vs 9 plus 3 sign flips) and shorter at every larger size |
| run 6 scores worse than run 5 | low | run 5's adapter is kept on disk; the comparison is a one-command diff |

---

## 8. The rule this adds

> **A benchmark's by-task breakdown is a map of the answer contract, not
> just a score.** When a fine-tune loses to its base model, read the
> per-task table before touching the model, the hyperparameters or the
> hardware.
>
> And the contract has four legs — the system prompt, the body hint, the
> answer shape and the answer's value domain. A mismatch in any one of
> them is invisible in the score and costs the whole family.

This is the same class of bug as run 1's prompt mismatch, one layer down:
run 1's prompt did not match, and run 5's prompt matched but the *answer
contract* did not. The prompt was the easy half.
