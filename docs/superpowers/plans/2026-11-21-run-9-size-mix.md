# Run 9 — the size mix, and the arithmetic the benchmark asks for

**Status:** implemented and verified without a GPU. **Not yet scored** — the
benchmark needs the user's GPU.

## Why run 9 exists

Run 8 scored **250/660 = 37.9%**, a regression from run 7's 282/660 = 42.7%
and still 56 problems behind the base model's 306/660 = 46.4%.

The diagnosis took three steps, and the first one invalidated everything
before it.

### Step 1: the corpus comparison was invalid

The `la` package is installed in the virtualenv as an **editable install
pointing at the main repository**. Running `python -m la.build_dataset`
from a git worktree at `2bfe297` therefore used run 8's code, not the
worktree's. The comparison reported "94.7% of the rows are identical" and
concluded that run 8's change was nearly a no-op.

Rebuilding with `PYTHONPATH=<worktree>/src` gives the truth: **only 24.2%
of rows were identical at the same index**, and every family's own sequence
had changed. Run 8's corpus was a completely different sample.

### Step 2: the harness is deterministic

The base model scored **306/660 with byte-identical per-family numbers** in
runs 6, 7 and 8. Zero measurement noise. Every difference between the runs
is a property of the adapter, so the comparisons are sound.

### Step 3: run 8 changed three things at once

| | loss @ step 200 | score |
|---|---|---|
| run 7 | 0.0542 | 42.7% |
| run 8 | 0.0692 | 37.9% |

The loss is 1.28× higher and **the loss and the score move together across
all four runs**. The corpus got harder to fit, and the score followed it
down.

| family | run 7 | run 8 | Δ score |
|---|---|---|---|
| rank | 28 ops (3×3 only) | 32 (3×3), 55 (4×4), 63 (5×5) | 29 → 27 |
| nullity | 33 ops (3×3 only) | 36, 47, 67 | 19 → 17 |
| trace | unchanged | unchanged | 56 → **59** |

The families whose corpus got harder lost points. The family whose corpus
did not change gained them.

## The contract leg that was never checked: the SIZE MIX

Measuring every matrix in the released benchmark files against the corpus's
generated data turned this up.

**The benchmark's size mix is exactly uniform for every family, and there
is not one 2×2 matrix anywhere in all 660 problems.**

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

Between 10 and 40 per cent of the corpus taught 2×2 problems the benchmark
never scores. Those are rows **not** spent on 5×5 — and a 5×5 derivation is
a different kind of task: more row operations, more arithmetic, more chances
to slip. The corpus under-taught exactly the size where the benchmark's
arithmetic is hardest, and the determinant family — where run 7 scored 4
against the base's 54 — is the family whose 5×5 cofactor products reach
9,112.

## What run 9 changes

1. **The size mix is the benchmark's.** A single `_bench_size()` helper
   draws 3, 4 or 5; every benchmark family uses it. No 2×2 anywhere.

2. **The eigenvalue corpus stops overshooting.** Run 8 scaled the
   large-entry matrix by a blind factor `k ∈ {6, 8, 12, 20}` and
   `|S D S⁻¹|` is unbounded, so the corpus reached 11,080 against the
   benchmark's 1,120 — a 23× overshoot in the p90. Run 9 scales so the
   largest entry **lands on** a target drawn from the benchmark's own
   per-size shape, and scales **down** when the base is already too large,
   which run 8's `max(1, target // peak)` silently refused to do.

3. **`CRootOf` answers are gone.** 2 per cent of the corpus's eigenvalue
   answers were `CRootOf(x**5 + ...)`, which the answer grammar cannot read
   and the grader cannot score. `_exact_eigenvalues` used to *fall back* to
   sympy's `eigenvals()` when the rational root theorem could not find every
   root. A matrix with an irreducible characteristic polynomial is now
   **rejected**, not answered.

4. **The decimal eigenvalue family is built symmetric.** A random integer
   matrix has a real spectrum only 2 per cent of the time at 5×5 — 98 per
   cent have a complex conjugate pair — while all 90 of the benchmark's
   eigenvalue answers are real decimals. A symmetric matrix has all-real
   eigenvalues, always.

5. **The corpus fits the sequence cap.** The 5×5 products and powers ran to
   1,818 tokens against the 1,536 cap, and a truncated example loses the end
   of its completion, which is where the plot spec is. The fix keeps the
   **partial sums** — the run with the sums removed scored 27 per cent on
   multiplication and 0 on powers — and drops the product list and the
   repeated result matrix at the large sizes only.

6. **Each generator owns its random stream**, pinned by
   `_rng_for(seed, index)`. Run 8's `draw_entry` consumed two draws per entry
   where the old code consumed one, and that rewrote 76% of the corpus,
   including every matrix in `matrix_vector` — the family that lost 20
   points.

7. **Generation is 20× faster.** `sp.nsimplify()` ran mpmath's PSLQ on
   expressions that were already exact: 13.8 seconds of a 21.7 second run.
   The exactification now skips the search for an already-exact value. The
   eigenvalue generator went from 1,096 ms to 54 ms per problem.

## The tests

- `tests/test_size_mix.py` — the corpus's size mix against the released
  benchmark files, in both directions: every size the benchmark scores is
  covered, and no size it never scores is taught.
- `tests/test_corpus_stability.py` — the corpus must not overshoot the
  benchmark's magnitudes; the arithmetic load must stay in a band; each
  generator's random stream must be independent of its neighbours'.
- `scripts/check_corpus_contract.py` now prints and checks the size mix
  alongside the prompt, tail and answer-shape contract.

**All three fail on run 8's code**, with the measured numbers in the
message.

## Verification

```
209 tests                                OK
make contract-check                      CONTRACT OK
make check-arithmetic  (3,800 problems)  every printed expression evaluates
```

## What is still a projection

The run-9 score. It needs the user's GPU, and it must be compared against
the base model's 306/660 with the same per-family table.

The rules this run adds:

1. **The base model is the control.** Score it in every run.
2. **The loss curve is the early warning.** It moves with the score and
   arrives three hours earlier.
3. **Matching the benchmark's magnitudes is necessary, not sufficient.**
4. **One change per run**, and check the corpus against the released data,
   not against a hand-written table of it.
5. **Pin the import path** before believing any corpus comparison — an
   editable install makes a worktree comparison meaningless.
