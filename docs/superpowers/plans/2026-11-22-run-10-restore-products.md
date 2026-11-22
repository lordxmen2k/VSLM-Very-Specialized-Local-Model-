# Run 10 — restore the multiplications, and nothing else

**Status:** implemented and verified without a GPU. **Not yet scored.**

## Why run 10 exists

Run 9 scored **225/660 = 34.1%** — worse than run 8 (250/660), worse
than the 259-run (259/660), and worse than run 7 (282/660).

The run used the **right code**. The pull fast-forwarded `0f64ed5..36d1bbc`,
the contract check printed the size-mix table, and corpus generation ran
at **1,268/s** against run 8's 373/s — the `_exactify` fast path, 3.4×
faster. All three fingerprints confirmed it.

And the training loss predicted the score, as it has every time:

| step | run 7 | run 8 | the 259-run | **run 9** |
|---|---|---|---|---|
| 200 | 0.0542 | 0.0692 | 0.0687 | **0.0761** |
| 400 | 0.0536 | 0.0784 | 0.0790 | 0.0812 |
| 600 | 0.0425 | 0.0553 | 0.0543 | 0.0578 |

**Run 9 has the highest loss of any run and the worst score.**

## The cause: `matrix_power` lost 26 problems

| family | 259-run | **run 9** | base | Δ |
|---|---|---|---|---|
| **matrix_power** | 41 | **15** | 34 | **−26** |
| matrix_vector | 18 | 8 | 44 | −10 |
| multiplication | 33 | 29 | 25 | −4 |
| determinant | 4 | 3 | 54 | −1 |
| eigenvalue | 4 | 2 | 2 | −2 |
| nullity | 14 | 20 | 19 | +6 |
| rank | 26 | 29 | 21 | +3 |
| trace | 59 | 59 | 58 | 0 |
| transpose | 60 | 60 | 49 | 0 |

`matrix_power` lost more problems than the whole 34-problem regression.
The 4×4 column fell from 95 to 70.

## The mechanism

Run 9 shortened the dot product to fit the 1,536-token sequence cap, at
4×4 and 5×5:

```
full:     4*4 + (-4)*(-4) + 3*3 + 0*0 = 16 + 16 + 9 + 0 = 41
compact:                        16 + 16 + 9 + 0 = 41
```

The compact form drops the **products**. The model is shown the partial
sums and never the multiplication that produced each term, so it is
taught to emit a number it cannot compute.

**The arithmetic checker could not see it.** It validates the expression
on the *left* of each `=`, and a compact line is a perfectly well formed,
perfectly well valued expression. 3,800 problems, every expression
correct, and the corpus quietly teaching less.

## The measurement that would have prevented it

The full dot product, one step per entry, measured at every size:

| | chars | tokens | cap |
|---|---|---|---|
| matrix_power 3×3 | 1,206 | 591 | ok |
| matrix_power 4×4 | 2,005 | **983** | ok |
| matrix_power 5×5 | 3,267 | 1,601 | over |
| matrix_multiply 4×4 | 2,008 | **984** | ok |
| matrix_multiply 5×5 | 3,237 | 1,587 | over |

**The 4×4 form fits the cap in full.** Run 9 measured the size that
failed (5×5), applied the fix to 4×4 as well, and lost 26 problems.

## What run 10 changes

Exactly one thing: **the full dot product is restored at 3×3 and 4×4**,
and the reduced form is kept at 5×5 only — the one size that genuinely
does not fit.

Nothing else moves. The size mix stays uniform. The magnitudes stay
where run 9 put them. The `CRootOf` rejection stays. The `_exactify` fast
path stays.

The arithmetic-load band in `test_corpus_stability.py` moves from 240 to
280, because restoring the products puts `matrix_power` at 248
operations. The band is a change detector, not a difficulty target, and
the failure message now says to check whether the extra length is work
the model needs before raising it.

## The tests

- `tests/test_derivation_shows_products.py` — the derivation must show
  its product terms at 3×3 and 4×4, and the 5×5 example must still fit
  the cap. It also demonstrates, as a test, that the arithmetic checker
  accepts a compact line: that is *why* this needs a separate check.
- `tests/test_corpus_stability.py` — the band, with the new ceiling and
  the reason recorded.

## Verification

```
213 tests                                OK
make contract-check                      CONTRACT OK
make check-arithmetic  (3,800 problems)  every printed expression evaluates
```

## The lesson

**A checker that asks "is what is present correct?" cannot see what has
stopped being present.** Run 9's arithmetic checker proved every printed
expression evaluated to the value it claimed, while the corpus was
quietly teaching less.

And when a change is made to fit a limit, **measure the limit at every
size the change touches**, not at the size that failed.

## What is still a projection

The run-10 score. The determinant is still the family's deficit — 4/150
against the base's 54/150 — and the 259-run showed the fine-tune is one
problem ahead of the base on every other family (255/510 = 50.0% against
252/510 = 49.4%). Run 10 does not touch the determinant; it undoes the
regression that made it worse.
