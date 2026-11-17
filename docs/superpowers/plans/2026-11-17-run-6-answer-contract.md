# Run 6 Answer Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the VSLM corpus so the fine-tune matches LinAlg-Bench's answer contract on all nine benchmark families and can be regenerated, trained, evaluated, and benchmarked end to end.

**Architecture:** Keep the benchmark's prompt tables in `la.benchprompt` as the single source of truth and map every corpus family to its benchmark subcategory. Split the combined rank/nullity family, split exact and decimal eigenvalues, make derivations show their row operations rather than assert results, and replace round-robin corpus selection with declared example weights.

**Tech Stack:** Python 3.10+, SymPy, NumPy, PyYAML, unittest, Matplotlib.

## Global Constraints

- Do not create or use GitHub Actions workflows.
- Keep the base model in `configs/default.yaml` as the only source of truth.
- Do not modify the released benchmark files or the grader.
- Keep `enable_thinking=False`, EOS supervision, and the shared `row_prompt()` rendering unchanged.
- Every implementation step follows red-green-refactor: write the failing test, observe the expected failure, implement the minimum, rerun.
- Run `make check` before every commit and before final delivery.

---

### Task 1: Family-to-subcategory contract mapping

**Files:**
- Modify: `src/la/benchprompt.py`
- Modify: `src/la/problems.py`
- Create: `tests/test_family_mapping.py`

**Interfaces:**
- Produces: `FAMILY_TO_SUBCAT: dict[str, str]`
- Produces: `benchprompt.system_prompt_for(family)` and `benchprompt.body_hint(family)` resolve corpus families through that map.

- [ ] **Step 1: Write the failing test**

Create `tests/test_family_mapping.py` asserting that every benchmark family maps to its benchmark subcategory, `Problem.system_prompt()` equals `SYSTEM_PROMPTS[subcat]`, and the question contains `BODY_HINTS[subcat]`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_family_mapping -v`
Expected: FAIL because `FAMILY_TO_SUBCAT` does not exist and corpus family names do not resolve to benchmark prompt tables.

- [ ] **Step 3: Implement the minimal mapping**

Add `FAMILY_TO_SUBCAT` to `src/la/benchprompt.py`; make `system_prompt_for()` and `body_hint()` resolve through it before falling back to the general prompt.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_family_mapping -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/la/benchprompt.py src/la/problems.py tests/test_family_mapping.py
git commit -m "Map corpus families onto the benchmark prompt contract"
```

### Task 2: Split rank and nullity into scalar families

**Files:**
- Modify: `src/la/problems.py`
- Modify: `src/la/augment.py`
- Create: `tests/test_rank_nullity_split.py`
- Modify: `tests/test_answer_contract.py`

**Interfaces:**
- Produces: generators `rank` and `nullity`; removes `rank_nullspace`.
- Produces: both families answer with a scalar only.

- [ ] **Step 1: Write the failing tests**

Assert both new generators exist, their answers parse as scalars, the question asks only one quantity, and each scalar scores against the corresponding benchmark reference.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_rank_nullity_split -v`
Expected: FAIL because the generators do not exist.

- [ ] **Step 3: Implement the split**

Replace `gen_rank_nullspace` with `gen_rank` and `gen_nullity`; use one scalar answer field; update `GENERATORS` and question templates.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_rank_nullity_split tests.test_answer_contract -v`
Expected: PASS after the old bug-pinning assertions are inverted to assert the fixed behavior.

- [ ] **Step 5: Commit**

```bash
git add src/la/problems.py src/la/augment.py tests/test_rank_nullity_split.py tests/test_answer_contract.py
git commit -m "Teach rank and nullity as separate scalar questions"
```

### Task 3: Rebuild the eigenvalue value domain

**Files:**
- Modify: `src/la/problems.py`
- Modify: `src/la/augment.py`
- Create: `tests/test_eigenvalue_domain.py`

**Interfaces:**
- Produces: generators `eigenvalue` (exact set literal) and `eigenvalue_numeric` (bare four-decimal list).
- Produces: `bisect_eigenvalue(matrix, index)` returns a reproducible approximate real eigenvalue.

- [ ] **Step 1: Write the failing tests**

Assert the exact family stays exact, the numeric family contains irrational values rounded to four decimals in a bare list, the numeric answer scores against the benchmark's decimal reference form, and generated matrices have real eigenvalues.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_eigenvalue_domain -v`
Expected: FAIL because the numeric family does not exist.

- [ ] **Step 3: Implement both families**

Rename the old generator to `eigenvalue`; add `eigenvalue_numeric`; implement bisection over the characteristic polynomial with enough halvings for four decimals; keep both mappings to the benchmark's `eigenvalue` subcategory.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_eigenvalue_domain -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/la/problems.py src/la/augment.py tests/test_eigenvalue_domain.py
git commit -m "Teach both exact and decimal eigenvalue domains"
```

### Task 4: Make derivations perform their work

**Files:**
- Modify: `src/la/problems.py`
- Create: `tests/test_determinant_methods.py`
- Extend: `tests/test_rank_nullity_split.py`

**Interfaces:**
- Produces: determinant derivation includes row-operation results and a 3x3 elimination path.
- Produces: rank/nullity derivations include explicit row operations.

- [ ] **Step 1: Write the failing tests**

Assert that generated 3x3 determinants include both cofactor and elimination derivations, elimination derivations contain the resulting matrix, and rank/nullity derivations contain `R1 <-`, `R2 <-`, or `R3 <-` operations.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_determinant_methods tests.test_rank_nullity_split -v`
Expected: FAIL because the derivations assert reduced forms instead of showing operations.

- [ ] **Step 3: Implement the derivations**

Extend `_eliminate` to retain row-operation results; choose the 3x3 determinant method deterministically from the RNG; implement an explicit row-reduction trace for rank and nullity.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_determinant_methods tests.test_rank_nullity_split -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/la/problems.py tests/test_determinant_methods.py tests/test_rank_nullity_split.py
git commit -m "Show the row operations in determinant and rank derivations"
```

### Task 5: Weight the corpus 70/30

**Files:**
- Modify: `configs/default.yaml`
- Modify: `src/la/build_dataset.py`
- Create: `tests/test_corpus_weights.py`

**Interfaces:**
- Produces: `data.family_weights` mapping every corpus family to a non-negative relative weight.
- Produces: `build(..., family_weights=...)` deterministic weighted selection.

- [ ] **Step 1: Write the failing tests**

Assert the config declares every family, benchmark families receive 70% of examples, background families receive 30%, and selection is reproducible from the seed.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_corpus_weights -v`
Expected: FAIL because weights are not declared or used.

- [ ] **Step 3: Implement weighted selection**

Read weights from `configs/default.yaml`; select families by cumulative weighted schedule; preserve seed reproducibility and parallel determinism.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_corpus_weights -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml src/la/build_dataset.py tests/test_corpus_weights.py
git commit -m "Weight the corpus 70 percent toward benchmark families"
```

### Task 6: Add the corpus contract checker and finish the runbook

**Files:**
- Create: `scripts/check_corpus_contract.py`
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `WORKING_STATE.md`

**Interfaces:**
- Produces: `python scripts/check_corpus_contract.py --n 200` prints the per-family contract map and exits nonzero on a mismatch.
- Produces: `make contract-check` target.

- [ ] **Step 1: Write the failing test**

Extend `tests/test_family_mapping.py` with a checker invocation against a small generated corpus and assert exit code zero and a report containing all benchmark subcategories.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_family_mapping -v`
Expected: FAIL because the checker does not exist.

- [ ] **Step 3: Implement the checker and Makefile target**

Build a small corpus, print system prompt, body hint, answer shape, and value domain per family, and fail on any benchmark-family mismatch.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_family_mapping -v`
Expected: PASS.

- [ ] **Step 5: Update the runbook**

Document `make contract-check`, `make data`, `make train`, `make eval`, and `make bench-all`, including expected runtime and the run-5 baseline.

- [ ] **Step 6: Run full verification**

Run: `make check`
Expected: all unit tests pass, arithmetic check exits zero, and the sample corpus builds and renders.

- [ ] **Step 7: Commit**

```bash
git add scripts/check_corpus_contract.py Makefile README.md WORKING_STATE.md tests/test_family_mapping.py
git commit -m "Add the corpus contract checker and run-6 rerun instructions"
```

## Final Verification

- [ ] `make check` exits 0.
- [ ] `make contract-check` exits 0.
- [ ] `git status --short` is clean.
- [ ] The README contains the exact end-to-end commands and no GitHub Actions workflow exists.
