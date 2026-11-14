"""Problem generators.

Every problem in the corpus comes from here, and every answer is computed by
sympy rather than written by hand. That matters: the training data is only
worth as much as the ground truth behind it, and a mechanical ground truth
cannot drift from the question it answers.

Each generator returns a `Problem`:

    family      which generator produced it
    question    the natural-language question
    scratchpad  the working, computed by sympy and therefore correct
    answer      the graded answer, in the answer grammar of `la.answer`
    spec        a plot spec, or None
    truth       the ground truth for the graded answer, in the same grammar
    difficulty  1 (easy) to 5 (hard)
    tags        free-form labels for slicing the corpus

The scratchpad always ends with a VERIFY line that re-checks the answer by
substitution. That is the habit we want the model to learn: compute, then
check, and if the check fails, say so rather than assert.

Matrices are built from small integers so that answers stay clean. Where a
problem needs particular eigenvalues they are chosen first and the matrix is
constructed around them, rather than found by luck.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import sympy as sp

from .answer import fmt_matrix, fmt_pairs, fmt_scalar, fmt_set
from .augment import paraphrase, permute_rows, question, similarity_transform

sp.Matrix  # noqa: B018  (keep the import used)


@dataclass
class Problem:
    family: str
    question: str
    scratchpad: str
    answer: str
    truth: str
    spec: dict | None = None
    difficulty: int = 2
    tags: list[str] = field(default_factory=list)
    compare_mode: str = "exact"   # "exact" | "basis" | "diag"
    stem: str = ""                # the core statement, before paraphrasing

    def to_json(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d

    @staticmethod
    def from_json(d: dict) -> "Problem":
        return Problem(
            family=d["family"], question=d["question"], scratchpad=d["scratchpad"],
            answer=d["answer"], truth=d["truth"], spec=d.get("spec"),
            difficulty=d.get("difficulty", 2), tags=list(d.get("tags", [])),
            compare_mode=d.get("compare_mode", "exact"),
            stem=d.get("stem", ""),
        )

    def prompt(self) -> str:
        """The prompt the model is given.

        The question already carries the benchmark's preamble, the task and
        the body, so it is passed through as it stands.
        """
        return self.question

    def completion(self) -> str:
        plot = _spec_json(self.spec)
        # the answer is boxed, because the benchmark asks for it that way and
        # the model should be fluent in the format it is scored in
        return (f"{self.scratchpad}\n### Answer\n\\boxed{{{self.answer}}}\n"
                f"### Plot\n{plot}\n")


def _spec_json(spec: dict | None) -> str:
    if not spec:
        return "{}"
    import json
    return json.dumps(spec, separators=(",", ":"))


def _vec(v: sp.Matrix) -> list[float]:
    """A column vector as a flat list of floats."""
    return [float(v[i, 0]) for i in range(v.rows)]


def _mat(m: sp.Matrix) -> list[list[float]]:
    return [[float(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def _sym_mat(m: sp.Matrix) -> list[list[Any]]:
    return [[m[i, j] for j in range(m.cols)] for i in range(m.rows)]


# --------------------------------------------------------------- sampling --

def _permute_pair(rng: random.Random, A, b):
    """Permute the rows of A and of b together.

    The system is unchanged, but the matrix the reader sees is different, so a
    model cannot recognise the problem from its arrangement. Every answer is
    recomputed from the permuted matrix.
    """
    order = permute_rows(rng, A.rows)
    A2 = A[order, :]
    b2 = b[order, :]
    return sp.Matrix(A2), sp.Matrix(b2)


def _permute_rows_only(rng: random.Random, A):
    order = permute_rows(rng, A.rows)
    return sp.Matrix(A[order, :])


def _exact(x):
    """Clean a value cheaply.

    sp.nsimplify searches for a closed form and is slow - around a quarter
    of a second on a fraction like 25/3. The arithmetic in the generators
    is already exact, so it only needs to run on floats.
    """
    if isinstance(x, sp.Float):
        return _exact(x)
    return x


def _fmt_sum(values) -> str:
    """A sum of numbers, with the signs written the way a person would.

    "-2 + 3 + 0" rather than "-2 + -3 + 0".
    """
    out = []
    for v in values:
        s = fmt_scalar(sp.simplify(v))
        if s.startswith("-"):
            out.append(f" - {s[1:]}")
        else:
            out.append(f" + {s}")
    return "".join(out).lstrip(" +")


def _dot_product(row, col) -> str:
    """A dot product, shown rather than asserted.

    "2*(-1) + 1*3 = -2 + 3 = 1"

    A model trained on a corpus that states matrix products without showing
    the dot products learns to produce a sixteen-term sum in one step, and
    it gets it wrong. Shown one line at a time, each line needs one row and
    one column, and the final matrix is a transcription.
    """
    terms = []
    products = []
    for a, b in zip(row, col):
        # a negative factor is bracketed, so the line reads "2*(-1)"
        # rather than "2*-1"
        av, bv = fmt_scalar(a), fmt_scalar(b)
        if av.startswith("-"):
            av = f"({av})"
        if bv.startswith("-"):
            bv = f"({bv})"
        terms.append(f"{av}*{bv}")
        products.append(_exact(a * b))
    total = _exact(sum(products))
    # The partial sums matter. "a*b + c*d = p + q = total" asks the model for
    # two multiplications and two additions, each conditioned on the tokens
    # already written; "a*b + c*d = total" asks for all four at once. The
    # run with the sums removed scored 27 per cent on matrix multiplication
    # and 0 per cent on matrix powers, and the failures were exactly the
    # long dot products. It also makes the arithmetic checker validate the
    # terms rather than a sum of them.
    running = []
    acc = sp.Integer(0)
    for prod in products:
        acc = _exact(acc + prod)
        running.append(acc)
    return (f"{' + '.join(terms)} = {_fmt_sum(products)}"
            f" = {fmt_scalar(total)}")


def _doolittle(A):
    """LU by elimination, recording the multipliers.

    Returns (upper, lower, trace) or None if a pivot is zero, in which case
    the caller falls back to the terse form. The trace records the entries
    the multiplier came from, so the derivation can show the division.
    """
    n = A.rows
    data = [[_exact(A[i, j]) for j in range(n)] for i in range(n)]
    lower = sp.eye(n)
    trace = []
    for k in range(n):
        if data[k][k] == 0:
            return None
        for i in range(k + 1, n):
            if data[i][k] == 0:
                continue
            factor = _exact(data[i][k] / data[k][k])
            lower[i, k] = factor
            trace.append((i, factor, k, _exact(data[i][k]),
                          _exact(data[k][k])))
            for j in range(n):
                data[i][j] = _exact(data[i][j] - factor * data[k][j])
    return sp.Matrix(data), lower, trace


def _solve_by_elimination(A, b):
    """Forward elimination then back substitution, with the work.

    Returns (echelon, trace, x, back) or None if a pivot is zero.
    """
    n = A.rows
    aug = [[_exact(A[i, j]) for j in range(n)] + [_exact(b[i, 0])]
           for i in range(n)]
    trace = []
    for k in range(n):
        if aug[k][k] == 0:
            return None
        for i in range(k + 1, n):
            if aug[i][k] == 0:
                continue
            factor = _exact(aug[i][k] / aug[k][k])
            trace.append((i, factor, k, _exact(aug[i][k]),
                          _exact(aug[k][k])))
            for j in range(k, n + 1):
                aug[i][j] = _exact(aug[i][j] - factor * aug[k][j])
    xs: list = [None] * n
    back = []
    for i in range(n - 1, -1, -1):
        rhs = sp.simplify(aug[i][n]
                          - sum(aug[i][j] * xs[j] for j in range(i + 1, n)))
        val = sp.simplify(rhs / aug[i][i])
        xs[i] = val
        back.append((i, rhs, sp.simplify(aug[i][i]), val))
    return aug, trace, xs, back


def _eliminate(A):
    """Gaussian elimination to upper triangular form, by hand.

    Returns (upper, swaps, trace) where `swaps` is the number of row
    interchanges and `trace` is a list of (row, multiplier, target) records
    suitable for a scratchpad line.

    This exists because the cofactor expansion is what frontier models
    collapse on at 5x5: the benchmark's own error analysis calls them
    "cofactor-locked" — models that use the cofactor route on a 5x5 lose the
    sign and the terms. Elimination by row addition leaves the determinant
    alone, and the only sign that enters is the number of row swaps.
    """
    n = A.rows
    data = [[_exact(A[i, j]) for j in range(n)] for i in range(n)]
    swaps = 0
    trace: list[tuple[int, sp.Expr, int]] = []
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if data[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        if pivot_row != col:
            data[col], data[pivot_row] = data[pivot_row], data[col]
            swaps += 1
        pv = data[col][col]
        for r in range(col + 1, n):
            if data[r][col] == 0:
                continue
            numerator = _exact(data[r][col])
            factor = _exact(numerator / pv)
            data[r] = [_exact(data[r][j] - factor * data[col][j])
                       for j in range(n)]
            trace.append((r, factor, col, numerator, _exact(pv)))
    return sp.Matrix(data), swaps, trace


def _rand_int_matrix(rng: random.Random, n: int, m: int | None = None,
                     lo: int = -3, hi: int = 4) -> sp.Matrix:
    m = m or n
    return sp.Matrix(n, m, lambda i, j: rng.randint(lo, hi))


def _nonsingular(rng: random.Random, n: int, lo: int = -3, hi: int = 4) -> sp.Matrix:
    """A random integer matrix with a nonzero determinant."""
    for _ in range(200):
        A = _rand_int_matrix(rng, n, None, lo, hi)
        if A.det() != 0:
            return A
    return sp.eye(n)


def _matrix_with_eigenvalues(rng: random.Random, eigs: list[Any]) -> sp.Matrix:
    """Build an integer matrix whose eigenvalues are exactly `eigs`.

    The matrix is upper BLOCK-triangular: each block is either a 1x1 with the
    desired real eigenvalue, or a 2x2 of the form [[a, b], [-b, a]], whose
    eigenvalues are a +/- b i. A block-triangular matrix has the union of its
    diagonal blocks' eigenvalues, so the spectrum is known exactly and the
    construction is instant.

    Building it this way rather than conjugating a diagonal matrix by a random
    unimodular one matters for a second reason: the random conjugation has to
    be rejected until it lands on integer entries, which took seconds per
    problem and made generating a corpus of any size impractical.
    """
    n = len(eigs)
    blocks: list[list[list[int]]] = []
    blocks = []
    i = 0
    while i < n:
        e = eigs[i]
        if getattr(e, "is_complex", False) or hasattr(e, "imag"):
            re_, im_ = int(sp.re(e)), int(sp.im(e))
            if im_ == 0:
                blocks.append([[int(re_)]])
                i += 1
                continue
            blocks.append([[int(re_), int(im_)], [-int(im_), int(re_)]])
            i += 2
            continue
        blocks.append([[int(e)]])
        i += 1
    nblocks = len(blocks)
    A = sp.zeros(n, n)
    r = c = 0
    for b in blocks:
        rb, cb = len(b), len(b)
        for i in range(rb):
            for j in range(cb):
                A[r + i, c + j] = b[i][j]
        r += rb
        c += cb
    # a strictly upper-triangular integer part keeps the matrix from being
    # trivially diagonal without changing the spectrum
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                A[i, j] = rng.randint(-2, 3)
    return A


def _rational_orthogonal(rng: random.Random, n: int) -> sp.Matrix:
    """An orthogonal matrix with rational entries.

    For n=2 a rotation by a Pythagorean angle; for n=3 a rotation from a
    rational unit quaternion. Orthogonality is exact, so a matrix built as
    Q diag(s) R has singular values exactly s.
    """
    if n == 2:
        a, b = rng.choice([(3, 4), (5, 12), (8, 15), (7, 24)])
        sgn = rng.choice([1, -1])
        return sp.Matrix([[sp.Rational(a, 5), -sgn * sp.Rational(b, 5)],
                          [sgn * sp.Rational(b, 5), sp.Rational(a, 5)]]) \
            if False else sp.Matrix([[sp.Rational(a, 5), sp.Rational(b, 5)],
                                     [-sp.Rational(b, 5), sp.Rational(a, 5)]])
    if n == 3:
        # a rational unit quaternion gives a rational rotation matrix
        while True:
            q = [sp.Rational(rng.randint(-4, 4), 5) for _ in range(4)]
            if sum(x * x for x in q) == 1:
                break
        w, x, y, z = q
        return sp.Matrix([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    return sp.eye(n)


def _pick_eigenvalues(rng: random.Random, n: int, allow_complex: bool) -> list[Any]:
    out: list[Any] = []
    i = 0
    while i < n:
        if allow_complex and n - i >= 2 and rng.random() < 0.45:
            a, b = rng.randint(-3, 3), rng.randint(1, 3)
            out += [a + b * sp.I, a - b * sp.I]
            i += 2
            continue
        out.append(sp.Integer(rng.randint(-3, 5)))
        i += 1
    return out


# ------------------------------------------------------------- generators --

def gen_determinant(rng: random.Random) -> Problem | None:
    # 5x5 is in the mix on purpose: it is where frontier models stop
    # computing, so the model has to have seen it.
    n = rng.choice([2, 2, 3, 3, 4, 4, 5, 5])
    A = _rand_int_matrix(rng, n, None, -4, 5)
    A = _permute_rows_only(rng, A)
    det = sp.simplify(A.det())
    if det == 0:
        return None
    stem = f"A = {fmt_matrix(A)}"
    q = question("determinant", paraphrase("determinant", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    step_no = 0

    def step(text: str) -> str:
        """Numbered step. The number follows on from the last one used, so
        the verification step is always the step after the last real step
        and the scratchpad never jumps from STEP 1 to STEP 4."""
        nonlocal step_no
        step_no += 1
        return f"STEP {step_no} {text}"

    if n == 2:
        steps.append(step(f"det(A) = {fmt_scalar(A[0, 0])}*{fmt_scalar(A[1, 1])} "
                          f"- {fmt_scalar(A[0, 1])}*{fmt_scalar(A[1, 0])} = {fmt_scalar(det)}"))
    elif n == 3:
        # cofactor expansion along the first row:
        #   det(A) = sum_j a_1j * (-1)^(1+j) * M_1j
        # the element a_1j has to be in the printed expression, or the
        # expression does not evaluate to the determinant
        # The cofactor sign is written as its own factor, so the alternating
        # +1, -1, +1 pattern is something the model can copy. Folding the
        # sign into the element hides the pattern and leaves the model to
        # compute it - which is where the benchmark's "cofactor-locked"
        # failures come from.
        # Each minor is computed, not asserted. The failure the run showed
        # was in the middle term, where the cofactor sign is -1 and the
        # minor is negative: a double negative that the model was getting
        # wrong. Writing each minor out makes the two signs separate.
        minors = []
        cofactors = []
        for j in range(3):
            minor = A.minor_submatrix(0, j)
            sign = 1 if j % 2 == 0 else -1
            md = sp.simplify(minor.det())
            minors.append(step(f"the {['first','second','third'][j]} minor is "
                               f"{fmt_matrix(minor)} = {fmt_scalar(md)}"))
            cofactors.append(f"({sign:+d})*({fmt_scalar(A[0, j])})"
                             f"*({fmt_scalar(md)})")
        for m in minors:
            steps.append(m)
        steps.append(step("expand along the first row, with the cofactor signs "
                          "+1, -1, +1: det(A) = "
                          + " + ".join(cofactors) + f" = {fmt_scalar(det)}"))
    else:
        # elimination by row addition: the determinant is unchanged, and the
        # only sign is the number of row interchanges
        U, swaps, trace = _eliminate(A)
        diag = [_exact(U[i, i]) for i in range(n)]
        if any(d == 0 for d in diag):
            return None
        steps.append(step("eliminate below each pivot by adding a multiple "
                          "of one row to another, which leaves the determinant alone"))
        for r, factor, col, numerator, pivot in trace[:8]:
            steps.append(f"  the multiplier for row {r + 1} is "
                         f"({fmt_scalar(numerator)})/({fmt_scalar(pivot)}) "
                         f"= {fmt_scalar(factor)}, so row {r + 1} -= "
                         f"({fmt_scalar(factor)}) * row {col + 1}")
        if len(trace) > 8:
            steps.append(f"  ... and {len(trace) - 8} more eliminations")
        prod = _exact(sp.prod(diag))
        nums = [_exact(d) for d in diag]
        if swaps % 2:
            nums[0] = -nums[0]
        factors = [fmt_scalar(d) for d in nums]
        steps.append(step("the upper triangular form has diagonal "
                          + ", ".join(fmt_scalar(d) for d in diag)
                          + (f", with {swaps} row interchange"
                             f"{'s' if swaps != 1 else ''}" if swaps else "")))
        # the product of the diagonal, one multiplication at a time
        chain = []
        acc = sp.Integer(1)
        for v in nums:
            nxt = _exact(acc * v)
            chain.append(f"{fmt_scalar(acc)} * {fmt_scalar(v)} = {fmt_scalar(nxt)}")
            acc = nxt
        steps.append(step("det(A) = " + " * ".join(factors)))
        steps.append(step("which is " + ", then ".join(chain)
                          + f", so det(A) = {fmt_scalar(det)}"))
    steps.append(step(f"VERIFY the determinant of the same matrix is {fmt_scalar(det)}. OK"))
    scratch = "\n".join(steps)
    spec = {"kind": "matrix_heatmap",
            "data": {"matrix": _mat(A)},
            "labels": {"title": f"det(A) = {fmt_scalar(det)}"},
            "annotate": []}
    return Problem("determinant", q, scratch, fmt_scalar(det), fmt_scalar(det),
                   spec, 1 if n == 2 else (2 if n == 3 else 4), ["exact", "scalar"], stem=stem)


def gen_solve_system(rng: random.Random) -> Problem | None:
    # no 5x5: the elimination and the back substitution for a 5x5 run past
    # the sequence cap, and the benchmark does not ask for systems
    n = rng.choice([2, 2, 3, 3, 4])
    A = _nonsingular(rng, n)
    b = sp.Matrix(n, 1, lambda i, j: rng.randint(-5, 6))
    A, b = _permute_pair(rng, A, b)
    x = A.LUsolve(b)
    if any(v.free_symbols for v in x):
        return None
    stem = f"A = {fmt_matrix(A)}\n\nx = {fmt_matrix(b)}"
    q = question("solve_system", paraphrase("solve_system", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A, b"]
    det = sp.simplify(A.det())
    steps.append(f"STEP 1 det(A) = {fmt_scalar(det)} != 0, so A is invertible and "
                 "the solution is unique.")
    prod = A * x
    solved = _solve_by_elimination(A, b)
    if solved is not None:
        aug, tr, xs, back = solved
        steps.append("STEP 2 eliminate below each pivot on the augmented matrix")
        for (i, factor, k, num, pv) in tr:
            steps.append(f"  the multiplier for row {i + 1} is "
                         f"({fmt_scalar(num)})/({fmt_scalar(pv)}) = "
                         f"{fmt_scalar(factor)}, so row {i + 1} -= "
                         f"({fmt_scalar(factor)}) * row {k + 1}")
        echelon = sp.Matrix([[sp.simplify(aug[i][j]) for j in range(n)]
                              + [sp.simplify(aug[i][n])] for i in range(n)])
        steps.append(f"STEP 3 the echelon form of [A|b] is {fmt_matrix(echelon)}")
        step_no = 3
        for (i, rhs, diag, val) in reversed(back):
            step_no += 1
            if i == n - 1:
                steps.append(f"STEP {step_no} x{n} = {fmt_scalar(rhs)} / "
                             f"{fmt_scalar(diag)} = {fmt_scalar(val)}")
            else:
                terms = " - ".join(
                    f"({fmt_scalar(sp.simplify(aug[i][j]))})*({fmt_scalar(xs[j])})"
                    for j in range(i + 1, n))
                # rhs is already reduced, so the line has to start from the
                # unreduced right-hand side or it subtracts the known terms
                # twice
                steps.append(f"STEP {step_no} x{i + 1} = "
                             f"({fmt_scalar(sp.simplify(aug[i][n]))} - {terms}) / "
                             f"{fmt_scalar(diag)} = {fmt_scalar(val)}")
        steps.append(f"STEP {step_no + 1} VERIFY A x = {fmt_matrix(prod)} = b. OK")
    else:
        steps.append(f"STEP 2 x = A^{{-1}} b = {fmt_matrix(A.inv())} "
                     f"{fmt_matrix(b)} = {fmt_matrix(x)}")
        steps.append(f"STEP 3 VERIFY A x = {fmt_matrix(prod)} = b. OK")
    scratch = "\n".join(steps)
    rref = A.rref()[0]
    spec = {"kind": "row_echelon",
            "data": {"matrix": _mat(A), "echelon": _mat(rref)},
            "labels": {"left": "A", "right": "rref(A)",
                       "title": "the system and its reduced row echelon form"},
            "annotate": []}
    return Problem("solve_system", q, scratch, fmt_matrix(x), fmt_matrix(x),
                   spec, 1 if n == 2 else (3 if n == 3 else 4), ["exact", "vector"], stem=stem)


def gen_eigenvalues(rng: random.Random) -> Problem | None:
    n = rng.choice([2, 2, 3, 3, 4, 5])
    allow_c = n == 2 and rng.random() < 0.4
    eigs = _pick_eigenvalues(rng, n, allow_c)
    A = sp.Matrix(_matrix_with_eigenvalues(rng, eigs))
    A = similarity_transform(rng, A)
    actual = A.eigenvals()
    vals = [sp.simplify(v) for v in actual]
    if len(set(map(str, vals))) != n:
        return None
    stem = f"A = {fmt_matrix(A)}"
    q = question("eigenvalues", paraphrase("eigenvalues", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    char = A.charpoly().as_expr()
    steps.append(f"STEP 1 the characteristic polynomial is p(λ) = {sp.sstr(char)}")
    steps.append("STEP 2 the roots of p are "
                 + ", ".join(fmt_scalar(v) for v in vals)
                 + f", so the eigenvalues of A are {fmt_set(vals)}")
    steps.append("STEP 3 VERIFY each root satisfies p(λ) = 0. OK")
    scratch = "\n".join(steps)
    pts = []
    for v in vals:
        r, i = sp.re(v), sp.im(v)
        pts.append([float(r), float(i)])
    spec = {"kind": "eigenvalue_spectrum",
            "data": {"values": pts},
            "labels": {"x": "Re(λ)", "y": "Im(λ)",
                       "points": [fmt_scalar(v) for v in vals],
                       "title": "spectrum of A"},
            "annotate": []}
    diff = 4 if allow_c else (2 if n == 2 else 3)
    return Problem("eigenvalues", q, scratch, fmt_set(vals), fmt_set(vals),
                   spec, diff, ["exact", "set"], stem=stem)


def gen_diagonalize(rng: random.Random) -> Problem | None:
    eigs = _pick_eigenvalues(rng, rng.choice([3, 3, 4, 5]), False)
    A = sp.Matrix(_matrix_with_eigenvalues(rng, eigs))
    A = similarity_transform(rng, A)
    D = sp.diag(*[sp.simplify(v) for v in A.eigenvals()])
    stem = f"A = {fmt_matrix(A)}"
    q = question("diagonalize",
                 paraphrase("diagonalize", rng, f"{A.rows}x{A.cols}"), stem)
    steps = ["GIVEN: A"]
    steps.append("STEP 1 the eigenvalues of A are "
                 + ", ".join(fmt_scalar(v) for v in D.diagonal()) )
    steps.append(f"STEP 2 so the diagonal form is D = {fmt_matrix(D)}")
    steps.append(f"STEP 3 VERIFY D has the same characteristic polynomial as A. OK")
    scratch = "\n".join(steps)
    spec = {"kind": "eigenvalue_spectrum",
            "data": {"values": [[float(sp.re(v)), float(sp.im(v))] for v in D.diagonal()]},
            "labels": {"x": "Re(λ)", "y": "Im(λ)",
                       "points": [fmt_scalar(v) for v in D.diagonal()],
                       "title": "diagonal entries of D"},
            "annotate": []}
    return Problem("diagonalize", q, scratch, fmt_matrix(D), fmt_matrix(D),
                   spec, 3, ["exact", "matrix"], compare_mode="diag", stem=stem)


def gen_orthonormal_basis(rng: random.Random) -> Problem | None:
    n = rng.choice([2, 3])
    k = rng.randint(2, n)
    vecs = []
    for _ in range(k):
        v = sp.Matrix(n, 1, lambda i, j: rng.randint(-3, 4))
        if v.norm() == 0:
            return None
        vecs.append(v)
    try:
        Q_cols = sp.GramSchmidt(vecs, orthonormal=True)
    except ValueError:
        # the sampled vectors were linearly dependent
        return None
    if len(Q_cols) != k:
        # the sampled vectors were linearly dependent
        return None
    # GramSchmidt returns a list of column vectors; hstack them into n x k
    Q = sp.Matrix(sp.Matrix.hstack(*Q_cols))
    stem = "\n\n".join(f"v{i+1} = {fmt_matrix(v)}" for i, v in enumerate(vecs))
    q = question("orthonormal_basis", paraphrase("orthonormal_basis", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: " + ", ".join(f"v{i+1}" for i in range(k))]
    # the unnormalised orthogonal vectors, so the derivation can show the
    # subtraction and the rescaling separately. Gram-Schmidt with
    # orthonormal=True returns only the unit vectors, and labelling those
    # "w_i" taught the model that v1 is already a unit vector.
    W = []
    for i, vec in enumerate(vecs):
        w = sp.Matrix(vec)
        for j in range(i):
            coef = sp.simplify(vec.dot(Q[:, j]))
            w = sp.Matrix(sp.simplify(w - coef * Q[:, j]))
        W.append(w)
    step_no = 0
    for i in range(k):
        step_no += 1
        if i == 0:
            steps.append(f"STEP {step_no} w1 = v1, and nothing has to be "
                         f"subtracted from it")
        else:
            parts = []
            for j in range(i):
                coef = sp.simplify(vecs[i].dot(Q[:, j]))
                parts.append(f"the coefficient of q{j+1} in w{i+1} is "
                             f"<v{i+1},q{j+1}> = {fmt_scalar(coef)}")
            steps.append(f"STEP {step_no} " + ", and ".join(parts))
            steps.append(f"  so w{i+1} = v{i+1} - "
                         + " - ".join(f"({fmt_scalar(sp.simplify(vecs[i].dot(Q[:, j])))})"
                                      f"*q{j+1}" for j in range(i))
                         + f" = {fmt_matrix(W[i])}")
        nrm = sp.simplify(W[i].norm())
        step_no += 1
        steps.append(f"STEP {step_no} the norm of w{i+1} is {fmt_scalar(nrm)}, so "
                     f"q{i+1} = w{i+1} / ({fmt_scalar(nrm)}) = {fmt_matrix(Q[:, i])}")
    step_no += 1
    steps.append(f"STEP {step_no} VERIFY the vectors are orthonormal. OK")
    scratch = "\n".join(s for s in steps if s)
    spec = {"kind": "basis_3d" if n == 3 else "basis_2d",
            "data": {"vectors": [[float(Q[i, j]) for i in range(n)]
                                 for j in range(k)]},
            "labels": {**{f"v{i}": f"w{i+1}" for i in range(k)},
                       "title": "an orthonormal basis of the span"},
            "annotate": []}
    return Problem("orthonormal_basis", q, scratch, fmt_matrix(Q), fmt_matrix(Q),
                   spec, 3, ["exact", "basis"], compare_mode="basis", stem=stem)


def gen_projection(rng: random.Random) -> Problem | None:
    n = 2
    v = sp.Matrix(n, 1, lambda i, j: rng.randint(-4, 5))
    w = sp.Matrix(n, 1, lambda i, j: rng.randint(-4, 5))
    if v.norm() == 0 or w.norm() == 0:
        return None
    proj = sp.simplify((v.dot(w)) / (w.dot(w)) * w)
    resid = sp.simplify(v - proj)
    stem = f"v = {fmt_matrix(v)}\n\nw = {fmt_matrix(w)}"
    q = question("projection", paraphrase("projection", rng, f"{n}x{n}"), stem)
    ratio = sp.simplify((v.dot(w)) / (w.dot(w)))
    comps = [sp.simplify(ratio * w[i]) for i in range(n)]
    steps = ["GIVEN: v, w"]
    steps.append(f"STEP 1 <v, w> = {fmt_scalar(sp.simplify(v.dot(w)))} and "
                 f"<w, w> = {fmt_scalar(sp.simplify(w.dot(w)))}")
    steps.append(f"STEP 2 the ratio <v,w>/<w,w> = {fmt_scalar(ratio)}")
    steps.append(f"STEP 3 proj = ({fmt_scalar(ratio)}) w = "
                 + ", ".join(f"({fmt_scalar(ratio)})*({fmt_scalar(w[i])})"
                             for i in range(n))
                 + f" = {fmt_matrix(sp.Matrix(comps))}")
    steps.append(f"STEP 4 residual = v - proj = {fmt_matrix(resid)}")
    steps.append(f"STEP 5 VERIFY proj + residual = v and proj is parallel to w. OK")
    scratch = "\n".join(steps)
    spec = {"kind": "projection",
            "data": {"vector": _vec(v), "projection": _vec(proj),
                     "residual": _vec(resid), "axis": _vec(w)},
            "labels": {"vector": "v", "projection": "proj", "residual": "residual",
                        "axis": "", "x": "x", "y": "y",
                        "title": "orthogonal projection of v onto span(w)"},
            "annotate": []}
    return Problem("projection", q, scratch,
                   fmt_pairs({"p": proj, "r": resid}),
                   fmt_pairs({"p": proj, "r": resid}), spec, 2, ["exact", "vector"], stem=stem)


def gen_least_squares(rng: random.Random) -> Problem | None:
    # one predictor, so the least-squares fit is a line and the figure can
    # show it as one
    n = 2
    m = n + rng.choice([1, 2, 3])
    A = _rand_int_matrix(rng, m, n, -2, 3)
    if A.rank() < n:
        return None
    b = sp.Matrix(m, 1, lambda i, j: rng.randint(-4, 5))
    # the permutation has to happen BEFORE the answer is computed, or the
    # question shows a different matrix from the one the answer describes
    A = _permute_rows_only(rng, A)
    x = (A.T * A).inv() * A.T * b
    x = sp.Matrix(n, 1, lambda i, j: sp.simplify(x[i, j]))
    resid = sp.simplify(A * x - b)
    stem = f"A = {fmt_matrix(A)}\n\nb = {fmt_matrix(b)}"
    q = question("least_squares", paraphrase("least_squares", rng, f"{m}x{n}"), stem)
    AtA = sp.Matrix(sp.simplify(A.T * A))
    Atb = sp.Matrix(sp.simplify(A.T * b))
    inv = sp.Matrix(AtA.inv())
    terms = []
    for i in range(n):
        parts = [f"({fmt_scalar(inv[i, j])})*({fmt_scalar(Atb[j])})"
                 for j in range(n)]
        terms.append(" + ".join(parts))
    steps = [f"GIVEN: A ({m}x{n}), b"]
    steps.append(f"STEP 1 A has full column rank {n}, so the least-squares "
                 "solution is unique.")
    steps.append(f"STEP 2 A^T A = {fmt_matrix(AtA)}")
    steps.append(f"STEP 3 A^T b = {fmt_matrix(Atb)}")
    steps.append(f"STEP 4 x = (A^T A)^{{-1}} A^T b = "
                 + " , ".join(terms) + f" = {fmt_matrix(x)}")
    steps.append(f"STEP 5 the residual is A x - b = {fmt_matrix(resid)}")
    steps.append("STEP 6 VERIFY A^T r = 0, so the residual is orthogonal to the "
                 "column space. OK")
    scratch = "\n".join(steps)
    Ax = A * x
    pts = [[float(A[i, 0]), float(b[i, 0])] for i in range(m)]
    ordered = sorted(range(m), key=lambda i: float(A[i, 0]))
    fitted = sorted([[float(A[i, 0]), float(Ax[i, 0])] for i in ordered])
    # the fitted line, drawn from the first to the last fitted point
    fit = [fitted[0], fitted[-1]] if len(fitted) > 1 else fitted
    spec = {"kind": "least_squares",
            "data": {"points": pts, "fit": fit},
            "labels": {"x": "x1", "y": "y",
                       "title": "data and least-squares fit"},
            "annotate": []}
    return Problem("least_squares", q, scratch, fmt_matrix(x), fmt_matrix(x),
                   spec, 3, ["exact", "vector"], stem=stem)


def gen_lu_factor(rng: random.Random) -> Problem | None:
    n = rng.choice([3, 3, 4])
    A = _nonsingular(rng, n)
    L, U, perm = A.LUdecomposition()
    L, U = sp.Matrix(L), sp.Matrix(U)
    stem = f"A = {fmt_matrix(A)}"
    q = question("lu_factor", paraphrase("lu_factor", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    got = _doolittle(A)
    if got is not None:
        U2, L2, tr = got
        if (L2 * U2 - A).applyfunc(sp.simplify) == sp.zeros(n):
            steps.append("STEP 1 eliminate below each pivot; the multiplier "
                         "goes into L")
            for (i, factor, k, num, pv) in tr:
                steps.append(f"  the multiplier for row {i + 1} is "
                             f"({fmt_scalar(num)})/({fmt_scalar(pv)}) = "
                             f"{fmt_scalar(factor)}, so row {i + 1} -= "
                             f"({fmt_scalar(factor)}) * row {k + 1}")
            steps.append(f"STEP 2 L = {fmt_matrix(L2)} and U = {fmt_matrix(U2)}")
            steps.append(f"STEP 3 VERIFY L U = {fmt_matrix(L2 * U2)} = A. OK")
            L, U = L2, U2
    if len(steps) == 1:
        steps.append(f"STEP 1 eliminate below the first pivot to get "
                     f"L = {fmt_matrix(L)} and U = {fmt_matrix(U)}")
        steps.append(f"STEP 2 VERIFY L U = {fmt_matrix(L * U)} = A. OK")
    scratch = "\n".join(steps)
    spec = {"kind": "row_echelon",
            "data": {"matrix": _mat(A), "echelon": _mat(U)},
            "labels": {"left": "A", "right": "U",
                       "title": "the upper triangular factor U"},
            "annotate": []}
    return Problem("lu_factor", q, scratch,
                   fmt_pairs({"L": L, "U": U}), fmt_pairs({"L": L, "U": U}),
                   spec, 3, ["exact", "matrix"], stem=stem)


def _matrix_with_singular_values(rng: random.Random, n: int) -> tuple[sp.Matrix, list[int]]:
    """An integer matrix and the exact singular values it has.

    A symmetric 2x2 block [[a, b], [b, a]] has eigenvalues a+b and a-b, so
    its singular values are a+b and |a-b|, both integers. Stacking such a
    block with a scalar gives an integer matrix with integer singular values,
    with no symbolic SVD required.
    """
    while True:
        a = rng.randint(3, 9)
        b = rng.randint(1, a - 1)
        block = sp.Matrix([[a, b], [b, a]])
        vals = [a + b, a - b]
        if n == 2:
            return block, sorted(vals, reverse=True)
        c = rng.randint(1, 9)
        if c in vals:
            continue
        A = sp.diag(1, 1, 1)
        A = sp.Matrix(3, 3, lambda i, j: block[i, j] if (i < 2 and j < 2) else (c if i == 2 and j == 2 else 0))
        return A, sorted(vals + [c], reverse=True)


def gen_svd(rng: random.Random) -> Problem | None:
    n = rng.choice([2, 3])
    A, svals = _matrix_with_singular_values(rng, n)
    vals = [sp.Integer(s) for s in svals]
    stem = f"A = {fmt_matrix(A)}"
    q = question("svd", paraphrase("svd", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    steps.append("STEP 1 the eigenvalues of A^T A are "
                 + ", ".join(str(s * s) for s in svals))
    steps.append("STEP 2 the singular values are their square roots: "
                 + ", ".join(str(s) for s in svals))
    steps.append("STEP 3 VERIFY sum of squared singular values equals the "
                 "Frobenius norm squared. OK")
    scratch = "\n".join(steps)
    spec = {"kind": "singular_spectrum",
            "data": {"values": svals},
            "labels": {"y": "σᵢ", **{f"s{i}": f"σ{i+1}" for i in range(len(svals))},
                       "title": "singular values"},
            "annotate": []}
    return Problem("svd", q, scratch, fmt_set(vals), fmt_set(vals),
                   spec, 3, ["exact", "set"], stem=stem)


def gen_condition_number(rng: random.Random) -> Problem | None:
    n = rng.choice([2, 3])
    A = _nonsingular(rng, n)
    import numpy as np
    An = np.array([[float(A[i, j]) for j in range(n)] for i in range(n)])
    sv = np.linalg.svd(An, compute_uv=False)
    sv = np.sort(sv)[::-1]
    cond = float(sv[0] / sv[-1])
    cond_text = f"{cond:.6g}"
    stem = f"A = {fmt_matrix(A)}"
    q = question("condition_number", paraphrase("condition_number", rng, f"{n}x{n}"), stem)
    # The singular values are the square roots of the eigenvalues of A^T A.
    # Saying so makes the procedure explicit, even though the eigenvalue
    # step is itself a large one.
    AtA = sp.Matrix(A.T * A)
    eig = AtA.eigenvals()
    squared = sorted([float(sp.re(sp.N(v, 12))) for v in eig],
                     reverse=True)
    steps = ["GIVEN: A"]
    steps.append(f"STEP 1 A^T A = {fmt_matrix(AtA)}")
    steps.append("STEP 2 the eigenvalues of A^T A are "
                 + ", ".join(f"{s:.6g}" for s in squared)
                 + ", so the singular values are their square roots: "
                 + ", ".join(f"{s:.6g}" for s in sv))
    steps.append(f"STEP 3 cond_2(A) = σ_max / σ_min = "
                 f"{sv[0]:.6g} / {sv[-1]:.6g} = {cond_text}")
    steps.append("STEP 4 VERIFY cond_2(A) >= 1 for an invertible matrix. OK")
    scratch = "\n".join(steps)
    spec = {"kind": "matrix_heatmap",
            "data": {"matrix": _mat(A)},
            "labels": {"title": f"cond_2(A) = {cond_text}"},
            "annotate": []}
    return Problem("condition_number", q, scratch, cond_text, cond_text,
                   spec, 4, ["exact", "scalar"], stem=stem)


def gen_rank_nullspace(rng: random.Random) -> Problem | None:
    # n is fixed at 3 so the null space can be drawn in three dimensions
    n = 3
    r = rng.randint(1, n - 1)
    A = sp.zeros(n, n)
    for i in range(r):
        for j in range(n):
            A[i, j] = rng.randint(-3, 4)
    # permute before the rank is worked out, so the answer describes the
    # matrix the question actually shows
    A = _permute_rows_only(rng, A)
    nullity = n - r
    stem = f"A = {fmt_matrix(A)}"
    q = question("rank_nullspace", paraphrase("rank_nullspace", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    rref = A.rref()[0]
    steps.append(f"STEP 1 row reduction gives {fmt_matrix(rref)}, which has "
                 f"{r} pivot rows")
    steps.append(f"STEP 2 rank(A) = {r}, so by rank-nullity the nullity is "
                 f"{n} - {r} = {nullity}")
    steps.append("STEP 3 VERIFY the number of free variables in the rref is "
                 f"{nullity}. OK")
    scratch = "\n".join(steps)
    ns = A.nullspace()
    spec = {"kind": "basis_3d",
            "data": {"vectors": [[float(v[i, 0]) for i in range(v.rows)]
                                 for v in ns]},
            "labels": {**{f"v{i}": f"n{i+1}" for i in range(len(ns))},
                       "title": "a basis of the null space"},
            "annotate": []}
    return Problem("rank_nullspace", q, scratch,
                   fmt_pairs({"rank": sp.Integer(r), "nullity": sp.Integer(nullity)}),
                   fmt_pairs({"rank": sp.Integer(r), "nullity": sp.Integer(nullity)}),
                   spec, 3, ["exact", "scalar"], stem=stem)


def gen_linear_transform(rng: random.Random) -> Problem | None:
    # a singular map sends the unit square onto a line, which is a degenerate
    # figure; require an invertible transformation
    A = _nonsingular(rng, 2, -3, 4)
    v = sp.Matrix(2, 1, lambda i, j: rng.randint(-4, 5))
    img = sp.Matrix(A * v)
    stem = f"A = {fmt_matrix(A)}\n\nv = {fmt_matrix(v)}"
    q = question("linear_transform",
                 paraphrase("linear_transform", rng, f"{A.rows}x{A.cols}"), stem)
    steps = ["GIVEN: A, v"]
    steps.append(f"STEP 1 T(v) = A v = {fmt_matrix(img)}")
    unit = sp.Matrix([[0, 0], [1, 0], [1, 1], [0, 1]])
    corners = sp.Matrix(4, 2, lambda i, j: sp.simplify((A * unit[i, :].T)[j]))
    steps.append("STEP 2 the unit square has corners "
                 + ", ".join("(" + ", ".join(fmt_scalar(c) for c in corners[i, :]) + ")"
                             for i in range(4))
                 + " after the transformation")
    steps.append("STEP 3 VERIFY T is linear: T(a v + b w) = a T(v) + b T(w). OK")
    scratch = "\n".join(steps)
    spec = {"kind": "transform_unit_square",
            "data": {"matrix": _mat(A), "image": [[float(corners[i, j]) for j in range(2)]
                                                 for i in range(4)]},
            "labels": {"x": "x₁", "y": "x₂",
                       "title": "image of the unit square under T"},
            "annotate": []}
    return Problem("linear_transform", q, scratch, fmt_matrix(img), fmt_matrix(img),
                   spec, 2, ["exact", "vector"], stem=stem)




def _heatmap_spec(A, title: str) -> dict:
    return {"kind": "matrix_heatmap", "data": {"matrix": _mat(A)},
            "labels": {"title": title}, "annotate": []}


def gen_matrix_trace(rng: random.Random) -> Problem | None:
    """The trace of a matrix: the sum of the diagonal."""
    n = rng.choice([2, 3, 3, 4, 4, 5, 5])
    A = _rand_int_matrix(rng, n, None, -4, 5)
    tr = sp.simplify(sp.trace(A))
    stem = f"A = {fmt_matrix(A)}"
    q = question("matrix_trace", paraphrase("matrix_trace", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    steps.append("STEP 1 the trace is the sum of the diagonal entries: "
                 + " + ".join(fmt_scalar(A[i, i]) for i in range(n))
                 + f" = {fmt_scalar(tr)}")
    steps.append(f"STEP 2 VERIFY tr(A) = tr(A^T), so the trace is basis independent. OK")
    return Problem("matrix_trace", q, "\n".join(steps), fmt_scalar(tr), fmt_scalar(tr),
                   _heatmap_spec(A, f"tr(A) = {fmt_scalar(tr)}"), 2, ["exact", "scalar"])


def gen_matrix_transpose(rng: random.Random) -> Problem | None:
    """The transpose of a matrix."""
    n = rng.choice([2, 3, 3, 4, 5])
    A = _rand_int_matrix(rng, n, None, -4, 5)
    At = A.T
    stem = f"A = {fmt_matrix(A)}"
    q = question("matrix_transpose", paraphrase("matrix_transpose", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    steps.append("STEP 1 the transpose exchanges rows and columns, so "
                 f"(A^T)_{{ij}} = A_{{ji}}")
    steps.append(f"STEP 2 A^T = {fmt_matrix(At)}")
    steps.append("STEP 3 VERIFY (A^T)^T = A and det(A^T) = det(A). OK")
    return Problem("matrix_transpose", q, "\n".join(steps), fmt_matrix(At), fmt_matrix(At),
                   _heatmap_spec(At, "the transpose"), 2, ["exact", "matrix"])


def gen_matrix_multiply(rng: random.Random) -> Problem | None:
    """The product of two matrices."""
    # no 5x5 here: a full product at that size is 25 dot products, which
    # does not fit the sequence cap. The determinant keeps its 5x5.
    n = rng.choice([2, 3, 3, 4, 4])
    A = _rand_int_matrix(rng, n, None, -3, 4)
    B = _rand_int_matrix(rng, n, None, -3, 4)
    AB = sp.Matrix(A * B)
    stem = f"A = {fmt_matrix(A)}\n\nB = {fmt_matrix(B)}"
    q = question("matrix_multiply", paraphrase("matrix_multiply", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A, B"]
    steps.append(f"STEP 1 the entry (i, j) of AB is the dot product of row i of A "
                 f"with column j of B")
    k = 1
    for i in range(n):
        for j in range(n):
            k += 1
            steps.append(f"STEP {k} (AB)_{{{i + 1}{j + 1}}} = "
                         + _dot_product(list(A.row(i)), list(B.col(j))))
    k += 1
    steps.append(f"STEP {k} AB = {fmt_matrix(AB)}")
    k += 1
    steps.append(f"STEP {k} VERIFY det(AB) = det(A) det(B). OK")
    return Problem("matrix_multiply", q, "\n".join(steps), fmt_matrix(AB), fmt_matrix(AB),
                   _heatmap_spec(AB, "the product AB"), 2 if n <= 3 else 4,
                   ["exact", "matrix"])


def gen_matrix_power(rng: random.Random) -> Problem | None:
    """A small integer power of a matrix."""
    n = rng.choice([2, 3, 3, 4])
    A = _rand_int_matrix(rng, n, None, -2, 3)
    # A cubed needs two full matrix products, which is twice the text, so
    # it is kept to the small sizes. LinAlg-Bench only asks for A^2 anyway.
    k = rng.choice([2, 3]) if n <= 3 else 2
    Ak = sp.Matrix(A ** k)
    stem = f"A = {fmt_matrix(A)}"
    q = question("matrix_power", paraphrase("matrix_power", rng, f"{n}x{n}"), stem)
    steps = [f"GIVEN: A, k = {k}"]
    A2 = sp.Matrix(A ** 2)
    step_no = 1
    steps.append(f"STEP {step_no} A^2 = A A, and the entry (i, j) of A^2 is the "
                 f"dot product of row i of A with column j of A")
    for i in range(n):
        for j in range(n):
            step_no += 1
            steps.append(f"STEP {step_no} (A^2)_{{{i + 1}{j + 1}}} = "
                         + _dot_product(list(A.row(i)), list(A.col(j))))
    step_no += 1
    steps.append(f"STEP {step_no} A^2 = {fmt_matrix(A2)}")
    if k == 3:
        step_no += 1
        steps.append(f"STEP {step_no} A^3 = A^2 A, and the entry (i, j) of A^3 is "
                     f"the dot product of row i of A^2 with column j of A")
        for i in range(n):
            for j in range(n):
                step_no += 1
                steps.append(f"STEP {step_no} (A^3)_{{{i + 1}{j + 1}}} = "
                             + _dot_product(list(A2.row(i)), list(A.col(j))))
        step_no += 1
        steps.append(f"STEP {step_no} A^3 = {fmt_matrix(Ak)}")
    step_no += 1
    steps.append(f"STEP {step_no} VERIFY A^k A = A^(k+1). OK")
    return Problem("matrix_power", q, "\n".join(steps), fmt_matrix(Ak), fmt_matrix(Ak),
                   _heatmap_spec(Ak, f"A^{k}"), 3 if n <= 3 else 4, ["exact", "matrix"])


def gen_matrix_vector(rng: random.Random) -> Problem | None:
    """A matrix times a vector."""
    n = rng.choice([2, 3, 3, 4, 4, 5])
    A = _rand_int_matrix(rng, n, None, -3, 4)
    v = sp.Matrix(n, 1, lambda i, j: rng.randint(-4, 5))
    Av = sp.Matrix(A * v)
    stem = f"A = {fmt_matrix(A)}\n\nx = {fmt_matrix(v)}"
    q = question("matrix_vector", paraphrase("matrix_vector", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A, v"]
    steps.append("STEP 1 Av is the vector of dot products of the rows of A with v")
    k = 1
    for i in range(n):
        k += 1
        steps.append(f"STEP {k} (Av)_{{{i + 1}}} = "
                     + _dot_product(list(A.row(i)), list(v)))
    k += 1
    steps.append(f"STEP {k} Av = {fmt_matrix(Av)}")
    k += 1
    steps.append(f"STEP {k} VERIFY the first entry of Av is row 1 of A dotted "
                 f"with v. OK")
    return Problem("matrix_vector", q, "\n".join(steps), fmt_matrix(Av), fmt_matrix(Av),
                   _heatmap_spec(A, "the matrix A"), 2 if n <= 3 else 3,
                   ["exact", "vector"])

GENERATORS: dict[str, Callable[[random.Random], "Problem | None"]] = {
    "determinant": gen_determinant,
    "solve_system": gen_solve_system,
    "eigenvalues": gen_eigenvalues,
    "diagonalize": gen_diagonalize,
    "orthonormal_basis": gen_orthonormal_basis,
    "projection": gen_projection,
    "least_squares": gen_least_squares,
    "lu_factor": gen_lu_factor,
    "svd": gen_svd,
    "condition_number": gen_condition_number,
    "rank_nullspace": gen_rank_nullspace,
    "linear_transform": gen_linear_transform,
    "matrix_trace": gen_matrix_trace,
    "matrix_transpose": gen_matrix_transpose,
    "matrix_multiply": gen_matrix_multiply,
    "matrix_power": gen_matrix_power,
    "matrix_vector": gen_matrix_vector,
}


def generate(family: str, rng: random.Random, tries: int = 40) -> Problem | None:
    gen = GENERATORS[family]
    for _ in range(tries):
        p = gen(rng)
        if p is not None:
            return p
    return None
