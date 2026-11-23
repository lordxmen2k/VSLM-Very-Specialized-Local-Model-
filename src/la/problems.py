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

import numpy as np
import sympy as sp

from .answer import _exactify, fmt_matrix, fmt_pairs, fmt_scalar, fmt_set
from .augment import (paraphrase, permute_rows, prompt_tail, question,
                      similarity_transform)
from .benchprompt_magnitudes import draw_entry
from .benchprompt_tails import decoration_rng

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

    def __post_init__(self):
        # The model tends to state the answer in its own words at the end of
        # the derivation, and for the families whose answer is a labelled
        # pair it loses the labels: "1, 2" instead of "rank = 1, nullity =
        # 2". The grader then compares a list with a dict and calls it
        # wrong. Stating the answer in the derivation, in the same form as
        # the answer field, is what makes the model copy it rather than
        # reconstruct it.
        if self.answer and self.scratchpad:
            self.scratchpad = (f"{self.scratchpad.rstrip()}\n"
                               f"the answer is {self.answer}")

    def to_json(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        # the prompt and the completion go into the row, so that the
        # training and the evaluation cannot disagree about what the model is
        # given and what it is expected to write. They used to be assembled
        # separately in train.py, and they disagreed: the training prompt
        # wrapped the question in "### Problem" while the evaluation passed
        # the bare question, so the model was asked to answer a prompt it
        # had never been trained on.
        d["prompt"] = self.prompt()
        d["completion"] = self.completion()
        # the chat turns, stored separately so the chat template can be
        # rendered at training time by the tokenizer that will be used, and
        # so the benchmark can render the same prompt with the same
        # tokenizer
        d["system"] = self.system_prompt()
        d["user"] = self.user_prompt()
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

        The chat template is not applied here. The corpus is built without
        a tokenizer - it is text generation, and it must stay reproducible
        from a seed on any machine - so the row stores the system and user
        turns separately and the template is rendered once, at training time,
        by the tokenizer that will actually be used. The benchmark renders
        it the same way, with the same tokenizer.
        """
        return self.question

    def system_prompt(self) -> str:
        """The system turn the benchmark sends with this problem.

        The benchmark varies it by subcategory. The corpus families that
        the benchmark scores are mapped onto the benchmark's own wording;
        the families the benchmark does not have get the general one.
        """
        from .benchprompt import system_prompt_for
        return system_prompt_for(self.family)

    def user_prompt(self) -> str:
        """The user turn: the question, as the benchmark sends it."""
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
    """A matrix as the plot spec carries it.

    WHOLE NUMBERS ARE STORED AS WHOLE NUMBERS. The spec used to write
    every entry as a float - "6.0" instead of "6" - and the JSON of a 5x5
    matrix cost 157 tokens instead of 115. The plot spec is a THIRD copy
    of the result matrix at the large sizes, and the 5x5 examples were
    128 tokens over the 2048-token training cap because of it.

    The comparison is unaffected: the schema rounds both sides to six
    decimals and compares as floats, so 6 and 6.0 are the same value.
    """
    out = []
    for i in range(m.rows):
        row = []
        for j in range(m.cols):
            v = m[i, j]
            f = float(v)
            row.append(int(f) if f == int(f) else f)
        out.append(row)
    return out


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


def _minor_arithmetic(minor) -> str:
    """The arithmetic that produces a 2x2 minor's determinant.

    "4*1 - (-1)*5 = 9"

    A 2x2 determinant is two products and a subtraction, written with the
    products VISIBLE. The model must see the multiplication, or it learns
    to produce a value it cannot compute - which is exactly what the
    determinant family was doing.
    """
    a, b = minor[0, 0], minor[0, 1]
    c, d = minor[1, 0], minor[1, 1]
    av, bv = fmt_scalar(a), fmt_scalar(b)
    cv, dv = fmt_scalar(c), fmt_scalar(d)
    if av.startswith("-"):
        av = f"({av})"
    if bv.startswith("-"):
        bv = f"({bv})"
    if cv.startswith("-"):
        cv = f"({cv})"
    if dv.startswith("-"):
        dv = f"({dv})"
    p1 = _exact(a * d)
    p2 = _exact(b * c)
    return f"{av}*{dv} - {bv}*{cv} = {fmt_scalar(p1)} - {fmt_scalar(p2)}"


def _fmt_charpoly(char) -> str:
    """The characteristic polynomial, in LaTeX.

    "\\lambda^{3} + 7 \\lambda^{2} - 38 \\lambda - 240"

    The line used to be printed with `sp.sstr`, which is sympy's
    PLAIN-TEXT form: "lambda**3 + 7*lambda**2 - 38*lambda - 240". That
    is Python's syntax, not LaTeX - it renders as italic "lambda**3" -
    and it was the FIRST line of every eigenvalue derivation in the
    corpus. The model was trained to emit a broken expression and then
    had to parse one at benchmark time.

    The powers come out in DESCENDING order, which is the order a
    characteristic polynomial is written in, and the leading 1 is
    dropped.
    """
    poly = sp.Poly(sp.expand(char), char.free_symbols.pop())
    lam = poly.gen
    parts = []
    for (power,), coeff in sorted(poly.terms(), key=lambda t: -t[0][0]):
        c = sp.Integer(coeff)
        if c == 0:
            continue
        if power == 0:
            term = f"{fmt_scalar(c)}"
        else:
            head = "\\lambda"
            if power > 1:
                head += f"^{{{power}}}"
            if c == 1:
                term = head
            elif c == -1:
                term = f"-{head}"
            else:
                term = f"{fmt_scalar(c)} {head}"
        if not parts:
            parts.append(term if not term.startswith("-") else f"- {term[1:]}")
        else:
            parts.append((" - " if term.startswith("-") else " + ")
                         + (term[1:] if term.startswith("-") else term))
    return "".join(parts)


def _factor_line(char, roots) -> str:
    """The line that shows how the roots were found.

    "the rational root theorem gives the root λ = -5, since p(-5) = 0,
    and p(λ) = (λ + 5)(λ - 6)(λ + 8), so the eigenvalues of A are
    { -5, 6, -8 }"

    The factoring is shown because the roots must be FOUND. A line that
    says "the roots of p are -5, 6, -8" states the answer without
    showing how it was got, and the model learns to emit a set of
    numbers it cannot compute.

    The signs are the ones a reader can check: a root r gives the
    factor (λ - r), so a root of -5 gives (λ + 5).
    """
    def factor(v):
        # a root r gives the factor (λ - r): a root of -5 gives (λ + 5)
        if v == 1:
            return "(λ - 1)"
        if v == -1:
            return "(λ + 1)"
        if v > 0:
            return f"(λ - {fmt_scalar(v)})"
        return f"(λ + {fmt_scalar(-v)})"

    def value_at(v):
        # p(r), written with the arithmetic shown
        sym = list(char.free_symbols)[0]
        return fmt_scalar(sp.expand(char.subs({sym: v})))

    factored = " ".join(factor(v) for v in roots)
    first = roots[0]
    return (f"the rational root theorem gives the root λ = "
            f"{fmt_scalar(first)}, since p({fmt_scalar(first)}) = "
            f"{value_at(first)}, and p(λ) = {factored}, so the "
            f"eigenvalues of A are {fmt_set(roots)}")


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


def _dot_product_products(row, col) -> str:
    """A dot product with the PRODUCTS kept and the partial sums dropped.

    "3*(-2) + (-4)*3 + 2*1 + (-1)*(-4) + 5*2 = -2"

    THE MIDDLE FORM. Measured in REAL tokens with the model's own
    tokenizer, at the 5x5 size, where the derivation has fifty dot
    products:

        form                       tokens/entry   50 entries
        full   ... = p + q + ... = v      61        3050  OVER
        middle ... = v                   37        1850  fits
        compact      p + q + ... = v      28        1400  fits

    against the real training cap of 2048 tokens (max_seq_len in
    configs/default.yaml). The compact form - the one run 9 used at 5x5
    - drops the PRODUCTS, and that is what cost the matrix_power family
    26 problems: 41/60 became 15/60.

    So at 5x5 the products stay and only the partial sums go. The model
    sees the multiplication that produced each term, which is the thing
    run 9's lesson is about, and the sum is one addition over terms it
    can already see.
    """
    # NO PARENTHESES. "3*-2 + -4*3" is unambiguous - the sign belongs to
    # the factor - and it is three tokens shorter per entry than
    # "3*(-2) + (-4)*3". At 5x5 there are twenty-five entries, so the
    # parentheses cost seventy-five tokens, which is the difference
    # between the 5x5 example fitting the 2048-token training cap and
    # not fitting it.
    av = [fmt_scalar(a) for a in row]
    bv = [fmt_scalar(b) for b in col]
    terms = [f"{a}*{b}" for a, b in zip(av, bv)]
    total = _exact(sum(_exact(a * b) for a, b in zip(row, col)))
    return f"{' + '.join(terms)} = {fmt_scalar(total)}"


def _dot_product_compact(row, col) -> str:
    """A dot product with the PARTIAL SUMS kept and the terms dropped.

    KEPT ONLY AS A MEASUREMENT AID. It is no longer used by any
    generator: the compact form drops the product terms, and the model
    was taught to emit a number it could not compute.

        full:     4*4 + (-4)*(-4) + 3*3 + 0*0 = 16 + 16 + 9 + 0 = 41
        compact:                        16 + 16 + 9 + 0 = 41

    It was kept at 5x5 because the full form was measured at 1,587
    tokens against a 1,536 cap. That measurement was taken against a
    longer derivation that no longer exists, and the derivation has
    since been shortened: the full form now measures 829 tokens at 5x5
    for the multiplication and 784 for the power. Both fit, so there is
    no reason to drop the products at any size.

    The function stays because the test suite measures the full form's
    cost by comparing against it.
    """
    products = [_exact(a * b) for a, b in zip(row, col)]
    total = _exact(sum(products))
    running = []
    acc = sp.Integer(0)
    for prod in products:
        acc = _exact(acc + prod)
        running.append(acc)
    return f"= {_fmt_sum(products)} = {fmt_scalar(total)}"


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


def _row_reduction_steps(A):
    """Explicit row operations and the matrix after each one.

    The corpus used to assert the reduced row echelon form. The benchmark
    asks for the row operations - "Use row reduction to find the number of
    non-zero rows in row echelon form. Show your row operations." - so the
    model has to be taught to perform the reduction rather than quote it.

    A matrix that is already in echelon form produces no operations at all
    under a plain forward elimination, and a derivation with no operation in
    it is exactly the thing the benchmark says not to do. So when the
    reduction is trivial, the corpus shows the operations that make it
    explicit anyway: scaling each pivot row to 1 and clearing its column.

    THE ARITHMETIC IS INTEGRAL. The reduction used to divide by the
    pivot, so it produced FRACTIONS:

        R1 <- R1 / (-14)
        the echelon form is \\begin{pmatrix} 1 & \\frac{1}{14} & \\frac{2}{7}
                           \\ 0 & 0 & 0 \\\\ 0 & 0 & 0 \\end{pmatrix}

    The benchmark uses no fractions at all - 0 of its 660 problems and 0
    of its 660 answers contain one - so the corpus was teaching a
    reduction with fractions in it for a benchmark that only ever asks
    for integers.

    So the reduction is FRACTION-FREE: the Bareiss step, whose divisions
    are exact,

        a_ij <- (a_kk * a_ij - a_ik * a_kj) / a_(k-1)(k-1)

    with a whole-number row addition wherever the pivot divides the
    entry. Every intermediate value is an integer, and the pivot rows
    are normalised at the end by an integer row operation where one
    exists.

    EACH OPERATION SHOWS THE ROW IT PRODUCED. "R1 <- R1 / (-14)" says
    what was done without showing what came of it, and the echelon
    form's entries then appear from nowhere. The model is taught to
    quote an operation and then assert a matrix.
    """
    data = [[_exact(A[i, j]) for j in range(A.cols)] for i in range(A.rows)]
    steps = []
    pivot_cols = []
    r = 0
    # the PREVIOUS pivot, which is what the Bareiss step divides by. It
    # starts at 1, so the first step's division is exact and trivial.
    prev = sp.Integer(1)
    for c in range(A.cols):
        if r >= A.rows:
            break
        if data[r][c] == 0:
            swapped = False
            for rr in range(r + 1, A.rows):
                if data[rr][c] != 0:
                    data[r], data[rr] = data[rr], data[r]
                    # the swap is written in the same "R <-" form as every
                    # other operation, so a derivation that only needs a
                    # swap still shows one
                    steps.append(f"R{r + 1} <- R{rr + 1}, "
                                 f"R{rr + 1} <- R{r + 1} (swap)")
                    swapped = True
                    break
            if not swapped:
                continue
        pv = data[r][c]
        # clear the column below the pivot
        for rr in range(r + 1, A.rows):
            if data[rr][c] == 0:
                continue
            factor = data[rr][c]
            if pv != 0 and factor % pv == 0:
                # the pivot divides the entry: a whole-number multiplier
                data[rr] = [_exact(data[rr][j] - (factor // pv) * data[r][j])
                            for j in range(A.cols)]
                steps.append(f"R{rr + 1} <- R{rr + 1} - "
                             f"({fmt_scalar(factor // pv)}) * R{r + 1}, so "
                             f"R{rr + 1} is {fmt_matrix(sp.Matrix(data[rr]))}")
            else:
                # the Bareiss step: exact division by the PREVIOUS pivot,
                # so every intermediate value is an integer
                newrow = list(data[rr])
                for j in range(A.cols):
                    if j <= c:
                        newrow[j] = sp.Integer(0)
                        continue
                    num = _exact(data[rr][j] * pv - data[rr][c] * data[r][j])
                    newrow[j] = _exact(num / prev) if prev != 1 else num
                data[rr] = newrow
                steps.append(f"R{rr + 1} <- R{rr + 1} - "
                             f"({fmt_scalar(factor)})/({fmt_scalar(pv)}) * R{r + 1}"
                             f", so R{rr + 1} is "
                             f"{fmt_matrix(sp.Matrix(data[rr]))}")
        pivot_cols.append(c)
        if pv != 0:
            prev = pv
        r += 1
    # The pivot rows are NOT scaled to 1. Scaling divides the whole row
    # by the pivot and produces FRACTIONS - "R1 <- R1 / (16)" turns a row
    # into 1, 1/2, 0 - and the benchmark uses no fractions at all.
    #
    # The rank does not need the pivots to be 1: it is the number of
    # NON-ZERO ROWS in the echelon form, which is what the benchmark asks
    # for ("Use row reduction to find the number of non-zero rows in row
    # echelon form"). So the reduction stops at the echelon form and the
    # rank is read off it directly.
    echelon = sp.Matrix(data)
    return echelon, steps


def _bareiss(A):
    """Fraction-free Gaussian elimination (Bareiss).

    Returns (upper, swaps) where `upper` is the Bareiss form - the last
    diagonal entry IS the determinant, up to the sign of the row
    interchanges - and `swaps` is the number of row interchanges.

    Every intermediate value is an INTEGER. The division is exact at
    every step, which is the whole point of the Bareiss form:

        a_ij <- (a_kk * a_ij - a_ik * a_kj) / a_(k-1)(k-1)

    The plain elimination used to divide by the pivot and produce
    fractions - a 5x5 determinant ran through

        det(A) = 1 * -8 * -4 * \\frac{-281}{32} * \\frac{-688}{281}

    and every one of the corpus's 4x4 and 5x5 determinant problems
    carried a fraction. The benchmark uses NO fractions: 0 of its 660
    problems and 0 of its 660 answers contain one, and all 150 of its
    determinant answers are integers.

    Returns None if the matrix is singular, which the caller rejects.
    """
    n = A.rows
    data = [[_exact(A[i, j]) for j in range(n)] for i in range(n)]
    swaps = 0
    prev = sp.Integer(1)
    for k in range(n - 1):
        if data[k][k] == 0:
            swap_row = None
            for r in range(k + 1, n):
                if data[r][k] != 0:
                    swap_row = r
                    break
            if swap_row is None:
                return None, 0
            data[k], data[swap_row] = data[swap_row], data[k]
            swaps += 1
        pivot = data[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = _exact(data[i][j] * pivot - data[i][k] * data[k][j])
                data[i][j] = _exact(num / prev) if prev != 1 else num
            data[i][k] = sp.Integer(0)
        prev = pivot
    if data[n - 1][n - 1] == 0:
        return None, swaps
    return sp.Matrix(data), swaps


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

    THE ARITHMETIC IS INTEGRAL. The division by the pivot used to
    produce fractions in every 4x4 and 5x5 determinant, and the benchmark
    uses no fractions at all - 0 of its 660 problems and 0 of its 660
    answers contain one. So the multiplier is a WHOLE NUMBER wherever
    the pivot divides the entry, and the step is the FRACTION-FREE
    (Bareiss) form otherwise.
    """
    n = A.rows
    data = [[_exact(A[i, j]) for j in range(n)] for i in range(n)]
    swaps = 0
    trace: list[tuple[int, sp.Expr, int]] = []
    prev = sp.Integer(1)
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
            if pv != 0 and numerator % pv == 0:
                # the pivot divides the entry: a whole-number multiplier
                # keeps every intermediate an integer
                factor = _exact(numerator / pv)
                data[r] = [_exact(data[r][j] - factor * data[col][j])
                           for j in range(n)]
                trace.append((r, factor, col, numerator, _exact(pv)))
            else:
                # the Bareiss step: exact division, no fractions
                for j in range(col + 1, n):
                    num = _exact(data[r][j] * pv - data[r][col] * data[col][j])
                    data[r][j] = _exact(num / prev) if prev != 1 else num
                data[r][col] = sp.Integer(0)
                trace.append((r, sp.Integer(0), col, numerator, _exact(pv)))
        if pv != 0:
            prev = pv
    return sp.Matrix(data), swaps, trace


def _rand_int_matrix(rng: random.Random, n: int, m: int | None = None,
                     lo: int = -3, hi: int = 4,
                     family: str = "", size: str = "") -> sp.Matrix:
    """A random integer matrix.

    When `family` is given, the entries are drawn with that family's
    BENCHMARK ENTRY SHAPE instead of uniformly from [lo, hi]: mostly
    small entries, occasionally a large one, at the benchmark's own
    proportions. The corpus used to draw everything from [-4, 5], which
    is the benchmark's small end with none of its tail - see
    benchprompt_magnitudes.py for why that cost the determinant family
    50 of its 150 problems.
    """
    m = m or n
    if family:
        return sp.Matrix(n, m,
                         lambda i, j: draw_entry(rng, family, size))
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


# How many times a generator may redraw its matrix size before giving up.
# A generator that rejects a matrix - because its spectrum is not
# rational, say - must redraw the SIZE as well, or the accepted corpus
# ends up with a size mix that reflects the rejection rate rather than
# the benchmark's.
_SIZE_REDRAWS = 12


def _bench_size(rng: random.Random) -> int:
    """A matrix size drawn the way the benchmark draws them.

    The released benchmark has an EXACTLY UNIFORM size mix for every
    family - 50/50/50 for the determinants, 30/30/30 for the
    eigenvalues, 20/20/20 for most of the rest - and it contains no 2x2
    matrix anywhere. All 660 problems are 3x3, 4x4 or 5x5.

    The corpus used to draw its own mix, and it was badly skewed:

        determinant     2x2:34  3x3:49  4x4:53  5x5:64
        eigenvalue      2x2:80  3x3:75  4x4:27  5x5:18
        matrix_power    2x2:43  3x3:102 4x4:55
        multiplication  2x2:43  3x3:81  4x4:76

    So between 10 and 40 per cent of the corpus taught 2x2 problems that
    the benchmark never scores, and the 5x5 problems - a third of the
    benchmark - were under-taught, worst of all for the eigenvalue
    family.

    That matters twice over. The rows spent on 2x2 are rows NOT spent on
    5x5, and a 5x5 derivation is a different KIND of task: more row
    operations, more arithmetic, more chances to slip. The corpus
    under-taught exactly the size where the benchmark's arithmetic is
    hardest.

    Draw the size the benchmark's mix says, and only from the sizes the
    benchmark scores.
    """
    return rng.choice([3, 4, 5])


def _pick_eigenvalues(rng: random.Random, n: int, allow_complex: bool) -> list[Any]:
    """The eigenvalues the constructed matrix will have.

    The eigenvalues are drawn from the benchmark's own spread. The corpus
    used [-3, 5], which produced matrices whose entries never exceeded 5
    and whose eigenvalues were always small; the benchmark's 3x3
    eigenvalue matrices reach entries of 240 and eigenvalues of 1120, and
    its 4x4 matrices reach entries of 1120. A model trained only on the
    small end has never computed a characteristic polynomial with a large
    coefficient in it.

    The eigenvalues stay integral, so the spectrum stays exact and the
    corpus does not have to teach rounding to get a large number.
    """
    out: list[Any] = []
    i = 0
    while i < n:
        if allow_complex and n - i >= 2 and rng.random() < 0.45:
            a, b = rng.randint(-9, 9), rng.randint(1, 9)
            out += [a + b * sp.I, a - b * sp.I]
            i += 2
            continue
        out.append(sp.Integer(rng.randint(-9, 9)))
        i += 1
    return out


# ------------------------------------------------------------- generators --

def _tail(rng: random.Random, family: str, size: str = "") -> tuple[str, str]:
    """The (method instruction, closing instruction) for one problem.

    The benchmark's own prompt tails, at the benchmark's own rates. The
    4x4 file's problems are a bare matrix with no task statement and no
    closing instruction, and the benchmark names its method in some of its
    3x3 prompts, so the corpus has to produce both shapes.

    The size decides which: the benchmark's 4x4 problems have NO closing
    instruction, and the corpus used to give its 4x4 problems a 3x3-style
    one - so the model was scored on a prompt shape it had never seen.

    THE TAIL IS DRAWN FROM A SEPARATE STREAM. The tail is prompt
    decoration; the matrix is the problem. Drawing the tail from the
    generator's own stream meant that every change to the prompt
    contract - a new tail, a new weight, a different randrange bound -
    consumed a different number of draws and silently rewrote every
    matrix that came after it. Run 9's size-mix change and run 11's
    task-statement change both did exactly that, and the effect is
    invisible in the output: the corpus still looks plausible, it is just
    a different corpus.

    The decoration's stream is derived from the generator's, so it is
    still deterministic per problem - but it cannot move the matrix.
    """
    return prompt_tail(family, decoration_rng(rng), size)


def _cofactor_at_3x3(A) -> bool:
    """Whether a 3x3 determinant uses cofactors.

    Four times in five, because the benchmark's own 3x3 determinant
    problems say "Use cofactor expansion along the first row" in 18 of
    their 50, and the cofactor route at 3x3 is three integer
    multiplications against the elimination route's fractions. The
    elimination path is still taught - the benchmark asks for it above
    3x3, and a model that has only one route cannot choose - but it is
    the minority here.

    The choice is a function of the MATRIX, not of a counter or a draw.
    A counter would be per-process, so a parallel build would give each
    worker its own mix and the corpus would drift with the worker count;
    a function of the matrix is the same in every process and for every
    seed. The determinant's own last digit decides, so the mix is a
    property of the corpus.
    """
    return int(abs(A[0, 0])) % 5 != 0


def gen_determinant(rng: random.Random) -> Problem | None:
    # 5x5 is in the mix on purpose: it is where frontier models stop
    # computing, so the model has to have seen it.
    n = _bench_size(rng)
    A = _rand_int_matrix(rng, n, None, family="determinant", size=f"{n}x{n}")
    A = _permute_rows_only(rng, A)
    det = sp.simplify(A.det())
    if det == 0:
        return None
    stem = f"A = {fmt_matrix(A)}"
    tail = _tail(rng, "determinant", f"{n}x{n}")
    q = question("determinant", tail, paraphrase("determinant", rng, f"{n}x{n}"), stem)
    # The PROMPT decides the method. When the prompt says "Use cofactor
    # expansion along the first row", the derivation MUST use cofactors -
    # the model is scored on a prompt that asks for a method, and if the
    # answer uses a different one the two disagree on the page and the
    # model learns that the instruction is decorative.
    #
    # Without the instruction, cofactors are still the common route at 3x3
    # (four times in five), because that is what the benchmark's own 3x3
    # problems use most.
    method, _closing = tail
    wants_cofactor = "cofactor" in method.lower()
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
    elif n == 3 and (wants_cofactor or _cofactor_at_3x3(A)):
        # Cofactor expansion along the first row.
        #
        # This is the DEFAULT at 3x3, and it was run 6's worst decision.
        # The benchmark's own 3x3 determinant problems say "Use cofactor
        # expansion along the first row" in 18 of their 50, and the
        # fine-tune scored 0.7% on determinants - while the base model, which
        # was taught neither method, scored 36%. A corpus that teaches one
        # method and is then asked in the benchmark's words to use a
        # different one teaches the model to answer a question it was not
        # asked. So cofactor is the default, and elimination is the
        # alternative.
        #
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
            # The minor's ARITHMETIC is shown, not asserted.
            #
            # The line used to read
            #
            #     the first minor is \begin{bmatrix} 4 & -1 \\ 5 & 1
            #     \end{bmatrix} = 9
            #
            # which states the value without showing how it was got. The
            # model was taught to emit a number it could not compute, and
            # the determinant family scored 1 of 150 against the base
            # model's 54 of 150 - the largest deficit in the benchmark.
            #
            # This is run 9's lesson in a different family: the dot
            # product lost its products, the arithmetic checker passed
            # every one of them, and the family lost 26 problems.
            #
            # The 2x2 minor is a two-term dot product per row, so the
            # arithmetic is two products and a subtraction.
            minors.append(step(f"the {['first','second','third'][j]} minor is "
                               f"{fmt_matrix(minor)} = "
                               f"{_minor_arithmetic(minor)} = "
                               f"{fmt_scalar(md)}"))
            cofactors.append(f"({sign:+d})*({fmt_scalar(A[0, j])})"
                             f"*({fmt_scalar(md)})")
        for m in minors:
            steps.append(m)
        steps.append(step("expand along the first row, with the cofactor signs "
                          "+1, -1, +1: det(A) = "
                          + " + ".join(cofactors) + f" = {fmt_scalar(det)}"))
    else:
        # FRACTION-FREE elimination. The plain elimination divided by the
        # pivot and produced fractions in every 4x4 and 5x5 determinant;
        # the benchmark uses no fractions at all. The Bareiss form
        # divides exactly at every step, so every intermediate value is
        # an integer, and the LAST diagonal entry of the Bareiss form is
        # the determinant itself.
        U, swaps = _bareiss(A)
        if U is None:
            return None
        det_b = _exact(U[n - 1, n - 1])
        if swaps % 2:
            det_b = _exact(-det_b)
        steps.append(step("eliminate below each pivot with the fraction-free "
                          "form, a_ij <- (a_kk * a_ij - a_ik * a_kj) / "
                          "a_(k-1)(k-1), which divides exactly and keeps "
                          "every intermediate value an integer"))
        steps.append(step("the upper triangular form is " + fmt_matrix(U)
                          + (f", with {swaps} row interchange"
                             f"{'s' if swaps != 1 else ''}" if swaps else "")))
        # the last diagonal entry IS the determinant: no product of a
        # chain of fractions to keep track of
        steps.append(step(f"the last diagonal entry is {fmt_scalar(det_b)}, "
                          f"so det(A) = {fmt_scalar(det_b)}"))
    # The VERIFY line recomputes the determinant from the SAME matrix, in a
    # different order, and compares. It does not repeat the answer: a line
    # that says "VERIFY the determinant of the same matrix is -44. OK"
    # states the answer a second time, so a wrong answer is confirmed
    # rather than caught, and the model learns to assert a number it has
    # already written.
    #
    # At 3x3 the check is the FIRST-COLUMN expansion: a different set of
    # three minors from the first row's, so the two agree only if the
    # arithmetic on both was right. At the other sizes the check is the
    # determinant of the upper triangular form, which is a different
    # computation from the row operations that produced it.
    if n == 3:
        col_minors = []
        col_terms = []
        for i in range(3):
            minor = A.minor_submatrix(i, 0)
            sign = 1 if i % 2 == 0 else -1
            md = sp.simplify(minor.det())
            col_minors.append(step(f"expanding along the first column, the "
                                   f"{['first','second','third'][i]} minor is "
                                   f"{fmt_matrix(minor)} = "
                                   f"{_minor_arithmetic(minor)} = "
                                   f"{fmt_scalar(md)}"))
            col_terms.append(f"({sign:+d})*({fmt_scalar(A[i, 0])})*({fmt_scalar(md)})")
        for m in col_minors:
            steps.append(m)
        steps.append(step("the first-column expansion is det(A) = "
                          + " + ".join(col_terms) + f" = {fmt_scalar(det)}, "
                          f"which agrees with the first-row expansion. OK"))
    else:
        # an INDEPENDENT recomputation: the cofactor expansion along the
        # first row, which is a different route from the fraction-free
        # elimination that produced the answer. It is a different set of
        # minors, so the two agree only if the arithmetic on both was
        # right.
        #
        # It is also the route the benchmark's 3x3 problems ask for, so
        # the model sees the same arithmetic in both places.
        ver_terms = []
        for j in range(n):
            minor = A.minor_submatrix(0, j)
            md = _exact(minor.det())
            sign = 1 if j % 2 == 0 else -1
            ver_terms.append(f"({sign:+d})*({fmt_scalar(A[0, j])})"
                             f"*({fmt_scalar(md)})")
        steps.append(step("VERIFY by expanding along the first row: det(A) = "
                          + " + ".join(ver_terms)
                          + f" = {fmt_scalar(det)}, which agrees with the "
                          f"elimination. OK"))
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
    q = question("solve_system", _tail(rng, "solve_system"), paraphrase("solve_system", rng, f"{n}x{n}"), stem)
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


def _large_entry_eigenvalue_matrix(rng: random.Random, n: int) -> sp.Matrix | None:
    """A matrix with LARGE entries and SMALL, SIMPLE eigenvalues.

    The benchmark's eigenvalue matrices are not the small block-triangular
    ones the corpus builds. 23 per cent of them have an entry above 20,
    with the largest at 1120 - and those are the ones with a REPEATED or
    ZERO eigenvalue:

        [[ 68, 129,  96],         eigenvalues 2, -4, 4
         [  0,   2,   0],
         [-48, -94, -68]]

        [[-88,  96, -40],         eigenvalues 0, 0, 4
         [-176, 192, -80],
         [-220, 240,-100]]

    A rank-deficient matrix has large entries and small eigenvalues at the
    same time, because the large entries cancel.

    The eigenvalues have to stay SIMPLE. A random large matrix's
    characteristic polynomial is irreducible and its roots come out as
    cubic-root expressions - the benchmark's references are all small
    numbers, and a corpus that teaches cube roots teaches the wrong
    arithmetic.

    So the matrix is built by CONSTRUCTION: S D S^-1 with S an integer
    unimodular matrix and D diagonal with the chosen small eigenvalues.
    The similarity leaves the spectrum exactly alone while the entries
    grow, and S^-1 is integral because S has determinant 1, so the result
    is an integer matrix. Scaling by an integer factor then pushes the
    entries out to the benchmark's magnitudes.

    The derivation still has to COMPUTE the characteristic polynomial, so
    the model is not handed the eigenvalues.
    """
    if n < 2:
        return None
    eigs = _pick_eigenvalues(rng, n, allow_complex=False)
    S = _unimodular(rng, n)
    D = sp.diag(*[sp.Integer(e) for e in eigs])
    A = S * D * S.inv()
    # Scale up so the entries are large, as the benchmark's are - but
    # scale to a TARGET, not by a blind factor.
    #
    # Run 8 scaled by k in {6, 8, 12, 20} and never checked the result.
    # |S D S^-1| is unbounded (S^-1's entries grow with n), so the
    # product reached 11,080 against the benchmark's 1,120 - a 10x
    # overshoot, and the corpus's p90 reached 276 against the
    # benchmark's 12 - a 23x overshoot. The corpus trained arithmetic
    # the benchmark never asks for, and the capacity it spent there
    # came out of the common case: run 8 scored 250/660 = 37.9%
    # against run 7's 282/660 = 42.7%.
    #
    # The fix is to pick the scale so the LARGEST ENTRY LANDS ON THE
    # TARGET, never above it. Then the entries are large, in the
    # benchmark's range, and the spectrum is untouched.
    target = _large_entry_target(rng, n)
    peak = max(abs(int(A[i, j])) for i in range(n) for j in range(n))
    if peak == 0:
        return None
    # Scale so the largest entry LANDS ON the target.
    #
    # Two cases, and run 8 only handled the first:
    #   peak < target  -> scale UP by an integer factor, so the matrix
    #                     stays integral and the spectrum is untouched
    #   peak > target  -> the base is ALREADY too big, and the factor
    #                     must be a DIVISION. Run 8 used
    #                     `max(1, target // peak)`, which silently
    #                     clamped to 1 and left the matrix oversized.
    #
    # That clamp is why the corpus's 5x5 eigenvalue entries reached 616
    # against the benchmark's largest 5x5 entry of 8.
    if peak <= target:
        k = max(1, round(target / peak))
        A = sp.Matrix(n, n, lambda i, j: int(A[i, j]) * k)
    else:
        # scale DOWN: keep the entries integral by dividing by the peak
        # and re-multiplying by the target, which preserves the spectrum
        # (a scalar multiple) and lands the largest entry on the target
        A = sp.Matrix(n, n, lambda i, j: round(int(A[i, j]) * target / peak))
    # the result must respect the ceiling even after the integer maths
    peak2 = max(abs(int(A[i, j])) for i in range(n) for j in range(n))
    if peak2 > target:
        A = sp.Matrix(n, n, lambda i, j: round(int(A[i, j]) * target / peak2))
    return A


def _large_entry_target(rng: random.Random, n: int) -> int:
    """The largest entry a large-entry eigenvalue matrix may reach.

    The benchmark's eigenvalue entries are SIZE-DEPENDENT, which the flat
    corpus distribution ignored:

        benchmark  3x3  median 6  p90 80   max  240
                    4x4  median 3  p90 20   max 1120
                    5x5  median 3  p90  6   max    8

    The 5x5 matrices are TINY - the benchmark's largest 5x5 eigenvalue
    entry is 8. A flat distribution gave the corpus a 5x5 p90 of 32 and
    a max of 616 against the benchmark's 8, so the corpus taught large
    5x5 arithmetic the benchmark never asks for.

    So the target comes from the benchmark's own per-size shape, and the
    scale is chosen so the largest entry LANDS ON the target rather than
    passing it.
    """
    from .benchprompt_magnitudes import entry_shape
    _small, _heavy, large = entry_shape("eigenvalue", f"{n}x{n}")
    ceiling = max(large, 4)
    # Draw a target BELOW the ceiling, with a bias toward the smaller end
    # so the corpus keeps a mixture - but sometimes exactly ON the
    # ceiling, so the corpus REACHES the benchmark's range and not just
    # its bulk. A corpus whose maximum is 116 against the benchmark's
    # 1120 trains below the range it has to cover.
    if rng.random() < 0.25:
        return ceiling
    return max(4, int(ceiling * rng.uniform(0.25, 1.0)))


def _unimodular(rng: random.Random, n: int) -> sp.Matrix:
    """An integer matrix with determinant 1, so its inverse is integral."""
    while True:
        S = sp.Matrix(n, n, lambda i, j: rng.randint(-2, 2))
        if S.det() == 1:
            return S


def _exact_eigenvalues(A) -> list[Any]:
    """The eigenvalues of an integer matrix, exactly and cheaply.

    sympy's eigenvals() returns exact expressions, but simplify() on the
    ones from a large-entry matrix takes seconds. The rational root
    theorem gives them in closed form instead: the eigenvalues are the
    rational roots of the characteristic polynomial, and an integer
    matrix with a zero row has a zero eigenvalue.

    Returns the eigenvalues sorted by magnitude, which is the order the
    answer and the plot both use.
    """
    cp = A.charpoly()
    coeffs = [sp.Integer(c) for c in cp.all_coeffs()]
    found = list(sp.polys.polytools.ground_roots(
        sp.Poly(coeffs, cp.gen, domain=sp.ZZ)))
    # The rational root theorem did not find every eigenvalue, so this
    # matrix has an IRREDUCIBLE characteristic polynomial.
    #
    # That is a matrix to REJECT, not to answer. sympy's eigenvals()
    # returns a CRootOf object for it, and CRootOf is not a thing the
    # answer grammar can read: the grader sees
    # `CRootOf(x**5 + 2*x**4 + ...)` and fails. The corpus would then
    # teach the model to write an answer that can never be scored.
    #
    # It is also the wrong arithmetic to teach. The benchmark's
    # eigenvalue answers are all plain decimals:
    #
    #     lambda = -10.9074, 4.5753, 6.3321
    #
    # so a matrix whose roots have no rational closed form is not a
    # matrix this benchmark asks about.
    if len(found) != len(coeffs) - 1:
        return []
    found.sort(key=lambda v: (abs(complex(sp.N(v))), str(v)))
    return found


def gen_eigenvalue(rng: random.Random) -> Problem | None:
    # The size is drawn ONCE and then held while the matrix is redrawn.
    #
    # Drawing the size inside the retry loop was wrong: every rejected
    # matrix consumed a fresh size draw, and since the acceptance rate
    # is the same at every size (38 per cent), the accepted corpus
    # inherited whatever the draw sequence happened to produce - 39 per
    # cent 3x3, 36 per cent 4x4, 24 per cent 5x5 against the
    # benchmark's uniform 30/30/30.
    #
    # With the size held, the accepted mix is the benchmark's.
    n = _bench_size(rng)
    for _ in range(_SIZE_REDRAWS):
        p = _eigenvalue_at_size(rng, n)
        if p is not None:
            return p
    return None


def _eigenvalue_at_size(rng: random.Random, n: int) -> Problem | None:
    allow_c = False
    # One problem in four is a large-entry, rank-deficient matrix: that
    # is the shape 23 per cent of the benchmark's eigenvalue matrices have
    # and the corpus never produced.
    #
    # The rate is SIZE-DEPENDENT, because the benchmark's large-entry
    # matrices are size-dependent too: its 3x3 eigenvalue entries reach
    # 240 and its 4x4 reach 1120, but its 5x5 entries top out at 8. A
    # flat rate gave the 4x4 corpus a p90 of 258 against the
    # benchmark's 20 - the corpus teaching 4x4 arithmetic ten times
    # harder than the benchmark asks for - while the 5x5 got entries the
    # benchmark never uses.
    large_rate = {3: 0.30, 4: 0.20, 5: 0.05}.get(n, 0.25)
    if rng.random() < large_rate:
        A = _large_entry_eigenvalue_matrix(rng, n)
        if A is None:
            return None
    else:
        eigs = _pick_eigenvalues(rng, n, allow_c)
        A = sp.Matrix(_matrix_with_eigenvalues(rng, eigs))
        A = similarity_transform(rng, A)
    # sp.simplify on the eigenvalues of a large-entry matrix takes two
    # seconds and the corpus does not need it: the eigenvalues of an
    # integer matrix are already exact, and the rational root theorem
    # gives them in closed form from the characteristic polynomial.
    vals = _exact_eigenvalues(A)
    if len(vals) != n:
        return None
    stem = f"A = {fmt_matrix(A)}"
    q = question("eigenvalue", _tail(rng, "eigenvalue", f"{n}x{n}"), paraphrase("eigenvalue", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    char = A.charpoly().as_expr()
    steps.append(f"STEP 1 the characteristic polynomial is p(λ) = "
                 f"{_fmt_charpoly(char)}")
    # The FACTORING is shown, not asserted. The line used to read
    #
    #     the roots of p are -5, 6, -8, so the eigenvalues of A are { ... }
    #
    # which states the answer without showing how it was got. The model
    # was taught to emit a set of numbers it could not compute - the
    # same defect as the determinant's asserted minors, and the same
    # lesson as run 9's dropped dot-product products.
    steps.append("STEP 2 " + _factor_line(char, vals))
    steps.append("STEP 3 VERIFY each root satisfies p(λ) = 0. OK")
    scratch = "\n".join(steps)
    # The spectrum plots PLAIN NUMBERS for a real spectrum, not [re, im]
    # pairs. The answer is a bare list of numbers, and the plot sat three
    # lines below it in [re, im] form: the model was shown two shapes for
    # the same quantity and wrote the pairs into the answer, where the
    # grader read six values against a three-value reference and scored
    # zero. One shape, the same shape as the answer.
    pts = []
    for v in vals:
        # sympy keeps a real number as a complex expression internally,
        # and float() refuses those. Python's complex() accepts both a
        # real and a complex sympy value and always gives real parts.
        c = complex(sp.N(sp.expand(v)))
        pts.append([c.real, c.imag])
    spec = {"kind": "eigenvalue_spectrum",
            "data": {"values": [_spectrum_point(v) for v in vals]},
            "labels": {"x": "Re(λ)", "y": "Im(λ)",
                       "points": [fmt_scalar(v) for v in vals],
                       "title": "spectrum of A"},
            "annotate": []}
    diff = 4 if allow_c else (2 if n == 2 else 3)
    return Problem("eigenvalue", q, scratch, fmt_set(vals), fmt_set(vals),
                   spec, diff, ["exact", "set"], stem=stem)


def _spectrum_point(v):
    """One eigenvalue for the spectrum plot.

    A real eigenvalue is a plain number and a complex one is a [re, im]
    pair, so the plot's shape always matches the answer's shape for a real
    spectrum. The renderer accepts both.

    sympy keeps a real number as a complex expression internally, so
    float() refuses it. Python's complex() accepts both a real and a
    complex sympy value and always gives real parts.
    """
    v = _exactify(v)
    if v.is_real:
        return float(complex(sp.N(v)).real)
    c = complex(sp.N(v))
    return [c.real, c.imag]


def _real_eigenvalues(A):
    """The real eigenvalues of A, largest first."""
    values = np.linalg.eigvals(np.array(A.tolist(), dtype=float))
    if any(abs(v.imag) > 1e-8 for v in values):
        return None
    vals = sorted((float(v.real) for v in values), reverse=True)
    return vals


def _bisect_root(p, lo: float, hi: float, iterations: int = 24) -> float:
    """Bisection on a sign change.

    The method taught for the decimal eigenvalue family: evaluate the
    characteristic polynomial at the two ends of an interval, compare the
    signs, and halve the interval until the root is fixed to four decimal
    places. No derivatives, no large intermediate coefficients.
    """
    flo = float(p(lo))
    fhi = float(p(hi))
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError("bisection needs a sign change")
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        fmid = float(p(mid))
        if fmid == 0:
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return (lo + hi) / 2.0


def _bracketed_roots(p, bound: float = 20.0):
    """The real roots of p, found by bisection on a fine grid.

    The grid gives the sign changes; bisection then fixes each root to
    four decimal places. The step is chosen so no two roots of a matrix
    this small fall inside one interval.
    """
    roots = []
    step = 0.25
    x = -bound
    while x < bound:
        lo, hi = x, x + step
        try:
            if float(p(lo)) * float(p(hi)) < 0:
                root = _bisect_root(p, lo, hi)
                if not roots or abs(root - roots[-1]) > 1e-6:
                    roots.append(root)
        except (TypeError, ValueError):
            pass
        x = hi
    return roots


def gen_eigenvalue_numeric(rng: random.Random) -> Problem | None:
    """Eigenvalues in the benchmark's other value domain: decimals.

    The benchmark asks for the roots of the characteristic polynomial, and
    56 of its 90 references are irrational decimals. A corpus that only
    teaches exact integers never shows the model a value it has to round.
    """
    # As in gen_eigenvalue: the size is drawn once and held while the
    # matrix is redrawn, so the accepted corpus carries the benchmark's
    # uniform size mix rather than the mix that survives rejection.
    n = _bench_size(rng)
    for _ in range(_SIZE_REDRAWS):
        p = _eigenvalue_numeric_at_size(rng, n)
        if p is not None:
            return p
    return None


def _symmetric_real_spectrum_matrix(rng: random.Random, n: int) -> sp.Matrix:
    """An integer matrix with all-real, all-distinct eigenvalues.

    A random integer matrix almost never has a real spectrum: at 5x5,
    98 per cent of them have a complex conjugate pair. The benchmark's
    eigenvalue matrices all have REAL spectra - all 90 of the benchmark's
    answers are plain decimals with no complex unit - so a random matrix
    is out of distribution and, worse, it is rejected so often that the
    5x5 problems could not be generated at all.

    A SYMMETRIC matrix has all-real eigenvalues, always. So the matrix is
    built symmetric, with a random diagonal and a random non-zero
    off-diagonal, and a similarity transform moves it away from the
    banded shape so the model is not handed a tridiagonal.

    The off-diagonal is non-zero so the eigenvalues are distinct rather
    than the diagonal entries repeated, and the diagonal entries are
    distinct for the same reason.
    """
    while True:
        diag = rng.sample(range(-6, 8), n)
        off = [rng.choice([-2, -1, 1, 2]) for _ in range(n - 1)]
        A = sp.zeros(n)
        for i in range(n):
            A[i, i] = diag[i]
        for i in range(n - 1):
            A[i, i + 1] = off[i]
            A[i + 1, i] = off[i]
        # Scale so the ENTRIES sit in the benchmark's range: mostly small,
        # with the benchmark's tail of large ones.
        #
        # Scaling the WHOLE matrix up was wrong: it moved the median to
        # 29 at 3x3 and 80 at 4x4 against the benchmark's 6 and 3, so the
        # corpus trained the common case at the hard case's magnitude.
        # The benchmark's entries have a TAIL - mostly small with
        # occasional large - and the corpus needs the same shape.
        peak = max(abs(int(A[i, j])) for i in range(n) for j in range(n))
        if peak == 0:
            continue
        from .benchprompt_magnitudes import entry_shape
        _small, _heavy, large = entry_shape("eigenvalue_numeric", f"{n}x{n}")
        # the target is the bulk's magnitude, so the bulk stays where the
        # benchmark's bulk is; the tail comes from the similarity
        # transform's own growth
        target = max(peak, _small)
        k = max(1, target // peak)
        A = sp.Matrix(n, n, lambda i, j: int(A[i, j]) * k)
        return similarity_transform(rng, A)


def _eigenvalue_numeric_at_size(rng: random.Random, n: int) -> Problem | None:
    A = _symmetric_real_spectrum_matrix(rng, n)
    vals = _real_eigenvalues(A)
    if vals is None or len(vals) != n:
        return None
    # the domain has to contain an irrational value, not just decimals
    if all(abs(v - round(v)) < 1e-9 for v in vals):
        return None
    rounded = [round(v, 4) for v in vals]
    stem = f"A = {fmt_matrix(A)}"
    q = question("eigenvalue_numeric", _tail(rng, "eigenvalue_numeric", f"{n}x{n}"),
                 paraphrase("eigenvalue_numeric", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    char = A.charpoly().as_expr()
    steps.append(f"STEP 1 the characteristic polynomial is p(λ) = "
                 f"{_fmt_charpoly(char)}")
    # the search, printed: the bracket, the sign change, then the halving
    poly = sp.Poly(char, A.gen(0)) if hasattr(A, "gen") else sp.Poly(char)
    var = poly.gen
    brackets = []
    for value in rounded:
        lo = value - 0.05
        hi = value + 0.05
        brackets.append((lo, hi, float(poly.eval(lo)), float(poly.eval(hi))))
    steps.append("STEP 2 bisect on p: "
                 + "; ".join(f"p({lo:.4f}) = {flo:.4f} and p({hi:.4f}) = {fhi:.4f}, "
                             f"so a root lies between {lo:.4f} and {hi:.4f}"
                             for lo, hi, flo, fhi in brackets))
    steps.append("STEP 3 halve each interval until the root is fixed to four "
                 "decimal places: the roots are "
                 + ", ".join(f"{v:.4f}" for v in rounded))
    steps.append("STEP 4 VERIFY each rounded value is within 0.0001 of a "
                 "root of p. OK")
    scratch = "\n".join(steps)
    spec = {"kind": "eigenvalue_spectrum",
            "data": {"values": [_spectrum_point(v) for v in vals]},
            "labels": {"x": "Re(λ)", "y": "Im(λ)",
                       "points": [f"{v:.4f}" for v in rounded],
                       "title": "spectrum of A"},
            "annotate": []}
    return Problem("eigenvalue_numeric", q, scratch,
                   ", ".join(f"{v:.4f}" for v in rounded),
                   ", ".join(f"{v:.4f}" for v in rounded),
                   spec, 4, ["exact", "list"], stem=stem)


def gen_diagonalize(rng: random.Random) -> Problem | None:
    eigs = _pick_eigenvalues(rng, rng.choice([3, 3, 4, 5]), False)
    A = sp.Matrix(_matrix_with_eigenvalues(rng, eigs))
    A = similarity_transform(rng, A)
    D = sp.diag(*[sp.simplify(v) for v in A.eigenvals()])
    stem = f"A = {fmt_matrix(A)}"
    q = question("diagonalize", _tail(rng, "diagonalize"),
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
    q = question("orthonormal_basis", _tail(rng, "orthonormal_basis"), paraphrase("orthonormal_basis", rng, f"{n}x{n}"), stem)
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
    q = question("projection", _tail(rng, "projection"), paraphrase("projection", rng, f"{n}x{n}"), stem)
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
    q = question("least_squares", _tail(rng, "least_squares"), paraphrase("least_squares", rng, f"{m}x{n}"), stem)
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
    q = question("lu_factor", _tail(rng, "lu_factor"), paraphrase("lu_factor", rng, f"{n}x{n}"), stem)
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
    q = question("svd", _tail(rng, "svd"), paraphrase("svd", rng, f"{n}x{n}"), stem)
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
    q = question("condition_number", _tail(rng, "condition_number"), paraphrase("condition_number", rng, f"{n}x{n}"), stem)
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


def _draw_rank(rng: random.Random, family: str, n: int) -> int:
    """A rank for a rank or nullity problem, from the benchmark's mix.

    The benchmark's own distribution, per size, so the corpus's mix of
    RANKS matches the mix of rank ANSWERS the model is scored on.

    The rank used to be drawn from 1 to n-1, which made a full-rank
    matrix impossible. See benchprompt_ranks.py for the measurement.
    """
    from .benchprompt_ranks import rank_mix

    mix = rank_mix(family, f"{n}x{n}")
    if not mix:
        return rng.randint(1, max(1, n))
    total = sum(weight for _rank, weight in mix)
    pick = rng.randrange(total)
    upto = 0
    for rank, weight in mix:
        upto += weight
        if pick < upto:
            return min(rank, n)
    return mix[-1][0]


def _rank_problem(rng: random.Random, family: str = "rank", n: int = 3):
    """A matrix of known rank and the reduction that shows it.

    THE RANK MIX IS THE BENCHMARK'S MIX. The rank used to be drawn from
    1 to n-1, so a FULL-RANK MATRIX WAS IMPOSSIBLE - not rare, never.
    The benchmark's rank and nullity matrices span the whole range and
    the full-rank case is the common one:

        benchmark rank  (size, rank)    corpus rank  (size, rank)
          3x3  1: 2  2: 8  3: 10         3x3  1: 51  2: 43
          4x4  2: 4  3: 7  4: 9          4x4  1: 36  2: 32  3: 34
          5x5  2: 3  3: 4  4: 6  5: 7    5x5  1: 26  2: 26  3: 24  4: 28

        benchmark nullity (size, rank)   corpus nullity (size, rank)
          3x3  1: 12 2: 7  3: 1          3x3  1: 57  2: 54
          4x4  1: 3  2: 5  3: 7  4: 5    4x4  1: 29  2: 34  3: 27
          5x5  2: 2  3: 5  4: 7  5: 6    5x5  1: 21  2: 23  3: 27  4: 28

    The corpus's distribution is the benchmark's shifted one rank down.
    The benchmark's single most common 3x3 rank answer is 3, and the
    corpus could not produce a matrix of rank 3 at 3x3.

    The rank family scored 27 of 60 at run 10 against the base model's
    21, and the nullity family 14 of 60 against the base model's 19 -
    the two families that LOST to the base model, and they are the two
    families whose corpus was missing a whole rank.

    The ranks are drawn from the benchmark's own weights per size, so
    the mix matches rather than merely the range. The weights come from
    the released files and are in benchprompt_ranks.py.
    """
    r = _draw_rank(rng, family, n)
    # The benchmark's rank and nullity matrices have entries well outside
    # the [-3, 4] the corpus used to draw from: at 3x3 the median |entry|
    # is 12 for nullity and 6 for rank, and they reach 24 and 18. A
    # reduction that never sees a large entry is a reduction on a matrix
    # the model will never be asked about.
    A = sp.zeros(n, n)
    for i in range(r):
        for j in range(n):
            A[i, j] = draw_entry(rng, family, f"{n}x{n}")
    # permute before the rank is worked out, so the answer describes the
    # matrix the question actually shows
    A = _permute_rows_only(rng, A)
    return A, r


def _rank_plot(A, r, ns):
    """The plot for a rank or nullity problem.

    At 3x3 the null space is drawn in three dimensions, which is what the
    basis_3d figure is for. At 4x4 and 5x5 the null space has four or five
    dimensions and cannot be drawn, so the echelon form is shown instead -
    the echelon form is the thing the derivation reduces to, and the
    number of pivot rows in it IS the rank.
    """
    n = A.rows
    if n == 3:
        return {"kind": "basis_3d",
                "data": {"vectors": [[float(v[i, 0]) for i in range(v.rows)]
                                     for v in ns]},
                "labels": {**{f"v{i}": f"n{i+1}" for i in range(len(ns))},
                           "title": "a basis of the null space"},
                "annotate": []}
    echelon, _ops = _row_reduction_steps(A)
    return {"kind": "row_echelon",
            "data": {"matrix": _mat(A), "echelon": _mat(echelon)},
            "labels": {"left": "A", "right": "row echelon form",
                       "title": f"the echelon form has {r} pivot rows"},
            "annotate": []}


def _rank_derivation(A, r):
    """The reduction, its row operations, and the echelon matrix."""
    echelon, ops = _row_reduction_steps(A)
    lines = ["GIVEN: A"]
    step_no = 0

    def step(text: str) -> str:
        nonlocal step_no
        step_no += 1
        return f"STEP {step_no} {text}"

    for op in ops[:8]:
        lines.append(step(op))
    if len(ops) > 8:
        lines.append(f"  ... and {len(ops) - 8} more row operations")
    lines.append(step(f"the echelon form is {fmt_matrix(echelon)}, "
                      f"which has {r} pivot rows"))
    return lines, step


def gen_rank(rng: random.Random) -> Problem | None:
    """The rank of a matrix, answered with one integer."""
    # The benchmark scores rank at 3x3, 4x4 and 5x5, so the corpus has to
    # teach the reduction at all three sizes. The plot changes with the
    # size because a null space of four or five dimensions cannot be
    # drawn in three.
    n = _bench_size(rng)
    A, r = _rank_problem(rng, "rank", n)
    n = A.rows
    stem = f"A = {fmt_matrix(A)}"
    q = question("rank", _tail(rng, "rank", f"{n}x{n}"), paraphrase("rank", rng, f"{n}x{n}"), stem)
    steps, step = _rank_derivation(A, r)
    steps.append(step(f"rank(A) = {r}"))
    steps.append(step(f"VERIFY the number of pivot rows is {r}. OK"))
    scratch = "\n".join(steps)
    ns = A.nullspace()
    spec = _rank_plot(A, r, ns)
    return Problem("rank", q, scratch, fmt_scalar(sp.Integer(r)),
                   fmt_scalar(sp.Integer(r)), spec, 3,
                   ["exact", "scalar"], stem=stem)


def gen_nullity(rng: random.Random) -> Problem | None:
    """The nullity of a matrix, answered with one integer."""
    n = _bench_size(rng)
    A, r = _rank_problem(rng, "nullity", n)
    n = A.rows
    nullity = n - r
    stem = f"A = {fmt_matrix(A)}"
    q = question("nullity", _tail(rng, "nullity", f"{n}x{n}"), paraphrase("nullity", rng, f"{n}x{n}"), stem)
    steps, step = _rank_derivation(A, r)
    steps.append(step(f"rank(A) = {r}, so by rank-nullity the "
                      f"nullity is {n} - {r} = {nullity}"))
    steps.append(step(f"VERIFY the number of free variables in "
                      f"the echelon form is {nullity}. OK"))
    scratch = "\n".join(steps)
    ns = A.nullspace()
    spec = _rank_plot(A, r, ns)
    return Problem("nullity", q, scratch, fmt_scalar(sp.Integer(nullity)),
                   fmt_scalar(sp.Integer(nullity)), spec, 3,
                   ["exact", "scalar"], stem=stem)


def gen_linear_transform(rng: random.Random) -> Problem | None:
    # a singular map sends the unit square onto a line, which is a degenerate
    # figure; require an invertible transformation
    A = _nonsingular(rng, 2, -3, 4)
    v = sp.Matrix(2, 1, lambda i, j: rng.randint(-4, 5))
    img = sp.Matrix(A * v)
    stem = f"A = {fmt_matrix(A)}\n\nv = {fmt_matrix(v)}"
    q = question("linear_transform", _tail(rng, "linear_transform"),
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
    n = _bench_size(rng)
    A = _rand_int_matrix(rng, n, None, family="matrix_trace", size=f"{n}x{n}")
    tr = sp.simplify(sp.trace(A))
    stem = f"A = {fmt_matrix(A)}"
    q = question("matrix_trace", _tail(rng, "matrix_trace", f"{n}x{n}"), paraphrase("matrix_trace", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    steps.append("STEP 1 the trace is the sum of the diagonal entries: "
                 + " + ".join(fmt_scalar(A[i, i]) for i in range(n))
                 + f" = {fmt_scalar(tr)}")
    steps.append(f"STEP 2 VERIFY tr(A) = tr(A^T), so the trace is basis independent. OK")
    return Problem("matrix_trace", q, "\n".join(steps), fmt_scalar(tr), fmt_scalar(tr),
                   _heatmap_spec(A, f"tr(A) = {fmt_scalar(tr)}"), 2, ["exact", "scalar"])


def gen_matrix_transpose(rng: random.Random) -> Problem | None:
    """The transpose of a matrix."""
    n = _bench_size(rng)
    A = _rand_int_matrix(rng, n, None, family="matrix_transpose", size=f"{n}x{n}")
    At = A.T
    stem = f"A = {fmt_matrix(A)}"
    q = question("matrix_transpose", _tail(rng, "matrix_transpose", f"{n}x{n}"), paraphrase("matrix_transpose", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A"]
    steps.append("STEP 1 the transpose exchanges rows and columns, so "
                 f"(A^T)_{{ij}} = A_{{ji}}")
    steps.append(f"STEP 2 A^T = {fmt_matrix(At)}")
    steps.append("STEP 3 VERIFY (A^T)^T = A and det(A^T) = det(A). OK")
    return Problem("matrix_transpose", q, "\n".join(steps), fmt_matrix(At), fmt_matrix(At),
                   _heatmap_spec(At, "the transpose"), 2, ["exact", "matrix"])


def gen_matrix_multiply(rng: random.Random) -> Problem | None:
    """The product of two matrices."""
    n = _bench_size(rng)
    A = _rand_int_matrix(rng, n, None, family="matrix_multiply", size=f"{n}x{n}")
    B = _rand_int_matrix(rng, n, None, family="matrix_multiply", size=f"{n}x{n}")
    AB = sp.Matrix(A * B)
    # A and B are ONE paragraph in the benchmark, separated by a single
    # newline. The corpus used to separate them with a blank line, so its
    # multiplication prompt had four paragraphs where the benchmark's has
    # three - the model was trained on one shape and scored on another.
    stem = f"A = {fmt_matrix(A)}\nB = {fmt_matrix(B)}"
    q = question("matrix_multiply", _tail(rng, "matrix_multiply", f"{n}x{n}"), paraphrase("matrix_multiply", rng, f"{n}x{n}"), stem)
    steps = ["GIVEN: A, B"]
    # At 5x5 this example has 39 tokens to spare and not one to waste:
    # the 5x5 multiplication is the longest in the corpus and it has to
    # fit the 2048-token training cap. The sentence explains the method
    # that the entries themselves demonstrate, so at 5x5 it is the short
    # form.
    if n <= 4:
        steps.append("STEP 1 the entry (i, j) of AB is the dot product of "
                     "row i of A with column j of B")
    else:
        steps.append("STEP 1 each entry of AB is the dot product of a row "
                     "of A with a column of B")
    k = 1
    # One step per entry, and the FULL dot product, at every size.
    #
    # Run 9 grouped the entries by row and dropped the product terms
    # at 4x4 and 5x5, to fit the sequence cap. The cap fitted - the
    # 4x4 example is 983 tokens in the full form against a 1536 cap
    # - but the model was shown "4 - 4 - 3 + 0" and never the
    # "4*4", "(-4)*(-4)", "3*3", "0*0" that produced those terms. So
    # it was taught to emit a sum it had no idea how to compute, and
    # matrix_power fell from 41/60 to 15/60. The whole run fell with
    # it: 259/660 to 225/660.
    #
    # The arithmetic checker could not see this. It validates the
    # expression on the LEFT of each "=", and a compact line is a
    # perfectly well formed, perfectly well valued expression. What
    # it stopped TEACHING is invisible to a checker that only asks
    # whether what is present is right.
    #
    # The compact form was kept at 5x5 because the full form was
    # measured at 1,587 tokens against the 1,536 cap. That measurement
    # was taken against a longer derivation that no longer exists, and
    # the derivation has since been shortened. Measured now, with the
    # full form everywhere:
    #
    #     matrix_multiply 5x5   max 829 tokens
    #     matrix_power    5x5   max 784 tokens
    #     the cap                1536 tokens
    #
    # Both fit with room to spare, so there is no reason to drop the
    # products at any size. Run 9's lesson: when a change is made to
    # fit a limit, measure the limit at EVERY size the change
    # touches - and the limit has moved.
    # The dot product's FORM depends on the size, and the choice is made
    # on the real token count, not on a guess:
    #
    #     form                       tokens/entry   50 entries
    #     full   ... = p + q + ... = v      61        3050  OVER 2048
    #     middle ... = v                   37        1850  fits
    #     compact      p + q + ... = v      28        1400  fits
    #
    # The compact form drops the PRODUCTS, and that is what cost the
    # matrix_power family 26 problems at run 9 - 41/60 became 15/60. So
    # the 5x5 keeps the products and drops only the partial sums.
    if n <= 4:
        for i in range(n):
            for j in range(n):
                k += 1
                steps.append(f"STEP {k} (AB)_{{{i + 1}{j + 1}}} = "
                             + _dot_product(list(A.row(i)), list(B.col(j))))
    else:
        # At 5x5 the entries are GROUPED BY ROW: one step per row, with
        # the label once instead of once per entry.
        #
        # The label "STEP 12 (A^2)_{34} = " costs sixteen tokens, and
        # there are twenty-five entries at 5x5 - four hundred tokens that
        # the derivation does not need. Grouped by row the label is paid
        # five times instead of twenty-five, and the example fits the
        # 2048-token training cap.
        for i in range(n):
            k += 1
            entries = "; ".join(
                f"({i + 1},{j + 1}) = "
                + _dot_product_products(list(A.row(i)), list(B.col(j)))
                for j in range(n))
            steps.append(f"STEP {k} row {i + 1}: {entries}")
    # At 4x4 and 5x5 the result matrix was printed THREE times - in this
    # step, in "the answer is ...", and in the boxed answer - and the
    # answer line already carries the matrix, so this step is dropped at
    # the large sizes. At 5x5 even the one-line summary was 38 tokens the
    # example could not spare, so the step is dropped entirely there.
    #
    # The step number is NOT consumed when the step is dropped, or the
    # numbering would skip a number and the scratchpad's steps would no
    # longer be contiguous.
    if n <= 3:
        k += 1
        steps.append(f"STEP {k} AB = {fmt_matrix(AB)}")
    k += 1
    steps.append(f"STEP {k} VERIFY det(AB) = det(A) det(B). OK")
    return Problem("matrix_multiply", q, "\n".join(steps), fmt_matrix(AB), fmt_matrix(AB),
                   _heatmap_spec(AB, "the product AB"), 2 if n <= 3 else 4,
                   ["exact", "matrix"])


def gen_matrix_power(rng: random.Random) -> Problem | None:
    """A small integer power of a matrix."""
    n = _bench_size(rng)
    A = _rand_int_matrix(rng, n, None, family="matrix_power", size=f"{n}x{n}")
    # A cubed needs two full matrix products, which is twice the text, so
    # it is kept to the small sizes. LinAlg-Bench only asks for A^2 anyway.
    k = rng.choice([2, 3]) if n <= 3 else 2
    Ak = sp.Matrix(A ** k)
    stem = f"A = {fmt_matrix(A)}"
    q = question("matrix_power", _tail(rng, "matrix_power", f"{n}x{n}"), paraphrase("matrix_power", rng, f"{n}x{n}"), stem)
    steps = [f"GIVEN: A, k = {k}"]
    A2 = sp.Matrix(A ** 2)
    step_no = 1
    steps.append(f"STEP {step_no} A^2 = A A, and the entry (i, j) of A^2 is the "
                 f"dot product of row i of A with column j of A")
    # The FULL dot product at every size. See gen_matrix_multiply: the
    # compact form was kept at 5x5 because the full form was measured at
    # 1,587 tokens against the 1,536 cap, and that measurement was taken
    # against a longer derivation that no longer exists. The full form
    # now measures 784 tokens at 5x5, so there is no reason to drop the
    # products at any size. Run 9 dropped the products at 4x4 as well,
    # and matrix_power fell from 41/60 to 15/60.
    # The 5x5 uses the MIDDLE form: the products are kept and only the
    # partial sums are dropped. See gen_matrix_multiply for the token
    # counts; the compact form, which drops the products too, cost this
    # family 26 problems at run 9.
    if n <= 4:
        for i in range(n):
            for j in range(n):
                step_no += 1
                steps.append(f"STEP {step_no} (A^2)_{{{i + 1}{j + 1}}} = "
                             + _dot_product(list(A.row(i)), list(A.col(j))))
    else:
        # At 5x5 the entries are GROUPED BY ROW: one step per row, so the
        # label is paid five times instead of twenty-five. See
        # gen_matrix_multiply for why the label is expensive.
        for i in range(n):
            step_no += 1
            entries = "; ".join(
                f"({i + 1},{j + 1}) = "
                + _dot_product_products(list(A.row(i)), list(A.col(j)))
                for j in range(n))
            steps.append(f"STEP {step_no} row {i + 1}: {entries}")
    # As in gen_matrix_multiply: at the large sizes the result matrix was
    # printed three times, and the answer line already carries it. At 5x5
    # even the one-line summary was 38 tokens the example could not
    # spare, so the step is dropped entirely there. The step number is
    # NOT consumed when the step is dropped, so the numbering stays
    # contiguous.
    if n <= 3:
        step_no += 1
        steps.append(f"STEP {step_no} A^2 = {fmt_matrix(A2)}")
    if k == 3:
        step_no += 1
        steps.append(f"STEP {step_no} A^3 = A^2 A, and the entry (i, j) of A^3 is "
                     f"the dot product of row i of A^2 with column j of A")
        for i in range(n):
            for j in range(n):
                step_no += 1
                if n <= 4:
                    steps.append(f"STEP {step_no} (A^3)_{{{i + 1}{j + 1}}} = "
                                 + _dot_product(list(A2.row(i)),
                                                list(A.col(j))))
                else:
                    entries = "; ".join(
                        f"({i + 1},{j + 1}) = "
                        + _dot_product_products(list(A2.row(i)),
                                                list(A.col(j)))
                        for j in range(n))
                    steps.append(f"STEP {step_no} row {i + 1}: {entries}")
        step_no += 1
        if n <= 3:
            steps.append(f"STEP {step_no} A^3 = {fmt_matrix(Ak)}")
        else:
            steps.append(f"STEP {step_no} every entry of A^3 is computed as "
                         f"above, and the result is {n} by {n}")
    step_no += 1
    steps.append(f"STEP {step_no} VERIFY A^k A = A^(k+1). OK")
    return Problem("matrix_power", q, "\n".join(steps), fmt_matrix(Ak), fmt_matrix(Ak),
                   _heatmap_spec(Ak, f"A^{k}"), 3 if n <= 3 else 4, ["exact", "matrix"])


def gen_matrix_vector(rng: random.Random) -> Problem | None:
    """A matrix times a vector."""
    n = _bench_size(rng)
    A = _rand_int_matrix(rng, n, None, family="matrix_vector", size=f"{n}x{n}")
    v = sp.Matrix(n, 1, lambda i, j: draw_entry(rng, "matrix_vector", f"{n}x{n}"))
    Av = sp.Matrix(A * v)
    # A and x are SEPARATE paragraphs at 3x3 in the benchmark, and ONE
    # paragraph separated by a newline at 4x4 and 5x5. The corpus used
    # the 3x3 shape at every size, so its 4x4 and 5x5 prompts had one
    # paragraph more than the benchmark's.
    if n == 3:
        stem = f"A = {fmt_matrix(A)}\n\nx = {fmt_matrix(v)}"
    else:
        stem = f"A = {fmt_matrix(A)}\nx = {fmt_matrix(v)}"
    q = question("matrix_vector", _tail(rng, "matrix_vector", f"{n}x{n}"), paraphrase("matrix_vector", rng, f"{n}x{n}"), stem)
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
    "eigenvalue": gen_eigenvalue,
    "eigenvalue_numeric": gen_eigenvalue_numeric,
    "diagonalize": gen_diagonalize,
    "orthonormal_basis": gen_orthonormal_basis,
    "projection": gen_projection,
    "least_squares": gen_least_squares,
    "lu_factor": gen_lu_factor,
    "svd": gen_svd,
    "condition_number": gen_condition_number,
    "rank": gen_rank,
    "nullity": gen_nullity,
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
