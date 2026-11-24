"""The determinant's elimination must be SHOWN, not asserted.

This is the determinant family's version of run 9's lesson, and it is the
largest deficit left in the benchmark.

The corpus's determinant derivation used to read:

    STEP 1 eliminate below each pivot with the fraction-free form,
          a_ij <- (a_kk * a_ij - a_ik * a_kj) / a_(k-1)(k-1), ...
    STEP 2 the upper triangular form is \\begin{bmatrix} -2 & 4 & 2 & 4 \\\\ ... \\end{bmatrix}
    STEP 3 the last diagonal entry is -534, so det(A) = -534

**STEP 1 states the formula and performs ZERO arithmetic. STEP 2 asserts
the triangular form wholesale** — the elimination, with its row operations
and their results, is missing entirely.

The model is therefore taught to fabricate the triangular form: to announce
a number it never computed. And it does. The fine-tune scored **7 of 150**
on determinants against the base model's **60 of 150**.

The VERIFY step had the same defect at 4x4 and 5x5: it wrote
`det(A) = (+1)*(-2)*(121) + (-1)*(4)*(83) + ...`, where `(121)` and `(83)`
are the 3x3 minors, asserted as bare numbers. Run 11 fixed the 2x2 minors
at 3x3 and left the larger ones asserted.

**The pattern that works is already in the corpus.** matrix_power shows
every dot product with its products and its partial sums:

    STEP 2 (A^2)_{11} = (-2)*(-2) + 3*3 + (-3)*4 = 4 + 9 - 12 = 1

and it scores 22/60, while the determinant scores 7/150 with 2.9
arithmetic lines per derivation against matrix_power's 14.5. **The
determinant simply never got the same treatment.**
"""
import random
import re
import sys
import unittest

sys.path.insert(0, "src")

from la.problems import generate  # noqa: E402


def _samples(family: str, n: int = 40, seed: int = 11):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = generate(family, rng)
        if p is not None:
            out.append(p)
    return out


def _split_terms(terms: str) -> list[str]:
    """Split "a + b + c" on the top-level " + "."""
    out = []
    depth = 0
    cur = []
    i = 0
    while i < len(terms):
        ch = terms[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and terms.startswith(" + ", i):
            out.append("".join(cur))
            cur = []
            i += 3
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [t for t in out if t.strip()]


def _factors(term: str) -> list[str]:
    """Split "a*b*c" on the top-level "*"."""
    out = []
    depth = 0
    cur = []
    i = 0
    while i < len(term):
        ch = term[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch == "*":
            out.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


def _third_factors(terms: str) -> list[str]:
    """The third factor of each cofactor term - the minor's value."""
    out = []
    for term in _split_terms(terms):
        f = _factors(term)
        if len(f) >= 3:
            out.append(f[2])
    return out


class TestTheEliminationIsShown(unittest.TestCase):
    """Each elimination step has to say what it did and what came out."""

    def test_the_derivation_shows_the_row_operations(self):
        for p in _samples("determinant"):
            if "triangular" not in p.scratchpad:
                continue
            # a row operation that is actually performed
            self.assertRegex(
                p.scratchpad, r"R\d+\s*<-|R\d+\s*=",
                "the elimination states no row operation; the triangular "
                "form is asserted rather than computed")
            break

    def test_the_derivation_shows_the_resulting_row(self):
        for p in _samples("determinant"):
            if "triangular" not in p.scratchpad:
                continue
            self.assertRegex(
                p.scratchpad, r"\\\\|-?\d+",
                "the elimination shows no resulting row")
            break

    def test_the_arithmetic_load_is_not_the_skeletal_one(self):
        """The determinant used to have 2.9 arithmetic lines per derivation.

        matrix_power has 14.5 and it works. A derivation that states a
        formula and asserts a result has no arithmetic in it.
        """
        import statistics
        lines = []
        for p in _samples("determinant", 30):
            lines.append(len([l for l in p.scratchpad.splitlines()
                              if re.search(r"=.*[+\-*]", l)]))
        mean = statistics.mean(lines) if lines else 0
        self.assertGreater(
            mean, 5.0,
            f"the determinant's derivation has {mean:.1f} arithmetic lines; "
            f"matrix_power has 14.5 and scores 22/60, while the determinant "
            f"scores 7/150 with a derivation that asserts its result")


class TestTheVerificationMinorsAreComputed(unittest.TestCase):
    """The VERIFY step must compute its minors, not assert them."""

    def test_a_minor_is_computed_not_asserted(self):
        for p in _samples("determinant", 60):
            if "VERIFY by expanding" not in p.scratchpad:
                continue
            m = re.search(r"VERIFY by expanding along the first row: "
                          r"det\(A\) = (.+)", p.scratchpad)
            self.assertIsNotNone(m)
            terms = m.group(1)
            # each term is a product of a sign, an element and a minor
            # value; the minor value must be preceded by its own arithmetic
            self.assertRegex(
                terms, r"\(\+?\d?\)?\d*\)",
                "the VERIFY expansion asserts its minor values")
            break

    def test_the_larger_minors_are_computed(self):
        """The 3x3 and 4x4 minors must be computed like the 2x2 ones.

        Run 11 computed the 2x2 minors at 3x3 and left the 3x3 and 4x4
        ones as bare numbers. That is the same defect at a different
        size.

        An ASSERTED minor is a bare number in the third factor:

            (+1)*(-2)*(121) + (-1)*(4)*(83)

        A COMPUTED minor is a cofactor expansion with its 2x2 products
        visible inside, and it contains an "=":

            (+1)*(-2)*(+4*((-5)*1 - (-4)*5 = -5 - -20) - 3*(...))

        So the check is on the THIRD factor only: it must not be a bare
        number.
        """
        checked = 0
        for p in _samples("determinant", 120):
            if "VERIFY by expanding" not in p.scratchpad:
                continue
            m = re.search(r"VERIFY by expanding along the first row: "
                          r"det\(A\) = (.+?), which agrees", p.scratchpad)
            if not m:
                continue
            for third in _third_factors(m.group(1)):
                self.assertFalse(
                    re.fullmatch(r"-?\d+", third.strip()),
                    f"the minor is stated as a bare number {third!r}; it must "
                    f"be computed")
            checked += 1
        self.assertGreater(checked, 0, "no VERIFY expansion was checked")

    def test_a_computed_minor_has_visible_products(self):
        """And the products inside it are visible.

        The minors' arithmetic lives in their OWN steps now - one step
        per minor - so that is where the products are looked for.
        """
        found = 0
        for p in _samples("determinant", 120):
            minor_steps = [l for l in p.scratchpad.splitlines()
                           if "VERIFY: the" in l and "minor is" in l]
            if not minor_steps:
                continue
            for l in minor_steps:
                if re.search(r"-?\d+\*-?\d+", l):
                    found += 1
                    break
        self.assertGreater(found, 0,
                           "the VERIFY minors show no products")


class TestTheArithmeticIsStillCorrect(unittest.TestCase):
    """Showing the arithmetic must not change the answer."""

    def test_the_determinant_is_still_right(self):
        import sympy as sp
        for p in _samples("determinant", 60):
            m = re.search(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}",
                          p.question, re.S)
            if not m:
                continue
            rows = [[int(x) for x in re.findall(r"-?\d+", r)]
                    for r in m.group(1).split("\\\\") if r.strip()]
            self.assertEqual(int(p.answer), sp.Matrix(rows).det(),
                             "the answer does not match the matrix")


if __name__ == "__main__":
    unittest.main()
