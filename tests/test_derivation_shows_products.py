"""The derivation must SHOW the multiplications, not just their results.

RUN 9 SCORED 225/660 = 34.1%, down from the 259-run's 39.2% and run 7's
42.7%.

The cause was a change run 9 made to fit the 5x5 examples under the
1536-token sequence cap. The full dot product is:

    4*4 + (-4)*(-4) + 3*3 + 0*0 = 16 + 16 + 9 + 0 = 41

and the compact form run 9 used at the large sizes is:

    = 16 + 16 + 9 + 0 = 41

The compact form drops the PRODUCTS. The model is shown the partial sums
but never the multiplication that produced each one, so it is taught to
emit a sum whose terms it has no idea how to compute.

The cost was immediate and confined to the families that use the dot
product:

    family            259-run   run 9   base
    matrix_power         41      15      34     -26
    matrix_vector        18       8      44     -10
    multiplication       33      29      25      -4
    determinant           4       3      54      -1

`matrix_power` lost 26 problems - more than the whole regression - and
the 4x4 column fell from 95 to 70.

The corpus's own arithmetic checker did NOT catch this, and that is the
point of this module: the checker validates the expression on the LEFT of
each "=", so a line whose left side is already a sum of products is
perfectly well formed and perfectly well valued. Nothing flagged that the
model was being taught less.

So the property is pinned here directly: at every size the corpus can
afford, the derivation must show the product terms.
"""
import os
import random
import re
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, os.path.join(REPO, "src"))

from la.problems import generate


def _dot_product_lines(scratchpad):
    """The derivation lines that state a dot product.

    A dot product line names an entry and gives it a value:

        STEP 2 row 1: (A^2)_{11} = 4 - 4 - 3 + 0 = -3; (A^2)_{12} = ...

    The line is recognised by the ENTRY it names - "(A^2)_{ij}" or
    "(AB)_{ij}" - not by its arithmetic, because the arithmetic is
    exactly what differs between the two forms.
    """
    out = []
    for line in scratchpad.splitlines():
        if re.search(r"\((?:A\^2|AB|AB|A)\)\{?\d", line):
            out.append(line)
        elif re.search(r"\(A\^2\)_\{?\d", line) or re.search(r"\(AB\)_\{?\d", line):
            out.append(line)
    return out


def _tokeniser():
    """The model's own tokenizer, if it has been fetched.

    `make tokenizer` puts it in .cache/. It is not committed - 11 MB of
    vocabulary is not worth a repository - so the test skips without it.
    """
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(repo, "scripts"))
    try:
        from qwen_bpe import QwenBPE
    except Exception:
        return None
    if not os.path.exists(os.path.join(repo, ".cache",
                                       "qwen3_tokenizer.json")):
        return None
    try:
        return QwenBPE()
    except Exception:
        return None


class TestTheDerivationShowsTheMultiplications(unittest.TestCase):
    """The products must be visible wherever there is room for them."""

    def test_matrix_power_shows_its_product_terms(self):
        """At 3x3 and 4x4 the full form fits easily - 591 and 983
        tokens against a 1536 cap - so there is no excuse for dropping
        the products."""
        for size in (3, 4):
            with self.subTest(size=f"{size}x{size}"):
                self._assert_products_shown("matrix_power", size)

    def test_matrix_multiply_shows_its_product_terms(self):
        for size in (3, 4):
            with self.subTest(size=f"{size}x{size}"):
                self._assert_products_shown("matrix_multiply", size)

    def _assert_products_shown(self, family, size):
        rng = random.Random(4242)
        seen = 0
        for _ in range(200):
            p = generate(family, rng)
            if p is None:
                continue
            if f"{size}x{size}" not in p.question:
                continue
            seen += 1
            lines = _dot_product_lines(p.scratchpad)
            if not lines:
                continue
            for line in lines:
                self.assertIn(
                    "*", line,
                    f"{family} at {size}x{size}: the derivation states "
                    f"{line!r} with no product terms. The model is shown "
                    f"the sum but not the multiplication that produced "
                    f"each term, so it is taught to emit a number it "
                    f"cannot compute.")
            if seen >= 3:
                break
        if not seen:
            self.skipTest(f"no {size}x{size} {family} problem was generated")

    def test_the_5x5_form_fits_the_real_training_cap(self):
        """The 5x5 form DOES fit, and it is measured in REAL tokens.

        The exception existed because "the full form runs to 1,601 tokens
        against a 1,536 cap". Both numbers were wrong:

            1,601  measured against a LONGER derivation that no longer
                   exists - the derivation has since been shortened
            1,536  MAX_SEQ_LEN, a stale constant that is enforced
                   NOWHERE. The real cap is max_seq_len: 2048 in
                   configs/default.yaml, and the trainer truncates at
                   that.

        So the exception was never needed, and it was costing the
        matrix_power family 26 problems: 41/60 became 15/60.

        The measurement is now in REAL tokens, from the model's own
        tokenizer, against the real cap. A character count divided by a
        heuristic is what made the exception look necessary.
        """
        import yaml

        from la.benchprompt import row_prompt

        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(repo, "configs", "default.yaml"),
                  encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        cap = int(cfg["training"]["max_seq_len"])

        tokeniser = _tokeniser()
        if tokeniser is None:
            self.skipTest("the model's tokenizer is not available")

        worst = 0
        for family in ("matrix_power", "matrix_multiply"):
            rng = random.Random(4242)
            for _ in range(200):
                p = generate(family, rng)
                if p is None:
                    continue
                m = re.search(r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}",
                              p.question, re.S)
                if not m:
                    continue
                rows = [ln for ln in m.group(1).split("\\\\") if ln.strip()]
                if len(rows) != 5:
                    continue
                n = tokeniser.count(p.prompt()) + tokeniser.count(p.completion())
                worst = max(worst, n)
        self.assertLessEqual(
            worst, cap,
            f"the longest 5x5 example is {worst} real tokens, over the "
            f"{cap} cap, so the products have to be dropped at that size")


class TestTheArithmeticCheckerWouldNotHaveCaughtIt(unittest.TestCase):
    """A demonstration, so the failure mode is not forgotten.

    The corpus's arithmetic checker validates the expression on the LEFT
    of each "=". A compact dot product line is
    "= 16 + 16 + 9 + 0 = 41" - the left side is a valid arithmetic
    expression and it is valued correctly. The checker passes it.

    So the checker cannot see a derivation that has quietly stopped
    teaching something. The contract has to be pinned by a test that
    looks at what is PRESENT, not at whether what is present is right.
    """

    def test_a_compact_line_is_well_formed_and_well_valued(self):
        from la.check_arithmetic import check_problem
        from la.problems import Problem

        p = Problem(
            "matrix_power",
            "Compute A^2 for the 3x3 matrix A.",
            "GIVEN: A, k = 2\n"
            "STEP 1 (A^2)_{11} = 16 + 16 + 9 + 0 = 41",
            "41", "41", {"kind": "matrix_heatmap", "data": {}, "labels": {},
                         "annotate": []}, 1, ["exact", "scalar"])
        self.assertEqual(
            [], check_problem(p),
            "the arithmetic checker accepted a line with no product terms, "
            "which is exactly why this needs a separate test")


if __name__ == "__main__":
    unittest.main()
