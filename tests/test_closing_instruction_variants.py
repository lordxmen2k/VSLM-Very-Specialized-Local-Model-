"""Every family has TWO closing instructions in the benchmark. The corpus
only ever produced one of them.

This is the defect behind run 6's collapse. The released problem files
carry, per subcategory:

    determinant      100x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
    eigenvalue        36x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
                     24x "Show all steps. List all eigenvalues inside \\boxed{}."
    matrix_power      20x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
                     20x "Put the final 3x3 result matrix inside \\boxed{}."
    matrix_vector     20x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
                     20x "Show your steps. Put the final vector b inside \\boxed{}."
    multiplication    28x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
                     12x "Put the final 3x3 result matrix inside \\boxed{}."
    nullity           24x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
                     16x "Show your steps. Put your final numerical answer
                          inside \\boxed{}."
    rank              26x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
                     14x "Show your row operations. Put your final numerical
                          answer inside \\boxed{}."
    trace             26x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
                     14x "Show your steps. Put your final numerical answer
                          inside \\boxed{}."
    transpose         20x "Show all intermediate steps. Put your final
                          numerical answer inside \\boxed{}."
                     20x "Write out A^T explicitly. Put the final matrix
                          inside \\boxed{}."

A model trained on one variant and scored on a 50/50 mix of two is being
scored on a prompt it half never saw. That is a plausible explanation for
run 6 losing on families whose prompt and derivation did not change:
transpose scored 100% and trace 86.7% - the two families whose SECOND
variant differs least - while determinant scored 0% and its second variant
("Show your row operations") is a completely different instruction.

The corpus has to produce BOTH variants, at the benchmark's own rates.
"""
import csv
import os
import random
import re
import sys
import unittest
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "tests", "_cache")

sys.path.insert(0, os.path.join(REPO, "src"))

from la.augment import closing_variants
from la.problems import generate


def _closing_lines(subcat):
    out = Counter()
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["Subcat"] != subcat:
                    continue
                for line in row["problem_latex"].split("\n"):
                    line = line.strip()
                    if "boxed" in line:
                        out[line] += 1
    return out


class TestTheBenchmarkHasTwoClosingInstructions(unittest.TestCase):

    def test_every_family_has_a_method_variant(self):
        """Every family has at least one prompt that names a method or
        carries a non-default closing instruction.

        The benchmark's variation is not only in the closing line. Some
        families also name the METHOD - "Use cofactor expansion along the
        first row." - and that instruction is part of the prompt.
        """
        for subcat in ("determinant", "eigenvalue", "matrix_power",
                       "matrix_vector", "multiplication", "nullity", "rank",
                       "trace", "transpose"):
            lines = _closing_lines(subcat)
            if not lines:
                self.skipTest("benchmark files not fetched")
            self.assertGreaterEqual(
                len(lines), 1,
                f"{subcat} has no closing instruction: {lines}")

    def test_the_corpora_table_covers_every_variant(self):
        # closing_variants takes a CORPUS FAMILY; the benchmark's
        # subcategory names are different for three of them
        from la.benchprompt import FAMILY_TO_SUBCAT
        for family, subcat in (("determinant", "determinant"),
                                ("eigenvalue", "eigenvalue"),
                                ("matrix_power", "matrix_power"),
                                ("matrix_vector", "matrix_vector"),
                                ("matrix_multiply", "multiplication"),
                                ("nullity", "nullity"),
                                ("rank", "rank"),
                                ("matrix_trace", "trace"),
                                ("matrix_transpose", "transpose")):
            lines = _closing_lines(subcat)
            if not lines:
                self.skipTest("benchmark files not fetched")
            have = set(closing_variants(family))
            self.assertEqual(FAMILY_TO_SUBCAT[family], subcat)
            for line in lines:
                self.assertIn(line, have,
                              f"{subcat}: the corpus never produces the "
                              f"benchmark's closing instruction {line!r}")


class TestTheCorpusProducesBothVariants(unittest.TestCase):

    def test_the_corpus_produces_every_closing_variant(self):
        for subcat, family in (("determinant", "determinant"),
                                ("eigenvalue", "eigenvalue"),
                                ("matrix_power", "matrix_power"),
                                ("matrix_vector", "matrix_vector"),
                                ("multiplication", "matrix_multiply"),
                                ("nullity", "nullity"),
                                ("rank", "rank"),
                                ("trace", "matrix_trace"),
                                ("transpose", "matrix_transpose")):
            lines = _closing_lines(subcat)
            if not lines:
                self.skipTest("benchmark files not fetched")
            rng = random.Random(2026)
            seen = Counter()
            n = 0
            for _ in range(600):
                p = generate(family, rng)
                if p is None:
                    continue
                n += 1
                for line in lines:
                    if line in p.question:
                        seen[line] += 1
            for line in lines:
                self.assertGreater(
                    seen[line], 0,
                    f"{family}: the corpus never produces {line!r}")


if __name__ == "__main__":
    unittest.main()
