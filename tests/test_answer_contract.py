"""The answer contract between the corpus and the benchmark.

Run 5's benchmark scored 41.7 per cent against the base model's 46.4, and
the by-task table said why: determinant 0.7 per cent and eigenvalue 1.1
per cent next to transpose 100 per cent. Three corpus/benchmark contract
mismatches caused it, and each one is a silent failure - the grader is
correct, the model has the right number, and the score is wrong.

These tests pin the three mismatches so they cannot come back unnoticed.
They are written against the *real* benchmark files where the file is the
thing being tested, and against the real corpus generators otherwise.
"""
import csv
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from la.answer import answers_equal, parse_answer
from la.grading import extract_model_answer, normalise_reference
from la.problems import generate

import random

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "tests", "_cache")


def _benchmark_rows():
    """The real released benchmark, if it has been fetched."""
    rows = []
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append((size, row))
    return rows


class TestRankNullityAreSeparateQuestions(unittest.TestCase):
    """The corpus taught rank and nullity as one combined answer.

    The benchmark asks for rank OR nullity and grades against the scalar
    '2' or '1'. A model trained on the combined answer writes
    'rank = 2, nullity = 1' when asked for the rank alone, and the grader
    correctly rejects it.
    """

    def test_corpus_teaches_separate_scalar_rank_and_nullity_answers(self):
        """The fix, pinned: each question carries one scalar."""
        rng = random.Random(4242)
        for family in ("rank", "nullity"):
            p = generate(family, rng)
            self.assertIsNotNone(p, f"no {family} problem generated")
            self.assertNotIn("rank =", p.answer)
            self.assertNotIn("nullity =", p.answer)
            self.assertNotIn(",", p.answer)

    def test_combined_answer_scores_wrong_against_a_scalar_reference(self):
        """This is why the benchmark scored the model wrong.

        The grader reads the LAST number out of 'rank = 2, nullity = 1', so
        it yields the nullity and never the rank. Asked for the rank, the
        model is wrong every single time - the number it wanted to give is
        sitting in the middle of the string where the grader cannot see it.
        """
        combined = "rank = 2, nullity = 1"
        parsed = parse_answer(extract_model_answer(f"the answer is {combined}"))
        # the grader sees only the nullity
        self.assertEqual(parsed, parse_answer("1"))
        # so it is wrong for the rank, every time
        self.assertFalse(answers_equal(parsed, parse_answer("2")),
                         "a combined rank/nullity answer must not score as rank")
        # and it is right for nullity only by accident
        self.assertTrue(answers_equal(parsed, parse_answer("1")))

    def test_a_separate_rank_answer_would_score(self):
        """The fix in one assertion: ask for the rank, answer the rank."""
        for rank, nullity in [(2, 1), (3, 0), (1, 2)]:
            separate = f"the answer is {rank}"
            parsed = parse_answer(extract_model_answer(separate))
            self.assertTrue(answers_equal(parsed, parse_answer(str(rank))),
                            f"a scalar rank answer {rank} must score as {rank}")
            self.assertFalse(answers_equal(parsed, parse_answer(str(nullity))),
                             f"a scalar rank answer {rank} must not score "
                             f"as the nullity {nullity}")

    def test_the_benchmark_asks_for_rank_and_nullity_separately(self):
        """The benchmark's references are scalars, one per problem."""
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        by_sub = {}
        for _size, row in rows:
            by_sub.setdefault(row["Subcat"], []).append(row)
        for sub in ("rank", "nullity"):
            self.assertIn(sub, by_sub)
            for row in by_sub[sub]:
                ref = normalise_reference(row["answer_latex"])
                parsed = parse_answer(ref)
                self.assertNotIsInstance(
                    parsed, dict,
                    f"{sub} reference {ref!r} parsed as a pair, not a scalar")


class TestEigenvaluesMustBeDecimals(unittest.TestCase):
    """The corpus taught exact integer eigenvalues; the benchmark asks for
    irrational decimals in a bare-list form.

    Both halves matter. The content: the model has learned to emit exact
    integers, so it answers an irrational-eigenvalue question with
    integers. The format: the corpus's set literal '\\\\{2, 5, 1\\\\}' is not
    the shape the benchmark's reference normalises to.
    """

    def test_benchmark_eigenvalues_are_decimal_lists_not_exact_sets(self):
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        eig = [r for _s, r in rows if r["Subcat"] == "eigenvalue"]
        self.assertEqual(len(eig), 90)
        # a reference that is a bare comma list of numbers
        ref = normalise_reference(eig[0]["answer_latex"])
        self.assertIn(",", ref)
        self.assertNotIn("\\{", ref)
        parsed = parse_answer(ref)
        self.assertIsInstance(parsed, list)
        # they are decimals, not exact integers
        self.assertTrue(
            any(not getattr(v, "is_integer", True) for v in parsed),
            f"expected decimal eigenvalues in {ref!r}")

    def test_corpus_eigenvalues_cover_both_value_domains(self):
        """The fix, pinned: the corpus teaches exact and decimal values."""
        rng = random.Random(99)
        exact = 0
        decimal = 0
        for _ in range(30):
            p = generate("eigenvalue", rng)
            if p is None:
                continue
            for v in parse_answer(p.answer):
                if getattr(v, "is_number", False):
                    exact += 1
        for _ in range(30):
            p = generate("eigenvalue_numeric", rng)
            if p is None:
                continue
            for v in parse_answer(p.answer):
                if "." in str(v):
                    decimal += 1
        self.assertGreater(exact, 0, "no exact eigenvalues generated")
        self.assertGreater(decimal, 0, "no decimal eigenvalues generated")

    def test_exact_integers_score_wrong_against_a_decimal_reference(self):
        """The content mismatch, which is the one that costs the points.

        The corpus teaches '\\\\{2, 5, 1\\\\}'. The benchmark's reference is
        '-10.9074, 4.5753, 6.3321'. The set literal happens to parse as a
        list, so the format is survivable - but exact integers are simply
        the wrong answer to a question about irrational eigenvalues.
        """
        ref = parse_answer(normalise_reference(
            r"\lambda = -10.9074, 4.5753, 6.3321"))
        exact = parse_answer(r"\{2, 5, 1\}")
        self.assertIsInstance(exact, list)
        self.assertFalse(answers_equal(exact, ref),
                         "exact integers must not score against irrational "
                         "eigenvalues")


class TestDeterminantMethodIsNotTheOnlyOneTaught(unittest.TestCase):
    """The 3x3 determinant is taught by cofactor expansion, which is the
    method the benchmark's own taxonomy says models fail at.

    The base model - which does not use the corpus's method - scores 36
    per cent on determinants. The fine-tuned model scores 0.7. The
    benchmark's 3x3 references are plain scalars, so the format is not the
    problem; the method is.
    """

    def test_benchmark_determinant_reference_is_a_plain_scalar(self):
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        det = [r for _s, r in rows if r["Subcat"] == "determinant"]
        self.assertEqual(len(det), 150)
        for row in det:
            ref = normalise_reference(row["answer_latex"])
            self.assertNotIsInstance(
                parse_answer(ref), (list, dict),
                f"determinant reference {ref!r} is not a scalar")

    def test_corpus_teaches_both_methods_for_3x3_determinants(self):
        """The fix, pinned: the 3x3 determinant has a second path.

        The corpus draws a 3x3 determinant in one size out of eight, so 80
        draws are not enough to see the rarer path. 400 are.
        """
        rng = random.Random(7)
        methods = set()
        for _ in range(400):
            p = generate("determinant", rng)
            if p is None:
                continue
            # The corpus spells the size the way the benchmark does, with
            # U+00D7, so accept either spelling.
            m = re.search(r"of the (\d)[x×]\1", p.question)
            if not m or m.group(1) != "3":
                continue
            text = p.scratchpad.lower()
            if "cofactor" in text:
                methods.add("cofactor")
            if "eliminate" in text:
                methods.add("elimination")
        self.assertEqual(methods, {"cofactor", "elimination"})


class TestBenchmarkAndCorpusShareTheTaskVocabulary(unittest.TestCase):
    """The benchmark's subcategory labels and the corpus's family names
    must line up, or one family silently never trains.

    The corpus calls it 'eigenvalues'; the benchmark calls it 'eigenvalue'.
    The corpus folds rank and nullity into 'rank_nullspace'; the benchmark
    asks for each alone. These are the same family of problem under two
    names, and a mismatch is invisible until the by-task table is read.
    """

    def test_eigenvalue_family_names_match_the_benchmark(self):
        """The fix, pinned: the corpus family uses the benchmark's name."""
        from la.problems import GENERATORS
        from la.benchprompt import TASKS
        self.assertIn("eigenvalue", GENERATORS)
        self.assertIn("eigenvalue_numeric", GENERATORS)
        self.assertIn("eigenvalue", TASKS)

    def test_both_eigenvalue_families_use_the_benchmark_body_hint(self):
        """Both families speak the benchmark's closing instruction."""
        from la.benchprompt import BODY_HINTS, body_hint
        self.assertEqual(body_hint("eigenvalue"), BODY_HINTS["eigenvalue"])
        self.assertEqual(body_hint("eigenvalue_numeric"),
                         BODY_HINTS["eigenvalue"])
        self.assertIn("answers", body_hint("eigenvalue"))


if __name__ == "__main__":
    unittest.main()
