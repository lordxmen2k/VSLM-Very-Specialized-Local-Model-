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

    def test_corpus_teaches_a_combined_rank_nullity_answer(self):
        """The bug, pinned: the corpus's only rank-ish answer is a pair."""
        rng = random.Random(4242)
        combined = None
        for _ in range(50):
            p = generate("rank_nullspace", rng)
            if p is None:
                continue
            combined = p.answer
            break
        self.assertIsNotNone(combined, "no rank_nullspace problem generated")
        self.assertIn("rank", combined)
        self.assertIn("nullity", combined)

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

    def test_corpus_eigenvalues_are_always_exact(self):
        """The bug, pinned: the corpus only ever teaches exact values.

        Every corpus eigenvalue is an exact number - an integer, or an
        exact complex like '3 - 2*I'. The benchmark's eigenvalues are
        roots of a characteristic polynomial and are irrational. The
        corpus never shows the model a value it has to round.
        """
        rng = random.Random(99)
        exact = 0
        total = 0
        complex_seen = 0
        for _ in range(30):
            p = generate("eigenvalues", rng)
            if p is None:
                continue
            values = parse_answer(p.answer)
            for v in values:
                total += 1
                if getattr(v, "is_integer", False):
                    exact += 1
                elif getattr(v, "is_number", False):
                    complex_seen += 1
        self.assertGreater(total, 0)
        self.assertEqual(exact + complex_seen, total,
                         "every corpus eigenvalue should be exact")
        # and none of them is an irrational
        for _ in range(30):
            p = generate("eigenvalues", rng)
            if p is None:
                continue
            for v in parse_answer(p.answer):
                self.assertTrue(
                    getattr(v, "is_Integer", None) is not None
                    or not v.free_symbols,
                    f"corpus eigenvalue {v} should be an exact number")

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

    def test_corpus_teaches_only_one_method_for_3x3_determinants(self):
        """The bug, pinned: there is no second path for 3x3."""
        rng = random.Random(7)
        saw_3x3 = False
        for _ in range(40):
            p = generate("determinant", rng)
            if p is None:
                continue
            m = re.search(r"of the (\d)x\1", p.question)
            if not m or m.group(1) != "3":
                continue
            saw_3x3 = True
            # the 3x3 derivation is cofactor expansion, every time
            self.assertIn("cofactor", p.scratchpad.lower())
            break
        self.assertTrue(saw_3x3, "no 3x3 determinant generated")


class TestBenchmarkAndCorpusShareTheTaskVocabulary(unittest.TestCase):
    """The benchmark's subcategory labels and the corpus's family names
    must line up, or one family silently never trains.

    The corpus calls it 'eigenvalues'; the benchmark calls it 'eigenvalue'.
    The corpus folds rank and nullity into 'rank_nullspace'; the benchmark
    asks for each alone. These are the same family of problem under two
    names, and a mismatch is invisible until the by-task table is read.
    """

    def test_eigenvalue_family_names_differ(self):
        """The naming mismatch, pinned so it stays visible."""
        from la.problems import GENERATORS
        from la.benchprompt import TASKS
        self.assertIn("eigenvalues", GENERATORS)
        self.assertIn("eigenvalue", TASKS)

    def test_the_two_eigenvalue_names_diverge_on_the_body_hint(self):
        """A second, smaller mismatch, in the same family.

        The benchmark's 'eigenvalue' asks for "your final answers", plural.
        The corpus's 'eigenvalues' falls through to the determinant hint,
        which asks for a "final numerical answer", singular. It is a
        cosmetic difference next to the exact-versus-decimal content
        problem, but it is the same root cause: two names for one family
        and nobody reconciled them.
        """
        from la.benchprompt import body_hint
        self.assertNotEqual(body_hint("eigenvalue"), body_hint("eigenvalues"))
        self.assertIn("answers", body_hint("eigenvalue"))
        self.assertIn("numerical answer", body_hint("eigenvalues"))


if __name__ == "__main__":
    unittest.main()
