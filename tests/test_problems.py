"""The problem generators and the augmentation."""
import random
import unittest

from la.answer import answers_equal, parse_answer
from la.benchprompt import PREAMBLE, body_hint
from la.problems import GENERATORS, generate
from la.schema import validate_spec

MATRIX_BLOCK = r"\\begin\{(?:p|b)matrix\}.*?\\end\{(?:p|b)matrix\}"


class TestProblems(unittest.TestCase):

    def test_every_family_generates_a_self_consistent_problem(self):
        for family in sorted(GENERATORS):
            with self.subTest(family=family):
                p = generate(family, random.Random(1234))
                self.assertIsNotNone(p, f"{family} produced nothing")
                self.assertTrue(p.question.strip())
                self.assertTrue(p.scratchpad.strip())
                self.assertIn("VERIFY", p.scratchpad,
                              "every derivation must end with a check")
                got = parse_answer(p.answer)
                truth = parse_answer(p.truth)
                self.assertTrue(answers_equal(got, truth, mode=p.compare_mode))
                if p.spec is not None:
                    self.assertEqual(validate_spec(p.spec), [])
                # the prompt is the question, in the benchmark's format:
                # the preamble, the task, the body, and the closing
                # instruction on the examples that carry one
                self.assertTrue(p.prompt().startswith(PREAMBLE))
                from la.augment import closing_variants
                allowed = closing_variants(family)
                found = [v for v in allowed if v in p.prompt()]
                if not found:
                    # no closing instruction is the bare-matrix shape, which
                    # is valid: all 220 of the benchmark's 4x4 problems are
                    # a matrix with no closing instruction
                    self.assertNotIn("Show all intermediate steps",
                                     p.prompt())
                    self.assertNotIn("Put the final", p.prompt())
                self.assertIn("### Answer", p.completion())
                self.assertIn("### Plot", p.completion())

    def test_every_family_produces_many_distinct_problems(self):
        for family in sorted(GENERATORS):
            with self.subTest(family=family):
                rng = random.Random(99)
                seen = set()
                for _ in range(12):
                    p = generate(family, rng)
                    if p is not None:
                        seen.add(p.question)
                self.assertGreaterEqual(len(seen), 6,
                                        f"{family} produced only {len(seen)} distinct questions")

    def test_corpus_is_reproducible_from_a_seed(self):
        a = [generate(f, random.Random(7)).question for f in sorted(GENERATORS)]
        b = [generate(f, random.Random(7)).question for f in sorted(GENERATORS)]
        self.assertEqual(a, b)

    def test_every_generated_answer_can_be_read_back_by_the_grader(self):
        """A model that writes an answer back exactly as the corpus printed it
        must score correct.

        One problem per family is not enough to find this. The grammar read a
        matrix cell as a single factor, so a cell holding
        \\frac{1}{10}\\sqrt{10} left the row one entry short and the matrix
        came out ragged - and that shape turns up in about a quarter of the
        orthonormalisation answers and almost none of the others. The test
        above, sampling one instance per family, passed throughout.
        """
        checked = 0
        for family in sorted(GENERATORS):
            rng = random.Random(2024)
            for _ in range(40):
                p = generate(family, rng)
                if p is None:
                    continue
                checked += 1
                with self.subTest(family=family, answer=p.answer[:80]):
                    got = parse_answer(p.answer)
                    truth = parse_answer(p.truth)
                    self.assertTrue(answers_equal(got, truth, mode=p.compare_mode))
        self.assertGreater(checked, 400)

    def test_augmentation_changes_the_question_but_not_the_answer(self):
        """The task statement varies - but only where the benchmark allows
        it to.

        For a family the BENCHMARK SCORES the task statement is the
        benchmark's own, verbatim, because the statement is deterministic
        per (family, size) and the model is scored on that exact string.
        One phrasing is the contract, not a bug.

        For a family the benchmark does NOT score, the wording is still
        chosen from this module's templates, and those must vary.
        """
        from la.augment import paraphrase
        from la.benchprompt import FAMILY_TO_SUBCAT

        rng = random.Random(5)
        scored = {f for f, s in FAMILY_TO_SUBCAT.items()}
        questions = {paraphrase("determinant", rng, "3x3") for _ in range(40)}
        if "determinant" in scored:
            self.assertEqual(
                1, len(questions),
                "a benchmark-scored family must use the benchmark's own "
                f"task statement, but got {questions}")
        else:
            self.assertGreaterEqual(len(questions), 3,
                                    "the paraphrase templates are not being used")

        # a family the benchmark does not score keeps its variety
        rng = random.Random(5)
        others = {paraphrase("svd", rng, "3x3") for _ in range(40)}
        self.assertGreaterEqual(len(others), 3,
                                "an unscored family should keep varied "
                                f"wordings, but got {others}")

    def test_similarity_transform_preserves_the_spectrum(self):
        import sympy as sp
        from la.augment import similarity_transform
        rng = random.Random(3)
        A = sp.Matrix([[2, 1], [0, 3]])
        before = sorted(str(sp.nsimplify(v)) for v in A.eigenvals())
        B = similarity_transform(rng, A)
        after = sorted(str(sp.nsimplify(v)) for v in B.eigenvals())
        self.assertEqual(before, after)

    def test_the_corpus_uses_both_matrix_environments(self):
        """LinAlg-Bench writes every matrix as bmatrix. A corpus that only
        ever writes pmatrix meets that for the first time on benchmark day."""
        import re as _re
        rng = random.Random(101)
        seen = set()
        for _ in range(120):
            p = generate("determinant", rng)
            if p is None:
                continue
            seen.update(_re.findall(r"\\begin\{(p|b)matrix\}", p.question))
        self.assertEqual({"p", "b"}, seen,
                         f"expected both environments, saw {seen}")

    def test_answer_describes_the_matrix_in_the_question(self):
        """Recompute from the question text and check it agrees.

        This is the invariant a self-consistency test cannot catch. If the
        matrix is relabelled after the answer was computed, the answer and
        the truth still agree with each other while both describe a
        different matrix from the one the reader is shown.
        """
        import re
        import sympy as sp
        from la.answer import fmt_matrix, parse_answer

        rng = random.Random(21)
        for _ in range(8):
            p = generate("determinant", rng)
            blocks = re.findall(MATRIX_BLOCK, p.question, re.S)
            self.assertTrue(blocks, "the question must print the matrix")
            A = parse_answer(blocks[0])
            self.assertEqual(sp.simplify(A.det()), parse_answer(p.answer),
                             "the answer must be the determinant of the matrix shown")

            q = generate("least_squares", rng)
            blocks = re.findall(MATRIX_BLOCK, q.question, re.S)
            A = parse_answer(blocks[0])
            b = parse_answer(blocks[1])
            x = (A.T * A).inv() * A.T * b
            self.assertEqual(sp.simplify(x - parse_answer(q.answer)), sp.zeros(2, 1),
                             "the answer must solve the system printed in the question")

    def test_row_permutation_preserves_the_solution(self):
        import sympy as sp
        from la.problems import _permute_pair
        rng = random.Random(3)
        A = sp.Matrix([[2, 1], [1, 3]])
        b = sp.Matrix([5, 7])
        A2, b2 = _permute_pair(rng, A, b)
        self.assertEqual(A2.rows, A.rows)
        x = A.LUsolve(b)
        x2 = A2.LUsolve(b2)
        self.assertEqual(sp.simplify(x - x2), sp.zeros(2, 1))


if __name__ == "__main__":
    unittest.main()
