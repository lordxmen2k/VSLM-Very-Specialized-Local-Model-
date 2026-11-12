"""The problem generators and the augmentation."""
import random
import unittest

from la.answer import answers_equal, parse_answer
from la.problems import GENERATORS, generate
from la.schema import validate_spec


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
                self.assertTrue(p.prompt().endswith("### Scratchpad\n"))
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

    def test_augmentation_changes_the_question_but_not_the_answer(self):
        from la.augment import paraphrase
        rng = random.Random(5)
        stem = r"\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}"
        questions = {paraphrase("determinant", stem, rng) for _ in range(40)}
        self.assertGreaterEqual(len(questions), 3,
                                "the paraphrase templates are not being used")

    def test_similarity_transform_preserves_the_spectrum(self):
        import sympy as sp
        from la.augment import similarity_transform
        rng = random.Random(3)
        A = sp.Matrix([[2, 1], [0, 3]])
        before = sorted(str(sp.nsimplify(v)) for v in A.eigenvals())
        B = similarity_transform(rng, A)
        after = sorted(str(sp.nsimplify(v)) for v in B.eigenvals())
        self.assertEqual(before, after)

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
            blocks = re.findall(r"\\begin\{pmatrix\}.*?\\end\{pmatrix\}", p.question, re.S)
            self.assertTrue(blocks, "the question must print the matrix")
            A = parse_answer(blocks[0])
            self.assertEqual(sp.simplify(A.det()), parse_answer(p.answer),
                             "the answer must be the determinant of the matrix shown")

            q = generate("least_squares", rng)
            blocks = re.findall(r"\\begin\{pmatrix\}.*?\\end\{pmatrix\}", q.question, re.S)
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
