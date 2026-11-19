"""Rank and nullity are two questions, not one paired answer."""
import random
import unittest

from la.answer import answers_equal, parse_answer
from la.problems import GENERATORS, generate


class TestRankNullitySplit(unittest.TestCase):

    def test_rank_and_nullity_are_separate_families(self):
        self.assertIn("rank", GENERATORS)
        self.assertIn("nullity", GENERATORS)
        self.assertNotIn("rank_nullspace", GENERATORS)

    def test_each_family_answers_with_one_scalar(self):
        for family in ("rank", "nullity"):
            with self.subTest(family=family):
                p = generate(family, random.Random(31))
                self.assertIsNotNone(p)
                parsed = parse_answer(p.answer)
                self.assertNotIsInstance(parsed, (list, dict))
                self.assertFalse(any(k in p.answer for k in ("rank =", "nullity =")),
                                 "the answer field must not contain a labelled pair")

    def test_the_question_asks_for_only_the_named_quantity(self):
        """The TASK asks for one quantity, not a pair.

        "rank" may appear in a nullity prompt - the benchmark's own nullity
        prompt says "First find the rank using row reduction, then compute
        the nullity" - but the TASK line must ask for the nullity alone.
        That is the defect run 5 had: the corpus taught a pair, and the
        grader read the last number, which was the nullity.
        """
        rank = generate("rank", random.Random(32))
        nullity = generate("nullity", random.Random(32))
        self.assertIsNotNone(rank)
        self.assertIsNotNone(nullity)
        self.assertIn("rank", rank.question.lower())
        self.assertNotIn("nullity", rank.question.lower())
        self.assertIn("nullity", nullity.question.lower())

        def task_line(question):
            # the task is the line after the preamble
            parts = [p for p in question.split("\n\n") if p.strip()]
            return parts[1].lower() if len(parts) > 1 else ""

        self.assertNotIn("rank", task_line(nullity.question))
        self.assertNotIn("nullity", task_line(rank.question))

    def test_a_scalar_rank_answer_scores_against_a_rank_reference(self):
        p = generate("rank", random.Random(33))
        self.assertTrue(answers_equal(parse_answer(p.answer),
                                      parse_answer(p.truth)))

    def test_a_scalar_nullity_answer_scores_against_a_nullity_reference(self):
        p = generate("nullity", random.Random(33))
        self.assertTrue(answers_equal(parse_answer(p.answer),
                                      parse_answer(p.truth)))


if __name__ == "__main__":
    unittest.main()
