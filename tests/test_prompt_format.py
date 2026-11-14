"""The corpus is written in the benchmark's prompt format.

A model trained on one question format and scored on another is being asked
to do a task it has never seen phrased, and the score measures that rather
than the mathematics. The first fine-tuned model scored 35 points below the
base model on the benchmark's 3x3 problems while scoring 50 per cent on the
held-out set, which used our own phrasing.
"""
import unittest

from la.augment import question
from la.benchprompt import (PREAMBLE, build_benchmark_prompt, body_hint,
                            has_task, task_for)
from la.problems import GENERATORS, generate


class TestBenchmarkPromptFormat(unittest.TestCase):
    def test_every_family_produces_a_benchmark_shaped_question(self):
        import random
        rng = random.Random(17)
        for family in sorted(GENERATORS):
            p = None
            for _ in range(40):
                p = GENERATORS[family](rng)
                if p is not None:
                    break
            self.assertIsNotNone(p, family)
            q = p.question
            self.assertTrue(q.startswith(PREAMBLE),
                            f"{family} does not start with the preamble")
            self.assertIn("Show all intermediate steps", q,
                          f"{family} is missing the closing instruction")
            self.assertIn("\\boxed{}", q,
                          f"{family} does not ask for the boxed answer")

    def test_the_closing_instruction_matches_the_family(self):
        import random
        rng = random.Random(19)
        for family in ("determinant", "matrix_multiply", "matrix_vector",
                       "matrix_trace", "rank_nullspace"):
            p = None
            for _ in range(40):
                p = GENERATORS[family](rng)
                if p is not None:
                    break
            self.assertIn(body_hint(family), p.question, family)

    def test_the_answer_is_boxed(self):
        import random
        rng = random.Random(23)
        p = None
        for _ in range(40):
            p = GENERATORS["determinant"](rng)
            if p is not None:
                break
        self.assertIn("\\boxed{", p.completion())
        self.assertIn("### Answer", p.completion())

    def test_the_grader_unwraps_a_boxed_answer(self):
        from la.answer import check_answer, strip_boxed
        self.assertEqual("24", strip_boxed("\\boxed{24}"))
        self.assertEqual(r"\begin{pmatrix}1&2\end{pmatrix}",
                         strip_boxed(r"\boxed{\begin{pmatrix}1&2\end{pmatrix}}"))
        self.assertTrue(check_answer("\\boxed{24}", "24")["correct"])
        self.assertTrue(check_answer(r"\boxed{\begin{pmatrix}1&2\end{pmatrix}}",
                                     r"\begin{pmatrix}1&2\end{pmatrix}")["correct"])

    def test_the_benchmark_prompt_reconstructs_a_missing_task(self):
        bare = "A = \\begin{bmatrix} 1 & 2 \\end{bmatrix}"
        out = build_benchmark_prompt(bare, "determinant", "4x4")
        self.assertIn(PREAMBLE, out)
        self.assertIn("Find the determinant of the 4x4 matrix A.", out)
        self.assertNotIn("4x4x4", out)
        self.assertTrue(has_task(bare) is False)

    def test_a_problem_with_its_own_task_is_left_alone(self):
        p3 = ("Find the determinant of the 3x3 matrix A.\n\n"
              "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}\n\n"
              "Show all intermediate steps. Put your final numerical answer "
              "inside \\boxed{}.")
        out = build_benchmark_prompt(p3, "determinant", "3x3")
        self.assertIn(PREAMBLE, out)
        self.assertEqual(1, out.count("Find the determinant of the 3x3"))
        self.assertTrue(has_task(p3))

    def test_no_corpus_example_is_truncated(self):
        """A truncated example loses the end of its completion, which is
        where the plot spec is.

        At a 1024-token cap, 3.9 per cent of the corpus was cut short and
        the model learned to stop before the plot block. The held-out set
        scored 94 per cent on the plot kind, which is the same number.
        """
        import random
        from la.benchprompt import MAX_SEQ_LEN as cap
        rng = random.Random(29)
        worst = 0
        for _ in range(300):
            family = rng.choice(sorted(GENERATORS))
            p = GENERATORS[family](rng)
            if p is None:
                continue
            length = len(p.question) + len(p.completion())
            worst = max(worst, length)
        self.assertLessEqual(worst / 2.04, cap,
                             f"the longest corpus example is about "
                             f"{worst / 2.04:.0f} tokens, over the {cap} cap")

    def test_the_training_prompt_and_the_evaluation_prompt_are_the_same(self):
        """They used to be different, and the model was answering a prompt it
        had never been trained on.

        The corpus's question is the benchmark's prompt, but train.py used to
        wrap it in "### Problem\n...\n### Scratchpad\n" while the
        evaluation passed the bare question. The model was therefore asked at
        evaluation time to continue something it had never seen, and it fell
        back on base-model behaviour: free prose with a boxed answer instead
        of the format it was trained on.

        The prompt and the completion now live in the corpus row, so there is
        one definition of each, and this test pins them together.
        """
        import random
        from la.benchprompt import model_prompt as build_prompt
        rng = random.Random(31)
        p = None
        for _ in range(40):
            p = GENERATORS["determinant"](rng)
            if p is not None:
                break
        row = p.to_json()
        self.assertEqual(row["prompt"], build_prompt(row["question"]),
                         "the training prompt and the evaluation prompt differ")
        self.assertEqual(row["prompt"], row["question"],
                         "the prompt should be the question, not a wrapper "
                         "around it")
        self.assertIn("### Answer", row["completion"])
        self.assertIn("\\boxed{", row["completion"],
                      "the answer must be boxed, because the benchmark asks "
                      "for it that way")

    def test_the_task_is_spelled_once(self):
        for size in ("3", "4", "5", "3x3", "4x4"):
            t = task_for("determinant", size)
            self.assertIn("4x4" if size.startswith("4") else
                          f"{size[0]}x{size[0]}", t, size)
            self.assertNotIn("x4x4", t, size)


if __name__ == "__main__":
    unittest.main()
