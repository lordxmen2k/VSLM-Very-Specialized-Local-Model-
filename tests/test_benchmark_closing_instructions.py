"""The benchmark's closing instructions differ from the corpus's.

This pins a measured fact about the released LinAlg-Bench data, and the
decision that follows from it: the closing instruction is a distribution
shift, not a format break, so it is recorded and monitored rather than
"fixed" by changing what the benchmark sends.

The measurement (660 released problems, from tests/_cache):

  * 440 of 660 carry a closing instruction that mentions \\boxed{}
  * 220 of 660 carry NO closing instruction at all - the entire 4x4 file
  * only 126 use the corpus's exact closing wording

The corpus always closes with its family's hint:

    Show all intermediate steps. Put your final numerical answer inside \\boxed{}.

The benchmark uses nine different wordings. The system prompt already says
"Always put your final ... answer inside \\boxed{}.", and the grader falls
back from \\boxed{} to the last value-like line, so a missing closing line
is not fatal for scoring. But a model trained on one closing meeting nine
others - and none at all on a third of the problems - is a shift.

If run 5 underperforms on the 4x4 file specifically, the fix is to add the
corpus's closing line to the benchmark prompt for the problems that lack
one. Do not change it before seeing the score.
"""
import csv
import collections
import os
import unittest

from la.benchprompt import (benchmark_prompt_pair, body_hint,
                            row_prompt, system_prompt_for)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
SIZES = ("3x3", "4x4", "5x5")


def load_rows(size):
    path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def closing_line(user):
    lines = [l for l in user.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


class TestBenchmarkClosingInstructions(unittest.TestCase):
    """What the released benchmark actually sends."""

    def setUp(self):
        self.rows = {s: load_rows(s) for s in SIZES}

    def test_the_three_files_hold_660_problems(self):
        total = sum(len(self.rows[s]) for s in SIZES)
        self.assertEqual(total, 660)

    def test_the_4x4_file_carries_no_closing_instruction(self):
        """The whole 4x4 file is the one with no closing line."""
        for p in self.rows["4x4"]:
            sub = p.get("Subcat") or ""
            _, user = benchmark_prompt_pair(p.get("problem_latex") or "",
                                            sub, "4x4")
            self.assertNotIn("\\boxed", user,
                             f"4x4 problem unexpectedly has a closing "
                             f"instruction: {closing_line(user)!r}")

    def test_3x3_and_5x5_always_mention_boxed(self):
        for size in ("3x3", "5x5"):
            for p in self.rows[size]:
                sub = p.get("Subcat") or ""
                _, user = benchmark_prompt_pair(p.get("problem_latex") or "",
                                                sub, size)
                self.assertIn("\\boxed", user,
                              f"{size} problem has no \\boxed instruction")

    def test_only_some_problems_use_the_corpus_wording(self):
        """The corpus teaches one closing; the benchmark uses nine."""
        wordings = collections.Counter()
        exact = 0
        total = 0
        for size in SIZES:
            for p in self.rows[size]:
                sub = p.get("Subcat") or ""
                _, user = benchmark_prompt_pair(p.get("problem_latex") or "",
                                                sub, size)
                tail = closing_line(user)
                wordings[tail] += 1
                total += 1
                if tail == body_hint(sub):
                    exact += 1
        self.assertGreater(len(wordings), 3,
                           "the benchmark is expected to use several "
                           "different closing instructions")
        self.assertLess(exact, total,
                        "the benchmark is not expected to use the corpus's "
                        "exact wording everywhere")

    def test_the_counts(self):
        """The measured numbers, pinned."""
        with_boxed = 0
        total = 0
        for size in SIZES:
            for p in self.rows[size]:
                sub = p.get("Subcat") or ""
                _, user = benchmark_prompt_pair(p.get("problem_latex") or "",
                                                sub, size)
                total += 1
                if "\\boxed" in user:
                    with_boxed += 1
        self.assertEqual(total, 660)
        self.assertEqual(with_boxed, 440)
        # 220 without = the whole 4x4 file


class TestTheSystemPromptCoversTheGap(unittest.TestCase):
    """Why this is a shift and not a break."""

    def test_every_system_prompt_asks_for_boxed(self):
        for sub in ("determinant", "eigenvalue", "rank", "nullity",
                    "multiplication", "matrix_power", "matrix_vector",
                    "transpose", "trace"):
            self.assertIn("\\boxed", system_prompt_for(sub))

    def test_the_system_prompt_is_the_same_on_both_sides(self):
        """The benchmark's system prompt and the corpus's are one function."""
        rows = load_rows("3x3")
        p = rows[0]
        sub = p.get("Subcat") or ""
        bench_sys, _ = benchmark_prompt_pair(p.get("problem_latex") or "",
                                             sub, "3x3")
        self.assertEqual(bench_sys, system_prompt_for(sub))


class TestBothSidesRenderThroughTheSameFunction(unittest.TestCase):
    """The template is applied once, by one function."""

    def test_row_prompt_renders_system_and_user(self):
        class FakeTok:
            chat_template = "x"

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=True):
                return "".join(
                    f"<|{m['role']}|>\n{m['content']}<|end|>\n"
                    for m in messages) + "<|assistant|>\n"

        rows = load_rows("3x3")
        p = rows[0]
        sub = p.get("Subcat") or ""
        sys_p, user_p = benchmark_prompt_pair(p.get("problem_latex") or "",
                                              sub, "3x3")
        out = row_prompt({"system": sys_p, "user": user_p}, FakeTok())
        self.assertIn("<|system|>", out)
        self.assertIn("<|user|>", out)
        self.assertIn("<|assistant|>", out)
        self.assertIn(sys_p, out)
        self.assertIn(user_p, out)


if __name__ == "__main__":
    unittest.main()
