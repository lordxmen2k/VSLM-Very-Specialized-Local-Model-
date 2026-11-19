"""Every corpus family must speak the benchmark's answer contract.

The benchmark varies its system prompt and its closing instruction by
subcategory. A corpus family that falls through to a generic prompt is
training the model on a contract the grader will not use.
"""
import random
import unittest

from la.benchprompt import BODY_HINTS, FAMILY_TO_SUBCAT, SYSTEM_PROMPTS
from la.problems import generate


BENCHMARK_FAMILIES = {
    "determinant": "determinant",
    "eigenvalue": "eigenvalue",
    "eigenvalue_numeric": "eigenvalue",
    "rank": "rank",
    "nullity": "nullity",
    "matrix_trace": "trace",
    "matrix_transpose": "transpose",
    "matrix_multiply": "multiplication",
    "matrix_power": "matrix_power",
    "matrix_vector": "matrix_vector",
}


class TestFamilyMapping(unittest.TestCase):

    def test_every_benchmark_family_has_a_subcategory_mapping(self):
        for family, subcat in BENCHMARK_FAMILIES.items():
            with self.subTest(family=family):
                self.assertEqual(FAMILY_TO_SUBCAT.get(family), subcat)

    def test_benchmark_family_prompt_is_the_benchmark_prompt(self):
        """Every benchmark family uses the benchmark's system prompt, and
        one of the benchmark's own closing instructions when it carries one.

        The benchmark uses more than one closing instruction per family,
        and the 4x4 file's problems carry none at all, so the closing
        instruction is checked for membership in the family's set.
        """
        from la.augment import closing_variants
        for family, subcat in BENCHMARK_FAMILIES.items():
            with self.subTest(family=family):
                allowed = closing_variants(family)
                rng = random.Random(10)
                saw_closing = False
                for _ in range(60):
                    p = generate(family, rng)
                    self.assertIsNotNone(p)
                    self.assertEqual(p.system_prompt(), SYSTEM_PROMPTS[subcat])
                    if any(v in p.question for v in allowed):
                        saw_closing = True
                self.assertTrue(saw_closing,
                                f"{family} never produced a closing "
                                f"instruction")

    def test_the_contract_checker_reports_a_clean_contract(self):
        import os
        import subprocess
        import sys
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script = os.path.join(repo, "scripts", "check_corpus_contract.py")
        self.assertTrue(os.path.exists(script),
                        "scripts/check_corpus_contract.py does not exist")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(repo, "src")
        out = subprocess.run(
            [sys.executable, script, "--n", "40"],
            cwd=repo, env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0,
                         f"contract checker failed:\n{out.stdout}\n{out.stderr}")
        for subcat in set(BENCHMARK_FAMILIES.values()):
            self.assertIn(subcat, out.stdout)


if __name__ == "__main__":
    unittest.main()
