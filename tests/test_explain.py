"""The explainer has to say what the model wrote, not just how many it missed."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestExplain(unittest.TestCase):
    def _run(self, records):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "results.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
            env = dict(os.environ)
            env["PYTHONPATH"] = os.path.join(ROOT, "src")
            proc = subprocess.run([sys.executable, "-m", "la.explain", path],
                                  capture_output=True, text=True, env=env, cwd=ROOT)
            self.assertEqual(0, proc.returncode, proc.stderr)
            return proc.stdout

    def test_it_reports_the_missing_scratchpads(self):
        out = self._run([
            {"family": "determinant", "raw": "### Scratchpad\nSTEP 1 x\n"
             "### Answer\n7", "answer": "7", "correct": False},
            {"family": "determinant", "raw": "### Answer\n5", "answer": "5",
             "correct": False},
            {"family": "determinant", "raw": "### Answer\n7", "answer": "7",
             "correct": True},
        ])
        self.assertIn("2 wrong", out)
        self.assertIn("wrote no scratchpad at all", out)
        self.assertIn("50%", out)
        self.assertIn("determinant", out)

    def test_it_shows_the_raw_completion(self):
        out = self._run([
            {"family": "matrix_multiply", "raw": "### Scratchpad\nSTEP 1 AB\n"
             "### Answer\n\\begin{pmatrix} 1 \\end{pmatrix}",
             "answer": "x", "correct": False},
        ])
        self.assertIn("STEP 1 AB", out)
        self.assertIn("matrix_multiply", out)


if __name__ == "__main__":
    unittest.main()
