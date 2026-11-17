"""The corpus mix must be declared, not implied by a round robin."""
import os
import unittest

import yaml

from la.build_dataset import build
from la.problems import GENERATORS


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_FAMILIES = {
    "determinant", "eigenvalue", "eigenvalue_numeric", "rank", "nullity",
    "matrix_trace", "matrix_transpose", "matrix_multiply", "matrix_power",
    "matrix_vector",
}


class TestCorpusWeights(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(REPO, "configs", "default.yaml"),
                  encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        self.weights = self.config["data"]["family_weights"]

    def test_every_family_has_a_declared_weight(self):
        self.assertEqual(set(self.weights), set(GENERATORS))

    def test_benchmark_and_background_families_are_declared_apart(self):
        benchmark = {f for f in self.weights if f in BENCHMARK_FAMILIES}
        background = {f for f in self.weights if f not in BENCHMARK_FAMILIES}
        self.assertEqual(benchmark, BENCHMARK_FAMILIES)
        self.assertEqual(len(background), 9)

    def test_declared_weights_are_70_30(self):
        total = sum(self.weights.values())
        benchmark = sum(self.weights[f] for f in BENCHMARK_FAMILIES)
        self.assertAlmostEqual(benchmark / total, 0.70, places=3)

    def test_build_produces_the_declared_mix(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train.jsonl")
            problems = build(400, 7, sorted(GENERATORS), path,
                             family_weights=self.weights)
            counts = {}
            for p in problems:
                counts[p.family] = counts.get(p.family, 0) + 1
            benchmark = sum(v for f, v in counts.items() if f in BENCHMARK_FAMILIES)
            self.assertGreaterEqual(benchmark / len(problems), 0.65)
            self.assertLessEqual(benchmark / len(problems), 0.75)


if __name__ == "__main__":
    unittest.main()
