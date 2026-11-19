"""A malformed plot spec must not crash the evaluation harness.

Run 6's `make eval` died with

    AttributeError: 'list' object has no attribute 'get'
    at la/schema.py:208, in spec_features

The model wrote a spec whose `data` was a list rather than an object, and
`spec_features` called `.get()` on it. The evaluation harness was reading a
model's output, so it has to survive whatever shape the model produces.

A crash here costs the whole held-out score, not one problem.
"""
import unittest

from la.schema import features_match, spec_features, validate_spec


class TestSpecFeaturesSurvivesMalformedSpecs(unittest.TestCase):

    def test_a_list_data_does_not_crash(self):
        spec = {"kind": "matrix_heatmap", "data": [[1, 2], [3, 4]]}
        self.assertIsInstance(spec_features(spec), dict)

    def test_a_bare_list_spec_does_not_crash(self):
        self.assertIsInstance(spec_features([1, 2, 3]), dict)

    def test_a_spec_with_no_data_does_not_crash(self):
        self.assertIsInstance(spec_features({"kind": "matrix_heatmap"}), dict)

    def test_a_spec_with_no_kind_does_not_crash(self):
        self.assertIsInstance(spec_features({}), dict)

    def test_a_malformed_spec_does_not_match_a_good_one(self):
        """It must not crash, and it must not score as a match either."""
        good = {"kind": "matrix_heatmap",
                "data": {"matrix": [[1.0, 2.0], [3.0, 4.0]]}}
        bad = {"kind": "matrix_heatmap", "data": [[1, 2], [3, 4]]}
        self.assertFalse(features_match(spec_features(bad),
                                        spec_features(good)))

    def test_validate_spec_rejects_a_list_data(self):
        """The validator already rejects it; the features must not crash."""
        spec = {"kind": "matrix_heatmap", "data": [[1, 2], [3, 4]]}
        self.assertNotEqual(validate_spec(spec), [])


if __name__ == "__main__":
    unittest.main()
