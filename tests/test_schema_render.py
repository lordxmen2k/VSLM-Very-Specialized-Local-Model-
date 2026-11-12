"""The plot-spec schema and the renderer."""
import os
import tempfile
import unittest

from la.render import render
from la.schema import (SPEC_KINDS, features_match, spec_features,
                        validate_spec)


def _spec(kind, data, labels=None, annotate=None):
    return {"kind": kind, "data": data, "labels": labels or {},
            "annotate": annotate or []}


GOOD = {
    "basis_2d": _spec("basis_2d", {"vectors": [[1, 0], [0, 1]]}),
    "basis_3d": _spec("basis_3d", {"vectors": [[1, 0, 0], [0, 1, 0]]}),
    "eigenvalue_spectrum": _spec("eigenvalue_spectrum",
                                 {"values": [[1, 0], [3, 0], [1, 1]]}),
    "row_echelon": _spec("row_echelon", {"matrix": [[2, 1], [1, 2]],
                         "echelon": [[1, 0], [0, 1]]}),
    "subspace_2d": _spec("subspace_2d", {"vectors": [[2, 1]], "origin": [0, 0]}),
    "projection": _spec("projection", {"vector": [3, 4], "projection": [1, 2],
                         "residual": [2, 2], "axis": [1, 2]}),
    "least_squares": _spec("least_squares", {"points": [[1, 2], [2, 4]],
                            "fit": [[1, 2], [2, 4]]}),
    "singular_spectrum": _spec("singular_spectrum", {"values": [5.0, 3.0, 1.0]}),
    "matrix_heatmap": _spec("matrix_heatmap", {"matrix": [[4, -2], [1, 3]]}),
    "transform_unit_square": _spec("transform_unit_square",
                                  {"matrix": [[2, 1], [1, 2]],
                                  "image": [[0, 0], [2, 0], [3, 3], [1, 3]]}),
}


class TestSchemaAndRender(unittest.TestCase):

    def test_every_kind_validates_and_renders(self):
        for kind, spec in sorted(GOOD.items()):
            with self.subTest(kind=kind):
                self.assertEqual(validate_spec(spec), [])
                with tempfile.TemporaryDirectory() as d:
                    out = os.path.join(d, "f.png")
                    render(spec, out)
                    self.assertGreater(os.path.getsize(out), 1000)

    def test_unknown_kind_is_rejected(self):
        self.assertNotEqual(validate_spec(_spec("pie_chart", {"values": [1]})), [])

    def test_missing_data_key_is_rejected(self):
        self.assertNotEqual(
            validate_spec(_spec("matrix_heatmap", {"matrixx": [[1]]})), [])

    def test_hallucinated_data_key_is_rejected(self):
        self.assertNotEqual(
            validate_spec(_spec("matrix_heatmap", {"matrix": [[1]], "extra": 3})), [])

    def test_nan_is_rejected(self):
        self.assertNotEqual(
            validate_spec(_spec("matrix_heatmap",
                               {"matrix": [[float("nan"), 1], [2, 3]]})), [])

    def test_wrong_shape_is_rejected(self):
        self.assertNotEqual(
            validate_spec(_spec("matrix_heatmap", {"matrix": [[1, 2], [3]]})), [])

    def test_labels_may_be_a_list(self):
        ok = _spec("eigenvalue_spectrum", {"values": [[1, 0]]},
                   labels={"points": ["1", "2"]})
        self.assertEqual(validate_spec(ok), [])
        bad = _spec("eigenvalue_spectrum", {"values": [[1, 0]]},
                    labels={"points": [1, 2]})
        self.assertNotEqual(validate_spec(bad), [])

    def test_feature_extraction_compares_numbers_not_pixels(self):
        a = GOOD["eigenvalue_spectrum"]
        b = _spec("eigenvalue_spectrum",
                  {"values": [[1.0000001, 0], [3, 0], [1, 1]]})
        self.assertTrue(features_match(spec_features(a), spec_features(b), tol=1e-4))
        c = _spec("eigenvalue_spectrum", {"values": [[1, 0], [3, 0], [2, 1]]})
        self.assertFalse(features_match(spec_features(a), spec_features(c)))

    def test_all_kinds_have_a_handler(self):
        from la import render as R
        self.assertEqual(set(R._HANDLERS), set(SPEC_KINDS))


if __name__ == "__main__":
    unittest.main()
