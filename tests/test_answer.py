"""The answer grammar: print, parse, compare.

The round trip is the point. If a value can be printed and parsed back to
itself, then a model's answer can be checked against a ground truth that was
produced by the same printer.

Written on unittest so the suite runs with the standard library alone; pytest
runs it unchanged if it happens to be installed.
"""
import unittest

import sympy as sp

from la.answer import (answers_equal, check_answer, fmt_matrix, fmt_pairs,
                        fmt_scalar, fmt_set, parse_answer)

SCALARS = [
    sp.Integer(42), sp.Integer(-7), sp.Rational(1, 2), sp.Rational(-3, 4),
    sp.Integer(5) / 2, 2 * sp.sqrt(3), sp.sqrt(2), 1 + sp.I, 3 - 2 * sp.I,
    sp.sqrt(14) / 14, -3 * sp.sqrt(14) / 14,
]


class TestAnswerGrammar(unittest.TestCase):

    def test_scalar_round_trip(self):
        for value in SCALARS:
            with self.subTest(value=value):
                text = fmt_scalar(value)
                back = parse_answer(text)
                self.assertTrue(answers_equal(back, value),
                                f"{value} -> {text!r} -> {back}")

    def test_matrix_round_trip(self):
        M = sp.Matrix([[1, 2], [3, 4]])
        text = fmt_matrix(M)
        self.assertTrue(text.startswith(r"\begin{pmatrix}"))
        back = parse_answer(text)
        self.assertTrue(answers_equal(back, M))
        self.assertEqual(back.shape, M.shape)

    def test_set_round_trip_and_order_irrelevance(self):
        values = [sp.Integer(1), sp.Integer(3), sp.Integer(-2)]
        back = parse_answer(fmt_set(values))
        self.assertTrue(answers_equal(back, values))
        self.assertTrue(answers_equal(parse_answer(r"\{1, 3, -2\}"),
                                       values))
        self.assertTrue(answers_equal(parse_answer(r"\{-2, 3, 1\}"),
                                       values))

    def test_complex_eigenvalue_set(self):
        vals = [1 + sp.I, 1 - sp.I, sp.Integer(3)]
        self.assertTrue(answers_equal(parse_answer(fmt_set(vals)), vals))

    def test_pairs_round_trip(self):
        pairs = {"x_1": sp.Integer(3), "x_2": sp.Integer(-1)}
        back = parse_answer(fmt_pairs(pairs))
        self.assertTrue(answers_equal(back, pairs))

    def test_wrong_answer_is_rejected(self):
        self.assertFalse(answers_equal(parse_answer("42"), sp.Integer(41)))
        self.assertFalse(answers_equal(parse_answer("42"), sp.Float(42.0001)))

    def test_matrix_shape_mismatch_is_rejected(self):
        a = parse_answer(fmt_matrix(sp.Matrix([[1, 2], [3, 4]])))
        b = parse_answer(fmt_matrix(sp.Matrix([[1, 2, 3], [4, 5, 6]])))
        self.assertFalse(answers_equal(a, b))

    def test_check_answer_reports(self):
        self.assertTrue(check_answer(r"\frac{1}{2}", r"\frac{1}{2}")["correct"])
        self.assertFalse(check_answer("garbage", "42")["correct"])

    def test_basis_mode_tolerates_sign_and_orientation(self):
        Q = sp.Matrix([[0, 1], [1, 0]])
        neg = sp.Matrix([[0, -1], [-1, 0]])
        self.assertTrue(answers_equal(parse_answer(fmt_matrix(Q)),
                                       parse_answer(fmt_matrix(neg)),
                                       mode="basis"))
        self.assertFalse(answers_equal(parse_answer(fmt_matrix(Q)),
                                        parse_answer(fmt_matrix(neg))))

    def test_diag_mode_tolerates_diagonal_order(self):
        D1 = sp.diag(1, 2, 3)
        D2 = sp.diag(3, 1, 2)
        self.assertTrue(answers_equal(D1, D2, mode="diag"))
        self.assertFalse(answers_equal(D1, D2))


if __name__ == "__main__":
    unittest.main()
