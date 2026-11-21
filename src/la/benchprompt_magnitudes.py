"""The matrix magnitudes the benchmark uses, per family and size.

Run 7 scored 282/660 = 42.7% against its base model's 306/660 = 46.4%.
Every family was competitive or better EXCEPT the determinant:

    determinant   run 7 = 4/150    base = 54/150

Remove that one family and the fine-tune WINS:

    run 7 without determinant = 278/510 = 54.5%
    base  without determinant = 252/510 = 49.4%

So the determinant family is the entire 24-problem deficit. The prompt
shape was fixed in run 7 and the corpus's determinant derivations are
arithmetically correct. What was left is the ARITHMETIC LOAD.

The corpus drew every matrix from [-4, 5]. The benchmark does not:

    subcat        size   median |entry|   p90   max
    determinant   3x3            8         9     9
    determinant   4x4            5         5     5
    determinant   5x5            5         5     5
    eigenvalue    3x3           22       179   240
    eigenvalue    4x4            8       102  1120
    eigenvalue    5x5            8         8     8
    nullity       3x3           12        22    24
    nullity       4x4           10        28    32
    nullity       5x5           11        25    27
    rank          3x3            6        12    18
    rank          4x4            6        28    31
    rank          5x5            8        20    32
    trace         3x3            6         9     9
    transpose     3x3            6         6     6
    matrix_vector 3x3            5         5     5
    matrix_power  3x3            4         4     4
    multiplication 3x3           4         9     9

A cofactor term is a_1j * M_1j. The corpus's median such term is 40; the
benchmark's is 168 - 5.2 times larger - and the corpus's median count of
terms over 100 is 0 against the benchmark's 1. At 5x5, 82 per cent of the
corpus's determinant derivations contain a fraction whose numerator is at
least 100, with the largest seen at 4009, while the benchmark's 5x5
determinants reach 9112.

The corpus taught a chain of exact arithmetic and then asked for a bigger
number than the chain was trained on. One slip loses the problem, and the
base model - never taught that chain - answers directly and lands more of
them.

This table is the fix's input: the corpus draws its matrices from the
benchmark's own magnitude range, per family and per size.

It is VERIFIED against the released files by
tests/test_entry_range_coverage.py, which compares these numbers with the
data, in both directions.
"""
from __future__ import annotations

import random

# The shape of the benchmark's entry distribution, per family and size.
#
# The benchmark's entries are NOT uniform on a range: they are mostly
# small with a TAIL of large ones.
#
#     rank        3x3   median 3, p75 5, p90 8,  p95 9,  max 18
#     rank        4x4   median 3, p75 5, p90 9,  p95 14, max 31
#     nullity     3x3   median 4, p75 8, p90 12, p95 16, max 24
#     nullity     4x4   median 4, p75 8, p90 12, p95 16, max 32
#     eigenvalue  3x3   median 6, p75 20, p90 80, p95 102, max 240
#     eigenvalue  4x4   median 3, p75 6, p90 20, p95 54, max 1120
#
# So the right thing to draw is not uniform on [-max, max] - that would
# make every problem as hard as the hardest one - but a distribution with
# that shape: mostly small, occasionally large.
#
# A mixture does it: with probability 1 - heavy, a small entry from
# [-small, small]; with probability heavy, a large one from
# [-large, large]. The corpus's old range [-4, 5] was the "small" part
# with no tail at all, and that is the whole defect.
#
# (family, size) -> (small, heavy_probability, large)
BENCHMARK_ENTRY_SHAPES: dict[str, dict[str, tuple[int, float, int]]] = {
    "determinant": {
        # The benchmark's 3x3 determinant entries: median 4, p75 6, p90 8,
        # max 9. That is close to uniform on [-9, 9] with a slight pull
        # toward zero, and the cofactor term a_1j * M_1j then has a median
        # of 60 and a p90 of 360. The corpus's old [-4, 5] gave a median
        # term of 20 - a third of the benchmark's.
        "3x3": (5, 0.75, 9),
        "4x4": (5, 0.10, 5),
        "5x5": (5, 0.10, 5),
    },
    "eigenvalue": {
        # The third number is the ceiling the corpus may reach, and it is
        # the benchmark's MAXIMUM for that size - not its p90. The p90 is
        # the bulk; the maximum is the reach. A corpus whose largest entry
        # is 116 against the benchmark's 1120 trains below the range it
        # has to cover, and the test that checks the range fails.
        #
        #     benchmark  3x3   median 6, p75 20, p90 80,  p95 102, max  240
        #                 4x4   median 3, p75  6, p90 20,  p95  54, max 1120
        #                 5x5   median 3, p75  4, p90  6,  p95   7, max    8
        "3x3": (6, 0.40, 240),
        "4x4": (5, 0.30, 1120),
        "5x5": (5, 0.10, 8),
    },
    "eigenvalue_numeric": {
        "3x3": (6, 0.40, 240),
        "4x4": (5, 0.30, 1120),
        "5x5": (5, 0.10, 8),
    },
    "nullity": {
        "3x3": (4, 0.45, 24),
        "4x4": (4, 0.40, 32),
        "5x5": (4, 0.25, 27),
    },
    "rank": {
        "3x3": (4, 0.40, 18),
        "4x4": (4, 0.30, 31),
        "5x5": (4, 0.20, 32),
    },
    "trace": {
        "3x3": (5, 0.25, 9),
        "4x4": (5, 0.20, 9),
        "5x5": (5, 0.15, 9),
    },
    "matrix_trace": {
        "3x3": (5, 0.25, 9),
        "4x4": (5, 0.20, 9),
        "5x5": (5, 0.15, 9),
    },
    "matrix_transpose": {
        "3x3": (5, 0.20, 6),
        "4x4": (5, 0.20, 9),
        "5x5": (5, 0.20, 9),
    },
    "transpose": {
        "3x3": (5, 0.20, 6),
        "4x4": (5, 0.20, 9),
        "5x5": (5, 0.20, 9),
    },
    "matrix_vector": {
        "3x3": (4, 0.15, 5),
        "4x4": (4, 0.15, 5),
        "5x5": (4, 0.15, 5),
    },
    "matrix_power": {
        "3x3": (4, 0.05, 4),
        "4x4": (3, 0.05, 3),
        "5x5": (3, 0.05, 3),
    },
    "matrix_multiply": {
        "3x3": (4, 0.30, 9),
        "4x4": (4, 0.05, 4),
        "5x5": (4, 0.05, 4),
    },
}

# The ranges the corpus used before this table existed. A family with no
# entry here falls back to these, so a family the benchmark does not have
# still gets a sane spread.
_OLD_RANGES: dict[str, tuple[int, int]] = {
    "determinant": (-4, 5),
    "eigenvalue": (-3, 4),
    "eigenvalue_numeric": (-3, 4),
    "nullity": (-3, 4),
    "rank": (-3, 4),
    "trace": (-4, 5),
    "matrix_trace": (-4, 5),
    "matrix_transpose": (-4, 5),
    "transpose": (-4, 5),
    "matrix_vector": (-2, 3),
    "matrix_power": (-3, 4),
    "matrix_multiply": (-3, 4),
}


def entry_shape(family: str, size: str = "") -> tuple[int, float, int]:
    """The (small, heavy_probability, large) shape for a family's entries.

    `size` is the matrix size as a string, "3x3" and so on. When the
    family has no entry for that size, the family's widest shape is used.
    """
    table = BENCHMARK_ENTRY_SHAPES.get(family)
    if not table:
        lo, hi = _OLD_RANGES.get(family, (-3, 4))
        return max(abs(lo), abs(hi)), 0.0, max(abs(lo), abs(hi))
    if size in table:
        return table[size]
    widest = max(table.items(), key=lambda kv: kv[1][2])
    return widest[1]


def draw_entry(rng: random.Random, family: str, size: str = "") -> int:
    """One matrix entry, drawn with the benchmark's shape.

    Mostly a small entry, sometimes a large one, with the benchmark's own
    proportions. The sign is even, because the benchmark's matrices are
    symmetric about zero in sign.
    """
    small, heavy, large = entry_shape(family, size)
    magnitude = large if rng.random() < heavy else small
    value = rng.randint(-magnitude, magnitude)
    return value
