"""The benchmark's rank mix, per size, for the rank and nullity families.

THE RANK OF A MATRIX IS THE FREQUENCY OF AN ANSWER, NOT A SHAPE. The
benchmark's rank and nullity matrices have a spread of ranks, and the
full-rank case is the common one:

    benchmark rank  (size, rank)    benchmark nullity (size, rank)
      3x3  1: 2  2: 8  3: 10          3x3  1: 12 2: 7  3: 1
      4x4  2: 4  3: 7  4: 9           4x4  1: 3  2: 5  3: 7  4: 5
      5x5  2: 3  3: 4  4: 6  5: 7     5x5  2: 2  3: 5  4: 7  5: 6

The corpus drew its rank from 1 to n-1, so a full-rank matrix was
IMPOSSIBLE - not rare, never. Its distribution was the benchmark's
shifted one rank down, and the rank family (27 of 60 against the base
model's 21) and the nullity family (14 of 60 against the base model's
19) were the two families that LOST to the base model.

These weights were extracted from the released files by
`scripts/extract_prompt_parts.py`, and `tests/test_rank_coverage.py`
checks the corpus against the files in both directions.
"""
from __future__ import annotations

# subcategory -> size -> [(rank, weight), ...]
BENCHMARK_RANK_MIX: dict[str, dict[str, list[tuple[int, int]]]] = {
    "rank": {
        "3x3": [(1, 2), (2, 8), (3, 10)],
        "4x4": [(2, 4), (3, 7), (4, 9)],
        "5x5": [(2, 3), (3, 4), (4, 6), (5, 7)],
    },
    "nullity": {
        "3x3": [(1, 12), (2, 7), (3, 1)],
        "4x4": [(1, 3), (2, 5), (3, 7), (4, 5)],
        "5x5": [(2, 2), (3, 5), (4, 7), (5, 6)],
    },
}

# the family's own name in the corpus
FAMILY_TO_SUBCAT = {"rank": "rank", "nullity": "nullity"}


def rank_mix(family: str, size: str) -> list[tuple[int, int]]:
    """The (rank, weight) pairs for a family and size.

    A size with no entry falls back to the 3x3 one, so a family at a
    size the benchmark does not cover still gets a sane mix.
    """
    subcat = FAMILY_TO_SUBCAT.get(family, family)
    table = BENCHMARK_RANK_MIX.get(subcat)
    if not table:
        return []
    if size in table:
        return table[size]
    for fallback in ("3x3", "4x4", "5x5"):
        if fallback in table:
            return table[fallback]
    return []
