"""Every prompt TAIL the benchmark uses, per subcategory AND SIZE.

The released problem files have, after the matrix, up to three different
tails: a bare one (the 4x4 file), the plain closing instruction, and a
method instruction followed by a different closing instruction.

The tails are SIZE-SPECIFIC, which the corpus's flat table did not
capture. Measured from the released files:

    determinant   3x3   32x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."
                       18x  "Use cofactor expansion along the first row." +
                            "Show all intermediate steps. Put your final
                            numerical answer inside \\boxed{}."
                  4x4   50x  (nothing)
                  5x5   50x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

    eigenvalue    3x3   24x  "Compute the characteristic polynomial
                               det(A − λI) = 0 and solve for λ." +
                            "Show all steps. List all eigenvalues inside
                            \\boxed{}."
                        6x  "Show all intermediate steps. Put your final
                             numerical answer inside \\boxed{}."
                  4x4   30x  (nothing)
                  5x5   30x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

    matrix_power  3x3   20x  "Multiply A by itself. Show all row-by-column
                               steps." + "Put the final 3×3 result matrix
                               inside \\boxed{}."
                  4x4   20x  (nothing)
                  5x5   20x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

    matrix_vector 3x3   20x  "Compute each entry of b by taking the dot
                               product of each row of A with x." + "Show
                               your steps. Put the final vector b inside
                               \\boxed{}."
                  4x4   20x  (nothing)
                  5x5   20x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

    multiplication 3x3  12x  "Show your row-by-column computation steps
                               clearly." + "Put the final 3×3 result matrix
                               inside \\boxed{}."
                        8x  "Show all intermediate steps. Put your final
                             numerical answer inside \\boxed{}."
                  4x4   20x  (nothing)
                  5x5   20x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

    nullity       3x3   16x  "Recall: nullity = number of columns − rank." +
                            "First find the rank using row reduction, then
                            compute the nullity." + "Show your steps. Put
                            your final numerical answer inside \\boxed{}."
                        4x  "Show all intermediate steps. Put your final
                             numerical answer inside \\boxed{}."
                  4x4   20x  (nothing)
                  5x5   20x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

    rank          3x3   14x  "Use row reduction to find the number of
                               non-zero rows in row echelon form." + "Show
                               your row operations. Put your final numerical
                               answer inside \\boxed{}."
                        6x  "Show all intermediate steps. Put your final
                             numerical answer inside \\boxed{}."
                  4x4   20x  (nothing)
                  5x5   20x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

    trace         3x3   14x  "The trace is the sum of the diagonal
                               elements." + "Show your steps. Put your final
                               numerical answer inside \\boxed{}."
                        6x  "Show all intermediate steps. Put your final
                             numerical answer inside \\boxed{}."
                  4x4   20x  (nothing)
                  5x5   20x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

    transpose     3x3   20x  "The transpose A^T is obtained by swapping rows
                               and columns." + "Write out A^T explicitly.
                               Put the final matrix inside \\boxed{}."
                  4x4   20x  (nothing)
                  5x5   20x  "Show all intermediate steps. Put your final
                               numerical answer inside \\boxed{}."

TWO THINGS THIS TABLE GOT WRONG BEFORE, AND BOTH COST POINTS:

1. **The method instruction and the closing instruction are ONE
   paragraph, separated by a single newline.** The nullity entry had a
   blank line where the benchmark has a newline, and the corpus's
   question assembly put the two in separate paragraphs - so a prompt
   with a method instruction had five paragraphs where the benchmark's
   has four.

2. **The tail depends on the SIZE.** The transpose's 3x3 tail is always
   "The transpose A^T is obtained by swapping rows and columns." and its
   5x5 tail is always the plain one; a flat table that mixed them gave
   the model a closing instruction the benchmark never sends at that
   size.

These strings were extracted from the released files by
`scripts/extract_prompt_parts.py`, and
`tests/test_tail_table_matches_benchmark.py` checks them against those
files in both directions.
"""
from __future__ import annotations

import random

_PLAIN = ("Show all intermediate steps. Put your final numerical answer "
          "inside \\boxed{}.")

# subcategory -> size -> [(method, closing, count), ...]
PROMPT_TAILS_BY_SIZE: dict[str, dict[str, list[tuple[str, str, int]]]] = {
    "determinant": {
        "3x3": [("", _PLAIN, 32),
                ("Use cofactor expansion along the first row.", _PLAIN, 18)],
        "4x4": [("", "", 50)],
        "5x5": [("", _PLAIN, 50)],
    },
    "eigenvalue": {
        "3x3": [("Compute the characteristic polynomial det(A − λI) = 0 and "
                 "solve for λ.", "Show all steps. List all eigenvalues inside "
                 "\\boxed{}.", 24),
                ("", _PLAIN, 6)],
        "4x4": [("", "", 30)],
        "5x5": [("", _PLAIN, 30)],
    },
    "matrix_power": {
        "3x3": [("Multiply A by itself. Show all row-by-column steps.",
                 "Put the final 3×3 result matrix inside \\boxed{}.", 20)],
        "4x4": [("", "", 20)],
        "5x5": [("", _PLAIN, 20)],
    },
    "matrix_vector": {
        "3x3": [("Compute each entry of b by taking the dot product of each "
                 "row of A with x.",
                 "Show your steps. Put the final vector b inside \\boxed{}.", 20)],
        "4x4": [("", "", 20)],
        "5x5": [("", _PLAIN, 20)],
    },
    "multiplication": {
        "3x3": [("Show your row-by-column computation steps clearly.",
                 "Put the final 3×3 result matrix inside \\boxed{}.", 12),
                ("", _PLAIN, 8)],
        "4x4": [("", "", 20)],
        "5x5": [("", _PLAIN, 20)],
    },
    "nullity": {
        "3x3": [("Recall: nullity = number of columns − rank.\nFirst find the "
                 "rank using row reduction, then compute the nullity.",
                 "Show your steps. Put your final numerical answer inside "
                 "\\boxed{}.", 16),
                ("", _PLAIN, 4)],
        "4x4": [("", "", 20)],
        "5x5": [("", _PLAIN, 20)],
    },
    "rank": {
        "3x3": [("Use row reduction to find the number of non-zero rows in "
                 "row echelon form.",
                 "Show your row operations. Put your final numerical answer "
                 "inside \\boxed{}.", 14),
                ("", _PLAIN, 6)],
        "4x4": [("", "", 20)],
        "5x5": [("", _PLAIN, 20)],
    },
    "trace": {
        "3x3": [("The trace is the sum of the diagonal elements.",
                 "Show your steps. Put your final numerical answer inside "
                 "\\boxed{}.", 14),
                ("", _PLAIN, 6)],
        "4x4": [("", "", 20)],
        "5x5": [("", _PLAIN, 20)],
    },
    "transpose": {
        "3x3": [("The transpose A^T is obtained by swapping rows and columns.",
                 "Write out A^T explicitly. Put the final matrix inside "
                 "\\boxed{}.", 20)],
        "4x4": [("", "", 20)],
        "5x5": [("", _PLAIN, 20)],
    },
}


def tails_for(subcat: str, size: str = "") -> list[tuple[str, str, int]]:
    """The (method, closing, count) tails for a subcategory and size.

    A size with no entry falls back to the 3x3 one, so a family the
    benchmark does not score still gets a sane tail.
    """
    table = PROMPT_TAILS_BY_SIZE.get(subcat)
    if not table:
        return []
    if size in table:
        return table[size]
    for fallback in ("3x3", "4x4", "5x5"):
        if fallback in table:
            return table[fallback]
    return []


# The flat view, kept for the families the benchmark does not score and
# for the checks that want every tail of a subcategory at once. The
# weights are the released files' own counts summed over the sizes.
PROMPT_TAILS: dict[str, list[tuple[str, str, int]]] = {}
for _sub, _by_size in PROMPT_TAILS_BY_SIZE.items():
    _merged: dict[tuple[str, str], int] = {}
    for _size, _entries in _by_size.items():
        for _method, _closing, _count in _entries:
            _key = (_method, _closing)
            _merged[_key] = _merged.get(_key, 0) + _count
    PROMPT_TAILS[_sub] = [(*_key, _count)
                          for _key, _count in _merged.items()]


def decoration_rng(rng: random.Random) -> random.Random:
    """A stream for the PROMPT DECORATION, separate from the problem's.

    The task statement and the tail are prompt decoration; the matrix is
    the problem. They used to share one stream, which meant that every
    change to the prompt contract - a new tail, a new weight, a
    different `randrange` bound - consumed a different number of draws
    and silently rewrote every matrix that came after it.

    The effect is invisible in the output: the corpus still looks
    plausible, it is just a different corpus. Run 11's task-statement
    change moved the corpus's largest eigenvalue entry from 1120 to 1118
    and the magnitude test failed with a number that looked like a
    regression and was actually a reshuffle.

    HOW THE SEPARATION IS DONE. The decoration's seed is ONE draw from
    the problem's stream, taken once per problem. After that the two
    streams are independent, so the decoration can make as many draws as
    it likes without touching the matrix.

    The seed is a draw and not a hash of the stream's state, because a
    hash of the state does not advance: every decoration_rng() call on
    an unchanged stream returned the SAME seed, so the corpus's task
    statements collapsed to one per family and the unscored families
    lost their wording variety entirely. One draw per problem keeps
    the variety AND keeps the problem's draws out of the decoration's
    way.

    Deriving from the generator's state and not from a counter: a
    counter would let the decoration's stream depend on how many
    problems came before it in the schedule, which is exactly the
    coupling that has to be broken.
    """
    return random.Random(rng.getrandbits(64))
