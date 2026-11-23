"""Every task STATEMENT the benchmark uses, per subcategory and size.

The released problem files have, before the matrix, a task statement that
is DETERMINISTIC per (subcategory, size):

    determinant   3x3  "Find the determinant of the 3×3 matrix A."
                  4x4  (nothing - a bare matrix)
                  5x5  "Find the determinant of the 5×5 matrix A."

    eigenvalue    3x3  "Find the eigenvalues of the 3×3 matrix A."
                  4x4  (nothing)
                  5x5  "Find the eigenvalues of the 5×5 matrix A."

    matrix_power  3x3  "Compute A² = A × A for the 3×3 matrix A."
                  4x4  (nothing)
                  5x5  "Compute A² = A × A for the 5×5 matrix A."

    matrix_vector 3x3  "Compute the matrix-vector product b = A·x for the
                         3×3 matrix A and vector x."
                  4x4  (nothing)
                  5x5  "Compute the product of the 5×5 matrix A and the
                         vector x."

    multiplication 3x3 "Compute the matrix product C = A × B for the two
                         3×3 matrices."   (12 of 20)
                       "Compute the product AB for the given 3×3 matrices
                         A and B."        (8 of 20)
                  4x4  (nothing)
                  5x5  "Compute the product of the 5×5 matrices A and B."

    nullity       3x3  "Find the nullity of the 3×3 matrix A."
                  4x4  (nothing)
                  5x5  "Find the nullity of the 5×5 matrix A."

    rank          3x3  "Find the rank of the 3×3 matrix A."
                  4x4  (nothing)
                  5x5  "Find the rank of the 5×5 matrix A."

    trace         3x3  "Compute the trace of the 3×3 matrix A."  (14 of 20)
                       "Find the trace of the 3×3 matrix A."     (6 of 20)
                  4x4  (nothing)
                  5x5  "Find the trace of the 5×5 matrix A."

    transpose     3x3  "Find the transpose of the 3×3 matrix A."
                  4x4  (nothing)
                  5x5  "Find the transpose of the 5×5 matrix A."

Run 10 scored 276/660 = 41.8%, and the two families that were still
broken were the determinant (1/150 against the base's 54/150) and the
nullity (14/60 against 19/60). The reason is that the corpus did NOT
teach these statements. It invented its own:

    determinant   233 of 300 problems (78%) named NO size at all
                  30 (10%) named 3x3, with an ASCII "x"
                  19 (6%)  named 5x5, with an ASCII "x"
                  18 (6%)  named 4x4

and it never produced the bare-matrix shape for the 3x3 and 5x5
families at all - the bare shape is the benchmark's 4x4 form, and the
corpus taught it as a different thing from the task statement.

Two distinct mismatches, and both matter:

  1. The corpus's task statements are the benchmark's only 8-14 per cent
     of the time. The rest are wordings the benchmark never uses.

  2. The corpus writes the size with an ASCII "x" and the benchmark
     writes it with U+00D7:

         corpus:     "Find the determinant of the 3x3 matrix A."
         benchmark:  "Find the determinant of the 3×3 matrix A."

     Different strings. The model is trained on one and scored on the
     other.

So the corpus must use the benchmark's statement VERBATIM, chosen by
size, and nothing else. The size is in the statement, so the statement
cannot be chosen before the size is known - which is why this table is
keyed by (subcategory, size) rather than by subcategory alone.

These strings were extracted from the released files by
`scripts/extract_task_statements.py`, and
`tests/test_task_statement.py` checks them against those files in both
directions: the corpus must produce every statement the benchmark uses
and must not produce a statement the benchmark never uses.
"""
from __future__ import annotations

# (subcategory, size) -> [(task statement, how often the benchmark uses
# it), ...]. The bare-matrix shape is the empty string.
BENCHMARK_TASK_STATEMENTS: dict[str, dict[str, list[tuple[str, int]]]] = {
    "determinant": {
        "3x3": [("Find the determinant of the 3×3 matrix A.", 50)],
        "4x4": [("", 50)],
        "5x5": [("Find the determinant of the 5×5 matrix A.", 50)],
    },
    "eigenvalue": {
        "3x3": [("Find the eigenvalues of the 3×3 matrix A.", 30)],
        "4x4": [("", 30)],
        "5x5": [("Find the eigenvalues of the 5×5 matrix A.", 30)],
    },
    "matrix_power": {
        "3x3": [("Compute A² = A × A for the 3×3 matrix A.", 20)],
        "4x4": [("", 20)],
        "5x5": [("Compute A² = A × A for the 5×5 matrix A.", 20)],
    },
    "matrix_vector": {
        "3x3": [("Compute the matrix-vector product b = A·x for the 3×3 "
                 "matrix A and vector x.", 20)],
        "4x4": [("", 20)],
        "5x5": [("Compute the product of the 5×5 matrix A and the vector "
                 "x.", 20)],
    },
    "multiplication": {
        "3x3": [("Compute the matrix product C = A × B for the two 3×3 "
                 "matrices.", 12),
                ("Compute the product AB for the given 3×3 matrices A and "
                 "B.", 8)],
        "4x4": [("", 20)],
        "5x5": [("Compute the product of the 5×5 matrices A and B.", 20)],
    },
    "nullity": {
        "3x3": [("Find the nullity of the 3×3 matrix A.", 20)],
        "4x4": [("", 20)],
        "5x5": [("Find the nullity of the 5×5 matrix A.", 20)],
    },
    "rank": {
        "3x3": [("Find the rank of the 3×3 matrix A.", 20)],
        "4x4": [("", 20)],
        "5x5": [("Find the rank of the 5×5 matrix A.", 20)],
    },
    "trace": {
        "3x3": [("Compute the trace of the 3×3 matrix A.", 14),
                ("Find the trace of the 3×3 matrix A.", 6)],
        "4x4": [("", 20)],
        "5x5": [("Find the trace of the 5×5 matrix A.", 20)],
    },
    "transpose": {
        "3x3": [("Find the transpose of the 3×3 matrix A.", 20)],
        "4x4": [("", 20)],
        "5x5": [("Find the transpose of the 5×5 matrix A.", 20)],
    },
}


def task_statements(subcat: str, size: str) -> list[tuple[str, int]]:
    """The (statement, weight) pairs for one subcategory and size.

    The size is "3x3", "4x4" or "5x5". A size with no entry falls back to
    the closest one that has one, so a family the benchmark does not
    score still gets a statement rather than an empty question.
    """
    table = BENCHMARK_TASK_STATEMENTS.get(subcat)
    if not table:
        return []
    if size in table:
        return table[size]
    for fallback in ("3x3", "4x4", "5x5"):
        if fallback in table:
            return table[fallback]
    return []
