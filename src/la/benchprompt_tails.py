"""Every prompt TAIL the benchmark uses, per subcategory.

The released problem files have, after the matrix, up to three different
tails: a bare one (the 4x4 file), the plain closing instruction, and a
method instruction followed by a different closing instruction. The corpus
only ever produced the plain one.

    determinant      82x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  18x  "Use cofactor expansion along the first row." +
                        "Show all intermediate steps. Put your final
                        numerical answer inside \\boxed{}."
                  50x  (nothing - the 4x4 file)

    eigenvalue       36x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  24x  "Compute the characteristic polynomial
                        det(A - lambda I) = 0 and solve for lambda." +
                        "Show all steps. List all eigenvalues inside \\boxed{}."
                  30x  (nothing - the 4x4 file)

    matrix_power     20x  "Multiply A by itself. Show all row-by-column
                           steps." +
                        "Put the final 3x3 result matrix inside \\boxed{}."
                  20x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  20x  (nothing - the 4x4 file)

    matrix_vector    20x  "Compute each entry of b by taking the dot product
                           of each row of A with x." +
                        "Show your steps. Put the final vector b inside \\boxed{}."
                  20x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  20x  (nothing - the 4x4 file)

    multiplication   28x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  12x  "Show your row-by-column computation steps clearly." +
                        "Put the final 3x3 result matrix inside \\boxed{}."
                  20x  (nothing - the 4x4 file)

    nullity          24x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  16x  "Recall: nullity = number of columns - rank." +
                        "First find the rank using row reduction, then
                        compute the nullity." +
                        "Show your steps. Put your final numerical answer
                        inside \\boxed{}."
                  20x  (nothing - the 4x4 file)

    rank             26x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  14x  "Use row reduction to find the number of non-zero rows
                        in row echelon form." +
                        "Show your row operations. Put your final numerical
                        answer inside \\boxed{}."
                  20x  (nothing - the 4x4 file)

    trace            26x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  14x  "The trace is the sum of the diagonal elements." +
                        "Show your steps. Put your final numerical answer
                        inside \\boxed{}."
                  20x  (nothing - the 4x4 file)

    transpose        20x  "Show all intermediate steps. Put your final
                           numerical answer inside \\boxed{}."
                  20x  "The transpose A^T is obtained by swapping rows and
                        columns." +
                        "Write out A^T explicitly. Put the final matrix
                        inside \\boxed{}."
                  20x  (nothing - the 4x4 file)

The 4x4 file is one third of the benchmark and every one of its problems
is a bare matrix. The harness reconstructs the task from the subcategory
label, so the corpus has to have examples of that shape too.

The counts are the benchmark's own, so the corpus's mix of tails matches
the mix the model is scored on.
"""
from __future__ import annotations

# The plain closing instruction, shared by most tails.
_PLAIN = "Show all intermediate steps. Put your final numerical answer inside \\boxed{}."

# A tail is (method instruction, closing instruction). The method
# instruction is "" when the tail has none, and the closing instruction is
# "" for the bare 4x4 shape.
#
# The counts are out of the subcategory's 150 or 60 problems.
PROMPT_TAILS: dict[str, list[tuple[str, str, int]]] = {
    "determinant": [
        ("", _PLAIN, 82),
        ("Use cofactor expansion along the first row.", _PLAIN, 18),
        ("", "", 50),
    ],
    "eigenvalue": [
        ("", _PLAIN, 36),
        ("Compute the characteristic polynomial det(A − λI) = 0 and solve "
         "for λ.", "Show all steps. List all eigenvalues inside \\boxed{}.", 24),
        ("", "", 30),
    ],
    "matrix_power": [
        ("Multiply A by itself. Show all row-by-column steps.",
         "Put the final 3×3 result matrix inside \\boxed{}.", 20),
        ("", _PLAIN, 20),
        ("", "", 20),
    ],
    "matrix_vector": [
        ("Compute each entry of b by taking the dot product of each row of A "
         "with x.", "Show your steps. Put the final vector b inside \\boxed{}.", 20),
        ("", _PLAIN, 20),
        ("", "", 20),
    ],
    "multiplication": [
        ("", _PLAIN, 28),
        ("Show your row-by-column computation steps clearly.",
         "Put the final 3×3 result matrix inside \\boxed{}.", 12),
        ("", "", 20),
    ],
    "nullity": [
        ("", _PLAIN, 24),
        ("Recall: nullity = number of columns − rank.\n\nFirst find the rank "
         "using row reduction, then compute the nullity.",
         "Show your steps. Put your final numerical answer inside \\boxed{}.", 16),
        ("", "", 20),
    ],
    "rank": [
        ("", _PLAIN, 26),
        ("Use row reduction to find the number of non-zero rows in row "
         "echelon form.",
         "Show your row operations. Put your final numerical answer inside "
         "\\boxed{}.", 14),
        ("", "", 20),
    ],
    "trace": [
        ("", _PLAIN, 26),
        ("The trace is the sum of the diagonal elements.",
         "Show your steps. Put your final numerical answer inside \\boxed{}.", 14),
        ("", "", 20),
    ],
    "transpose": [
        ("", _PLAIN, 20),
        ("The transpose A^T is obtained by swapping rows and columns.",
         "Write out A^T explicitly. Put the final matrix inside \\boxed{}.", 20),
        ("", "", 20),
    ],
}
