"""Augmentation.

The failure mode this exists to prevent: a model trained on one phrasing of
one shape of problem learns to recognise that shape, not to do the mathematics.
Allen-Zhu and Li's controlled experiments show the opposite extreme - knowledge
memorised without augmentation is not extractable at all, and scores zero
however long you fine-tune it.

Two kinds of augmentation are applied, and both are safe because the ground
truth is recomputed from whatever actually appears in the question:

1. **Paraphrase.** The same mathematics asked several ways. This is the
   augmentation the literature demands.
2. **Relabel.** A row permutation, a column permutation, or a similarity
   transform of the matrix. A similarity transform preserves the spectrum;
   a row permutation does not, and in both cases the answer is recomputed, so
   the data can never disagree with the question.
"""
from __future__ import annotations

import random

# Question templates per family. Each takes the core statement of the problem
# and asks it a different way.
TEMPLATES: dict[str, list[str]] = {
    "determinant": [
        "Compute the determinant of {stem}.",
        "What is det of the matrix {stem}?",
        "For the matrix {stem}, find its determinant.",
        "Find the value of the determinant of {stem}.",
        "Evaluate det({stem}).",
    ],
    "solve_system": [
        "Solve the linear system {stem} for the unknown vector.",
        "Find the vector x that satisfies {stem}.",
        "Given {stem}, determine x.",
        "Work out the solution of {stem}.",
        "Solve {stem}.",
    ],
    "eigenvalues": [
        "Find the eigenvalues of {stem}.",
        "What are the eigenvalues of the matrix {stem}?",
        "Determine the spectrum of {stem}.",
        "Compute the eigenvalues of {stem}.",
        "Which eigenvalues does {stem} have?",
    ],
    "diagonalize": [
        "The matrix {stem} is diagonalisable. Give the diagonal matrix D in a "
        "diagonalisation A = P D P^{{-1}}, and state the eigenvalues on the diagonal.",
        "For {stem}, find the diagonal form D in A = P D P^{{-1}}.",
        "Diagonalise {stem}: give the diagonal matrix D in A = P D P^{{-1}}.",
    ],
    "orthonormal_basis": [
        "Apply Gram-Schmidt to {stem} to obtain an orthonormal basis of the span.",
        "Produce an orthonormal basis for the span of {stem} using Gram-Schmidt.",
        "Using Gram-Schmidt, find an orthonormal set spanning the space generated "
        "by {stem}.",
        "Give an orthonormal basis of the subspace spanned by {stem}.",
    ],
    "projection": [
        "Let {stem}. Project the first vector onto the line spanned by the "
        "second, and give the projected vector and the residual.",
        "For {stem}, compute the orthogonal projection of the first vector onto "
        "the second, and the residual.",
        "With {stem}, find the projection of v onto w and the residual v - proj.",
    ],
    "least_squares": [
        "Find the least-squares solution of {stem} (the system is overdetermined).",
        "The system {stem} has more equations than unknowns. Compute its "
        "least-squares solution.",
        "Solve {stem} in the least-squares sense.",
    ],
    "lu_factor": [
        "Compute an LU factorisation of {stem}: give L and U with L U = A.",
        "Factorise {stem} as L U.",
        "Give the lower and upper triangular factors L and U for {stem}.",
    ],
    "svd": [
        "Compute the singular values of {stem}. Give them as a set.",
        "What are the singular values of {stem}?",
        "Find the singular values of the matrix {stem}.",
    ],
    "condition_number": [
        "Compute the 2-norm condition number of {stem}. Give it as a decimal.",
        "What is the 2-norm condition number of {stem}?",
        "Find cond_2 of {stem}.",
    ],
    "rank_nullspace": [
        "Determine the rank and the nullity (the dimension of the null space) of "
        "{stem}.",
        "Give the rank and the nullity of {stem}.",
        "What are rank({stem}) and the dimension of its null space?",
    ],
    "linear_transform": [
        "Let T be the linear transformation of R^2 with matrix {stem}. Compute "
        "T(v) for the given v, and describe the image of the unit square under T.",
        "With the transformation matrix {stem}, find the image of the given "
        "vector, and describe the image of the unit square.",
        "Apply the linear map with matrix {stem} to the given vector.",
    ],
}


def paraphrase(family: str, stem: str, rng: random.Random) -> str:
    """Ask the same question a different way."""
    options = TEMPLATES.get(family)
    if not options:
        return stem
    return rng.choice(options).format(stem=stem)


def permute_rows(rng: random.Random, n: int) -> list[int]:
    order = list(range(n))
    rng.shuffle(order)
    return order


def similarity_transform(rng: random.Random, A):
    """Return P^-1 A P for a random permutation matrix P.

    A similarity transform preserves the spectrum exactly, so the eigenvalues
    are unchanged while the matrix - and therefore the question - is
    different. The entries stay integral because P is a permutation matrix.
    """
    import sympy as sp
    n = A.rows
    order = permute_rows(rng, n)
    P = sp.zeros(n, n)
    for i, j in enumerate(order):
        P[i, j] = 1
    return sp.Matrix(P.inv() * A * P)
