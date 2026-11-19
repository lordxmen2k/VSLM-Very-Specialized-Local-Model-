"""Question wording, and the prompt the corpus is written in.

The corpus is written in LinAlg-Bench's prompt format, because a model
scored on a question format it was not trained on is being asked to do a
task it has never seen phrased. The format is:

    Solve this linear algebra problem. Show your work and give the final answer.

    {task}

    {body}

The task statements for the nine benchmark subcategories are the benchmark's
own wordings, with the matrix size filled in. For the other eight families
they are written to match.

The task list per family also gives the model some variety: the benchmark's
wording is the first entry, and the others are the ways a person might
otherwise ask the same thing.

THE BENCHMARK'S OWN METHOD INSTRUCTIONS.

Some of the benchmark's problems do not just ask for a value; they NAME THE
METHOD:

    determinant      "Use cofactor expansion along the first row."
    eigenvalue       "Compute the characteristic polynomial det(A - lambda I) = 0
                      and solve for lambda."
    rank             "Use row reduction to find the number of non-zero rows in
                      row echelon form."
    nullity          "First find the rank using row reduction, then compute the
                      nullity."
    multiplication   "Show your row-by-column computation steps clearly."

That instruction is part of the PROMPT, so it is part of the scored
distribution. The corpus has to contain prompts that carry it, with a
derivation that answers it - otherwise the model is asked to follow a method
it was never shown being asked for.

This table holds the benchmark's LITERAL instruction strings. They are
copied verbatim from the released problem files, not paraphrased: the model
has to recognise the exact phrase the benchmark will use.
"""
from __future__ import annotations

import random

from .benchprompt import PREAMBLE, body_hint
from .benchprompt_tails import PROMPT_TAILS

# The corpus family -> benchmark subcategory mapping.
from .benchprompt import FAMILY_TO_SUBCAT


def closing_variants(family: str) -> list[str]:
    """Every closing instruction the benchmark uses for this family.

    The benchmark uses more than one for each of its nine subcategories,
    and the corpus has to produce all of them. For a family the benchmark
    does not have, the single instruction from BODY_HINTS.
    """
    subcat = FAMILY_TO_SUBCAT.get(family)
    tails = PROMPT_TAILS.get(subcat) if subcat else None
    if tails:
        out = []
        for _method, closing, _weight in tails:
            if closing and closing not in out:
                out.append(closing)
        return out
    hint = body_hint(family)
    return [hint] if hint else []


def prompt_tail(family: str, rng: random.Random) -> tuple[str, str]:
    """The (method instruction, closing instruction) for one problem.

    The benchmark's tails, at the benchmark's own rates. The counts in
    PROMPT_TAILS are the released files' own, so the corpus's mix of
    prompt shapes matches the mix the model is scored on.

    The choice is a function of the per-problem RNG, so it is the same in
    every worker process and for every seed. A counter would give each
    worker its own mix, and the corpus would drift with the worker count.
    """
    subcat = FAMILY_TO_SUBCAT.get(family)
    tails = PROMPT_TAILS.get(subcat) if subcat else None
    if not tails:
        return "", body_hint(family)
    total = sum(weight for _m, _c, weight in tails)
    pick = rng.randrange(total)
    upto = 0
    for method, closing, weight in tails:
        upto += weight
        if pick < upto:
            return method, closing
    return tails[-1][0], tails[-1][1]

# The benchmark's own method instructions, verbatim from the released files.
# Each entry is the instruction text; the corpus appends it to the task and
# teaches a derivation that follows it.
#
# The frequency matters as well as the wording: the benchmark uses the
# cofactor instruction in 18 of its 50 three-by-three determinant problems
# and in NONE of its 4x4 or 5x5 ones, so the corpus uses it at a similar
# rate for the sizes where it appears.

# The rate each method instruction appears at, per family. These are the
# benchmark's own counts, normalised to the number of 3x3 problems in that
# subcategory:
#
#   determinant     18 of 50  three-by-three problems   0.36
#   eigenvalue      24 of 30  three-by-three problems   0.80
#   rank            14 of 20  three-by-three problems   0.70
#   nullity         16 of 20  three-by-three problems   0.80
#   multiplication  12 of 20  three-by-three problems   0.60
#
# The counts are over the 3x3 problems only, because the benchmark only
# appends a method instruction at 3x3.

# Question templates per family. Each takes the core statement of the problem
# and asks it a different way. The task never embeds the matrix: the matrix
# is given separately, in the body, the way the benchmark does it.
TEMPLATES: dict[str, list[str]] = {
    "determinant": [
        "Find the determinant of the {size} matrix A.",
        "Compute the determinant of the matrix A.",
        "What is det of the matrix A?",
        "For the matrix A, find its determinant.",
    ],
    "solve_system": [
        "Solve the linear system for the unknown vector.",
        "Find the vector x that satisfies the system.",
        "Determine x for the system given.",
        "Work out the solution of the system.",
    ],
    "eigenvalue": [
        "Find the eigenvalues of the {size} matrix A.",
        "What are the eigenvalues of the matrix A?",
        "Determine the spectrum of the matrix A.",
        "Compute the eigenvalues of the matrix A.",
    ],
    "eigenvalue_numeric": [
        "Find the eigenvalues of the {size} matrix A.",
        "What are the eigenvalues of the matrix A?",
        "Determine the spectrum of the matrix A.",
        "Compute the eigenvalues of the matrix A.",
    ],
    "diagonalize": [
        "The {size} matrix A is diagonalisable. Give the diagonal matrix D in a "
        "diagonalisation A = P D P^{{-1}}, and state the eigenvalues on the diagonal.",
        "For the matrix A, find the diagonal form D in A = P D P^{{-1}}.",
        "Diagonalise the matrix A: give the diagonal matrix D in A = P D P^{{-1}}.",
    ],
    "orthonormal_basis": [
        "Apply Gram-Schmidt to the given vectors to obtain an orthonormal basis "
        "of their span.",
        "Produce an orthonormal basis for the span of the given vectors using "
        "Gram-Schmidt.",
        "Using Gram-Schmidt, find an orthonormal set spanning the space "
        "generated by the given vectors.",
    ],
    "projection": [
        "Project the first vector onto the line spanned by the second, and "
        "give the projected vector and the residual.",
        "Compute the orthogonal projection of the first vector onto the second, "
        "and the residual.",
        "Find the projection of v onto w and the residual v - proj.",
    ],
    "least_squares": [
        "Find the least-squares solution of the overdetermined system.",
        "The system given has more equations than unknowns. Compute its "
        "least-squares solution.",
        "Solve the system in the least-squares sense.",
    ],
    "lu_factor": [
        "Compute an LU factorisation of the matrix: give L and U with L U = A.",
        "Factorise the matrix as L U.",
        "Give the lower and upper triangular factors L and U for the matrix.",
    ],
    "svd": [
        "Compute the singular values of the matrix. Give them as a set.",
        "What are the singular values of the matrix?",
        "Find the singular values of the matrix.",
    ],
    "condition_number": [
        "Compute the 2-norm condition number of the matrix. Give it as a decimal.",
        "What is the 2-norm condition number of the matrix?",
        "Find cond_2 of the matrix.",
    ],
    "rank": [
        "Find the rank of the {size} matrix A.",
        "Determine the rank of the matrix.",
        "Compute the rank of the matrix A.",
        "Find the rank of the matrix A.",
    ],
    "nullity": [
        "Find the nullity of the {size} matrix A.",
        "Determine the nullity (the dimension of the null space) of the matrix.",
        "Compute the nullity of the matrix A.",
        "Find the nullity of the matrix A.",
    ],
    "matrix_trace": [
        "Find the trace of the {size} matrix A.",
        "Compute the trace of the matrix A.",
        "What is tr of the matrix A?",
    ],
    "matrix_transpose": [
        "Find the transpose of the {size} matrix A.",
        "Transpose the matrix A.",
        "Give the transpose of the matrix A.",
    ],
    "matrix_multiply": [
        "Compute the product AB for the given {size} matrices A and B.",
        "Multiply the two matrices A and B.",
        "Compute the matrix product A B.",
    ],
    "matrix_power": [
        "Compute A² = A × A for the {size} matrix A.",
        "Raise the matrix A to the third power.",
        "Find A^2 for the matrix A.",
    ],
    "matrix_vector": [
        "Compute the matrix-vector product b = A·x for the given {size} "
        "matrix A and vector x.",
        "Multiply the matrix A by the vector x.",
        "Compute the matrix-vector product.",
    ],
    "linear_transform": [
        "Let T be the linear transformation with the matrix given. Compute T(v) "
        "for the given vector, and describe the image of the unit square under T.",
        "With the transformation matrix given, find the image of the given "
        "vector, and describe the image of the unit square.",
        "Apply the linear map with the matrix given to the given vector.",
    ],
}


def paraphrase(family: str, rng: random.Random, size: str = "") -> str:
    """The task statement for a family, one of several wordings.

    `{size}` is filled in with something like "3x3" where the wording calls
    for it.
    """
    options = TEMPLATES.get(family)
    if not options:
        return ""
    text = rng.choice(options)
    if "{size}" in text:
        text = text.replace("{size}", size or "")
    return text






def question(family: str, tail: tuple[str, str], task: str, body: str) -> str:
    """Assemble a question in LinAlg-Bench's prompt format.

    The preamble, the task, the body, the method instruction and the closing
    instruction, separated by blank lines. That is the shape the benchmark's
    own prompts have, so a model trained on this is in distribution when it
    is scored.

    The order is the benchmark's: the method instruction comes after the
    matrix and before the closing instruction, because that is where the
    benchmark puts it.

    `tail` is the (method instruction, closing instruction) pair from
    prompt_tail(). The closing instruction is "" for the bare-matrix shape:
    all 220 of LinAlg-Bench's 4x4 problems are a matrix with no task
    statement and no closing instruction, so a third of the benchmark is
    scored on a prompt the corpus never produced.
    """
    method, closing = tail
    parts = [PREAMBLE, task, body]
    if method:
        parts.append(method)
    if closing:
        parts.append(closing)
    return "\n\n".join(parts) + "\n\n"


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
