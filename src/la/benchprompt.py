"""LinAlg-Bench's prompt protocol, reproduced.

The benchmark does not feed the model a bare matrix. Its pipeline
(`pipeline/inference.py`) sends a system prompt and a user prompt built like
this:

    Solve this linear algebra problem. Show your work and give the final answer.

    {task}

    {problem}

with a system prompt that varies by subcategory:

    You are a precise mathematical assistant. Show all computation steps
    clearly. Always put your final numerical answer inside \\boxed{}.

The released CSVs carry the task statement inside the `problem_latex` column
for the 3x3 and 5x5 files, but the 4x4 file has a bare matrix with no
instruction at all. The benchmark's own pipeline reads a separate
`problem_text` column and falls back to a per-subcategory hint. This module
reconstructs the task for the 4x4 subset from the subcategory label, so the
problem is answerable, and says so when it does.

Reproducing the protocol matters more than it looks. A model trained on one
question format and scored on another is being asked to do a task it has
never seen phrased, and the score measures that rather than the mathematics.
"""
from __future__ import annotations

import re

PREAMBLE = ("Solve this linear algebra problem. Show your work and give the "
            "final answer.")

# The sequence cap for training and for generation. The corpus's longest
# completion is about 1,300 tokens; at a 1,024 cap, 3.9 per cent of it was
# cut short and the model learned to stop before the plot block. One
# module holds it so the training default and the test that guards it cannot
# drift apart.
MAX_SEQ_LEN = 1536
MAX_NEW_TOKENS = 2048

BASE_SYSTEM = ("You are a precise mathematical assistant. "
               "Show all computation steps clearly. ")

# The per-subcategory system prompt, verbatim from the benchmark's
# SUBCAT_CONFIG.
SYSTEM_PROMPTS = {
    "determinant": BASE_SYSTEM + "Always put your final numerical answer inside \\boxed{}.",
    "eigenvalue": BASE_SYSTEM + "Always put your final answers inside \\boxed{}.",
    "rank": BASE_SYSTEM + "Always put your final integer answer inside \\boxed{}.",
    "nullity": BASE_SYSTEM + "Always put your final integer answer inside \\boxed{}.",
    "multiplication": BASE_SYSTEM + "Always put your final matrix answer inside \\boxed{}.",
    "matrix_power": BASE_SYSTEM + "Always put your final matrix answer inside \\boxed{}.",
    "matrix_vector": BASE_SYSTEM + "Always put your final vector answer inside \\boxed{}.",
    "transpose": BASE_SYSTEM + "Always put your final matrix answer inside \\boxed{}.",
    "trace": BASE_SYSTEM + "Always put your final numerical answer inside \\boxed{}.",
}

# The benchmark's fallback hint, used when a problem carries no task text.
BOXED_HINTS = {
    "determinant": "Put your final numerical answer inside \\boxed{} at the end.",
    "eigenvalue": "Put your final eigenvalue(s) inside \\boxed{} at the end.",
    "rank": "Put your final integer answer inside \\boxed{} at the end.",
    "nullity": "Put your final integer answer inside \\boxed{} at the end.",
    "multiplication": "Put your final matrix answer inside \\boxed{} at the end.",
    "matrix_power": "Put your final matrix answer inside \\boxed{} at the end.",
    "matrix_vector": "Put your final vector answer inside \\boxed{} at the end.",
    "transpose": "Put your final matrix answer inside \\boxed{} at the end.",
    "trace": "Put your final numerical answer inside \\boxed{} at the end.",
}

# The task statements the benchmark uses, with the matrix size filled in.
# These are the benchmark's own 3x3 wordings, generalised across the three
# sizes, and they are what reconstructs the missing 4x4 instruction.
TASKS = {
    "determinant": "Find the determinant of the {size} matrix A.",
    "eigenvalue": "Find the eigenvalues of the {size} matrix A.",
    "rank": "Find the rank of the {size} matrix A.",
    "nullity": "Find the nullity of the {size} matrix A.",
    "multiplication": "Compute the product AB for the given {size} matrices A and B.",
    "matrix_power": "Compute A² = A × A for the {size} matrix A.",
    "matrix_vector": "Compute the matrix-vector product b = A·x for the {size} "
                     "matrix A and vector x.",
    "transpose": "Find the transpose of the {size} matrix A.",
    "trace": "Find the trace of the {size} matrix A.",
}

# The closing instruction the benchmark puts at the end of the problem body.
# The released problem_latex carries it, so the corpus's body carries it too.
BODY_HINTS = {
    "determinant": "Show all intermediate steps. Put your final numerical answer inside \\boxed{}.",
    "eigenvalue": "Show all intermediate steps. Put your final answers inside \\boxed{}.",
    "rank": "Show all intermediate steps. Put your final integer answer inside \\boxed{}.",
    "nullity": "Show all intermediate steps. Put your final integer answer inside \\boxed{}.",
    "multiplication": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "matrix_power": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "matrix_vector": "Show all intermediate steps. Put your final vector answer inside \\boxed{}.",
    "transpose": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "trace": "Show all intermediate steps. Put your final numerical answer inside \\boxed{}.",
    # the families this project has that the benchmark does not
    "solve_system": "Show all intermediate steps. Put your final vector answer inside \\boxed{}.",
    "orthonormal_basis": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "projection": "Show all intermediate steps. Put your final vector answer inside \\boxed{}.",
    "least_squares": "Show all intermediate steps. Put your final vector answer inside \\boxed{}.",
    "lu_factor": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "svd": "Show all intermediate steps. Put your final answers inside \\boxed{}.",
    "condition_number": "Show all intermediate steps. Put your final numerical answer inside \\boxed{}.",
    "rank_nullspace": "Show all intermediate steps. Put your final integer answer inside \\boxed{}.",
    "diagonalize": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "linear_transform": "Show all intermediate steps. Put your final vector answer inside \\boxed{}.",
    "matrix_trace": "Show all intermediate steps. Put your final numerical answer inside \\boxed{}.",
    "matrix_transpose": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "matrix_multiply": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "matrix_power": "Show all intermediate steps. Put your final matrix answer inside \\boxed{}.",
    "matrix_vector": "Show all intermediate steps. Put your final vector answer inside \\boxed{}.",
}

# The subcategory labels the benchmark uses in its CSVs.
SUBCAT_ALIASES = {
    "det": "determinant",
    "eig": "eigenvalue",
    "rank": "rank",
    "nullity": "nullity",
    "mult": "multiplication",
    "pow": "matrix_power",
    "vec": "matrix_vector",
    "trans": "transpose",
    "trace": "trace",
}

# A problem text that states a task starts with one of these verbs. The 4x4
# file's problems start with "A =" and so have none.
_STARTS_WITH_TASK = re.compile(
    r"^\s*(find|compute|calculate|evaluate|determine|what|for the|with|let|given|"
    r"apply|produce|using|the matrix|diagonalise)\b", re.IGNORECASE)


def normalise_subcat(subcat: str) -> str:
    """Map the benchmark's short labels onto the canonical names."""
    s = (subcat or "").strip().lower()
    return SUBCAT_ALIASES.get(s, s)


def has_task(problem: str) -> bool:
    """Does this problem text state what it wants?"""
    return bool(_STARTS_WITH_TASK.search(problem or ""))


def task_for(subcat: str, size: str | None = None) -> str:
    """The task statement for a subcategory, with the size filled in."""
    subcat = normalise_subcat(subcat)
    tpl = TASKS.get(subcat)
    if tpl is None:
        return BOXED_HINTS.get(subcat, "Give the final answer inside \\boxed{}.")
    if not size:
        return tpl.format(size="")
    # accept either "4" or "4x4" and always spell it the way the benchmark
    # does: "4x4"
    dim = str(size).split("x")[0]
    return tpl.format(size=f"{dim}x{dim}")


def build_prompt(task: str, problem: str) -> str:
    """The benchmark's user prompt."""
    return f"{PREAMBLE}\n\n{task}\n\n{problem}\n\n"


def model_prompt(question: str) -> str:
    """The prompt the local model is given.

    The corpus's questions already carry the preamble, the task, the body
    and the closing instruction, so a question is its own prompt. This
    lives here, with the rest of the prompt protocol, so that the training
    and the evaluation cannot drift apart: the corpus row's prompt and this
    function must return the same string.
    """
    return question


def body_hint(subcat: str) -> str:
    """The closing instruction for a family."""
    return BODY_HINTS.get(normalise_subcat(subcat),
                          BODY_HINTS["determinant"])


def build_benchmark_prompt(problem: str, subcat: str,
                            size: str | None = None) -> str:
    """The benchmark's full prompt: a system prompt and a user prompt.

    Where the released problem carries its own instruction - the 3x3 and 5x5
    files do - it is used as it stands. Where it does not - the whole 4x4
    file - the task is reconstructed from the subcategory label, because a
    bare matrix does not say what is being asked, and the problem would be
    unanswerable.
    """
    subcat = normalise_subcat(subcat)
    system = SYSTEM_PROMPTS.get(
        subcat, BASE_SYSTEM + "Always put your final answer inside \\boxed{}.")
    problem = (problem or "").strip()
    if has_task(problem):
        user = f"{PREAMBLE}\n\n{problem}\n\n"
    else:
        user = f"{PREAMBLE}\n\n{task_for(subcat, size)}\n\n{problem}\n\n"
    return f"{system}\n\n{user}"


def split_problem(problem: str, subcat: str, size: str | None = None):
    """Split a benchmark problem into (task, problem-body).

    Where the CSV carries the instruction, it is used as it stands. Where it
    does not - the whole 4x4 file - the task is reconstructed from the
    subcategory label, because a bare matrix does not say what is wanted.
    """
    if has_task(problem):
        return problem.strip(), problem.strip()
    return task_for(subcat, size), problem.strip()
