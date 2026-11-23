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
#
# THE VALUE IS 2048, NOT 1536. The 1,536 came from a character-count
# estimate - the corpus's length divided by 2.04 - which is not a token
# count and was never checked against the tokenizer. The real cap is
# `max_seq_len: 2048` in configs/default.yaml, and the trainer truncates
# at that.
#
# The cost of the stale number was a real one: the 5x5 dot products were
# measured at "1,601 tokens against a 1,536 cap", so they were taught in a
# COMPACT form that dropped the product terms - the run 9 defect that cost
# the matrix_power family 26 problems. Measured in REAL tokens with the
# model's own tokenizer, the 5x5 examples with their products shown come
# to 1,892 and 2,042 - inside 2048.
#
# It is read from the config so that the cap has ONE home.
def _max_seq_len() -> int:
    """The training cap, from the config that the trainer reads."""
    import os

    import yaml

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", ".."))
    try:
        with open(os.path.join(repo, "configs", "default.yaml"),
                  encoding="utf-8") as fh:
            return int(yaml.safe_load(fh)["training"]["max_seq_len"])
    except Exception:
        return 2048


MAX_SEQ_LEN = _max_seq_len()
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
#
# For the nine benchmark subcategories this table is only a FALLBACK. The
# benchmark uses more than one closing instruction per subcategory, and the
# corpus has to produce all of them - the full set, with the benchmark's own
# counts, is in `benchprompt_tails.py` (PROMPT_TAILS), and that table is
# checked against the released problem files line for line by
# tests/test_tail_table_matches_benchmark.py.
#
# The families the benchmark does not have use this table directly.
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

# The corpus family -> benchmark subcategory mapping.
#
# The corpus and the benchmark do not always use the same name for the same
# family. One table is the only place that reconciles them, so the system
# prompt and the closing instruction always come from the benchmark's own
# tables. A family with no benchmark counterpart is absent and gets the
# general prompt.
FAMILY_TO_SUBCAT = {
    "determinant": "determinant",
    "eigenvalue": "eigenvalue",
    "eigenvalue_numeric": "eigenvalue",
    "rank": "rank",
    "nullity": "nullity",
    "matrix_trace": "trace",
    "matrix_transpose": "transpose",
    "matrix_multiply": "multiplication",
    "matrix_power": "matrix_power",
    "matrix_vector": "matrix_vector",
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
    """The task statement for a subcategory, with the size filled in.

    The size is spelled with U+00D7, the way the benchmark spells it:
    "Find the determinant of the 4×4 matrix A." This function
    reconstructs the task for the benchmark's 4x4 problems, which are a
    bare matrix with no task statement at all, and the model is scored on
    what this returns. The corpus has to be trained on the SAME string,
    so both sides use this one function.

    (The corpus used to write "4x4" with an ASCII "x" while the
    benchmark's own 3x3 and 5x5 statements use U+00D7 - so the model was
    trained on one spelling and scored on another for the same family.)
    """
    subcat = normalise_subcat(subcat)
    tpl = TASKS.get(subcat)
    if tpl is None:
        return BOXED_HINTS.get(subcat, "Give the final answer inside \\boxed{}.")
    if not size:
        return tpl.format(size="")
    # accept either "4" or "4x4" and always spell it the way the
    # benchmark does: "4×4"
    dim = str(size).split("x")[0]
    return tpl.format(size=f"{dim}×{dim}")


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


# --------------------------------------------------------------- chat ---

# Qwen3's chat template, and the two flags that decide what the model does
# with it.
#
# The model is `Qwen/Qwen3-1.7B`, the instruction-tuned checkpoint. Its
# natural turn-ender is <|im_end|>, and its weights are trained to emit it
# after a complete answer. The Base checkpoint has no such signal: it stops
# only because it recognises the corpus format, and under a prompt it does
# not recognise it keeps generating until it hits the token cap.
#
# `enable_thinking=False` is deliberate. Qwen3 can wrap its answer in a
# <think> block, but this corpus already teaches explicit reasoning - every
# derivation is a numbered STEP line ending in a VERIFY check - so a hidden
# reasoning pass would duplicate it, and Qwen's own documentation warns that
# greedy decoding in thinking mode "can lead to performance degradation and
# endless repetitions".
DEFAULT_THINKING = False


def render_chat(tokenizer, system: str, user: str,
                thinking: bool = DEFAULT_THINKING) -> str:
    """Render a system-and-user pair through the tokenizer's chat template.

    One definition, used by the corpus builder and by inference, so that
    what the model is trained on and what it is scored on are the same
    string. The first project used the Base checkpoint and fed it raw text
    with no template at all; the benchmark then prepended its own system
    prompt, and the model was answering a prompt it had never seen. It did
    not stop, and the first batch of the benchmark did not finish in six
    minutes.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking)
    except TypeError:
        # A tokenizer whose template does not take enable_thinking. Qwen3's
        # does; this keeps the function usable with one that does not.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def system_prompt_for(task: str) -> str:
    """The benchmark's system prompt for a corpus family or subcategory."""
    subcat = FAMILY_TO_SUBCAT.get(task, task)
    return SYSTEM_PROMPTS.get(
        subcat, BASE_SYSTEM + "Always put your final numerical answer inside \\boxed{}.")


def row_prompt(row: dict, tokenizer, thinking: bool = DEFAULT_THINKING) -> str:
    """The rendered prompt for a corpus row.

    One definition, used by the corpus builder, the trainer and the
    benchmark. The row carries the system turn and the user turn; this
    renders them through the tokenizer's chat template.

    A row that has no chat turns - one written before the migration - falls
    back to its `prompt` field, which is the bare question. That keeps the
    old corpus usable and makes the difference visible: the old rows have
    no system turn, so the benchmark's system prompt has nothing to match
    against.
    """
    system = row.get("system")
    user = row.get("user")
    if system is None and user is None:
        return row.get("prompt", row.get("question", ""))
    return render_chat(tokenizer, system or "", user or "",
                       thinking=thinking)


def body_hint(subcat: str) -> str:
    """The closing instruction for a corpus family or benchmark subcategory."""
    subcat = FAMILY_TO_SUBCAT.get(subcat, subcat)
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
    system, user = benchmark_prompt_pair(problem, subcat, size)
    return f"{system}\n\n{user}"


def benchmark_prompt_pair(problem: str, subcat: str,
                           size: str | None = None) -> tuple[str, str]:
    """The benchmark's system prompt and user prompt, separately.

    The system prompt is the benchmark's own, verbatim, per subcategory. The
    user prompt is the preamble, the task and the body.

    The model is given these through the tokenizer's chat template, so the
    two arrive as the system and user turns the model was instruction-tuned
    to expect. Feeding them as one flat string is what made the first
    benchmark batch run for six minutes: the model did not recognise the
    prefix, so it did not stop.
    """
    subcat = normalise_subcat(subcat)
    system = SYSTEM_PROMPTS.get(
        subcat, BASE_SYSTEM + "Always put your final answer inside \\boxed{}.")
    problem = (problem or "").strip()
    if has_task(problem):
        user = f"{PREAMBLE}\n\n{problem}\n\n"
    else:
        user = f"{PREAMBLE}\n\n{task_for(subcat, size)}\n\n{problem}\n\n"
    return system, user


def split_problem(problem: str, subcat: str, size: str | None = None):
    """Split a benchmark problem into (task, problem-body).

    Where the CSV carries the instruction, it is used as it stands. Where it
    does not - the whole 4x4 file - the task is reconstructed from the
    subcategory label, because a bare matrix does not say what is wanted.
    """
    if has_task(problem):
        return problem.strip(), problem.strip()
    return task_for(subcat, size), problem.strip()
