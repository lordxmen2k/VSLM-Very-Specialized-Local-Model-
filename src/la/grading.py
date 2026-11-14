"""Grade a model's answer against a reference.

Torch-free on purpose. This is the code that produces the headline number,
and it should be testable without a GPU and without a model.
"""
from __future__ import annotations

import re

from .parse import parse_completion

# LinAlg-Bench states the task in the question text - "Find the determinant
# of the 4x4 matrix A", and "Show all intermediate steps". The 4x4 file has
# neither: its questions are a bare matrix with no instruction at all, so
# nothing tells the model, or the grader, what is being asked. Every 4x4
# problem in the benchmark is unanswerable as written. That is a property of
# the data, not of the model.
INSTRUCTION_RE = re.compile(r"\b(find|compute|calculate|evaluate|determine)\b",
                            re.IGNORECASE)


def is_instructed(question: str) -> bool:
    """Does this question say what it wants?"""
    return INSTRUCTION_RE.search(question) is not None


def extract_boxed(text: str) -> list[str]:
    """Every \\boxed{...} in the text, with the braces balanced.

    A non-greedy regex is wrong here. \\boxed{A = \\begin{bmatrix} 1 & 2
    \\end{bmatrix}} stops at the first closing brace, which is the one in
    \\begin{bmatrix}, and yields "\\begin{bmatrix". Every matrix-valued
    answer in the benchmark - 420 of its 660 problems - is written that way.
    """
    out: list[str] = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        i = m.end()
        while i < len(text) and depth:
            c = text[i]
            if c in "{}" and (i == 0 or text[i - 1] != "\\"):
                depth += 1 if c == "{" else -1
            i += 1
        if depth == 0:
            out.append(text[m.end():i - 1])
    return out


def normalise_reference(answer: str) -> str:
    """Strip the prose around a benchmark's reference answer.

    The benchmark writes answers like "det(A) = 5" or
    "\\lambda_1 = 1, \\lambda_2 = 2". Only the values are wanted.
    """
    a = answer.strip()
    boxes = extract_boxed(a)
    if boxes:
        a = boxes[-1].strip()
    # drop a leading label such as "det(A) = " or "x = "
    m = re.match(r"^[^=]{1,24}=\s*(.+)$", a, flags=re.S)
    if m and "=" not in m.group(1):
        a = m.group(1).strip()
    a = a.rstrip(".")
    return a


def extract_model_answer(raw: str) -> str:
    """Pull the model's answer out of a completion, whatever shape it took.

    The fine-tuned model uses the ### Answer block. A base model does not, so
    fall back to the last \\boxed{} and then to the last line that looks like
    an answer.

    Whatever comes out is put through the same normalisation as the
    reference. A base model answers "\\det(A) = 97" inside the box, and the
    label has to come off before the answer grammar ever sees it, or every
    determinant problem it gets right still scores zero.
    """
    parsed = parse_completion(raw)
    if parsed["answer"].strip():
        return normalise_reference(parsed["answer"])
    boxes = extract_boxed(raw)
    if boxes:
        return normalise_reference(boxes[-1])
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    return normalise_reference(lines[-1]) if lines else ""
