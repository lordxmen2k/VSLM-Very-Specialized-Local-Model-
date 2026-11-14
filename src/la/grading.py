"""Grade a model's answer against a reference.

Torch-free on purpose. This is the code that produces the headline number,
and it should be testable without a GPU and without a model.
"""
from __future__ import annotations

import re

from .answer import parse_answer
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


def extract_last_value(raw: str) -> str:
    """The last line of a response that parses as a value.

    The lenient reading. A model that writes "So, the determinant of the
    matrix A is -40" has answered; it just did not put the answer where a
    grader looking for \boxed{} would find it. This is a diagnostic, not
    the benchmark's protocol, and it is reported beside that number so the
    two can be compared.
    """
    trailing_matrix = re.compile(
        r"(\\begin\{[a-z]*matrix\}.*?\\end\{[a-z]*matrix\})", re.S)
    trailing_number = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")
    for line in reversed([l.strip() for l in (raw or "").splitlines()]):
        if not line or line.startswith("```"):
            continue
        # the whole line first, then a trailing matrix, then a trailing
        # number
        candidates = [line, line.rstrip("."), line.split("=")[-1].strip()]
        m = trailing_matrix.search(line)
        if m:
            candidates.append(m.group(1))
        m = trailing_number.search(line)
        if m:
            candidates.append(m.group(1))
        for candidate in candidates:
            candidate = candidate.strip().strip("$ ")
            # a candidate with no digit in it is prose, and the answer
            # grammar will happily read a row of words as a product of
            # symbols, so the digit has to be there
            if not candidate or not re.search(r"\d", candidate):
                continue
            try:
                value = parse_answer(candidate)
            except Exception:
                continue
            # and it has to be a value, not a product of words that happens
            # to parse: "So the 2-norm condition number of A is 1.00000"
            # reads as a product of symbols, which is not an answer
            if getattr(value, "free_symbols", None):
                continue
            return candidate
    return ""


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
    last = extract_last_value(raw)
    if last:
        return normalise_reference(last)
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    return normalise_reference(lines[-1]) if lines else ""
