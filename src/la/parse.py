"""Split a completion into its blocks, and pull a plot spec out of one.

Torch-free on purpose. This is the code that decides what the model actually
said, and it should be testable without a GPU.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .schema import validate_spec


def extract_json(text: str) -> Any:
    """Pull the first balanced JSON object out of a block of text."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def parse_completion(text: str) -> dict:
    """Split a completion into its three blocks.

    Parsing is deliberately tolerant. A model that produces a good answer and
    a broken plot should still be credited with the answer, so a failure in
    one block does not discard the others.
    """
    scratch = text
    answer = ""
    plot_text = ""
    m = re.search(r"###\s*Answer", text)
    if m:
        scratch = text[: m.start()]
        rest = text[m.end():]
        p = re.search(r"###\s*Plot", rest)
        if p:
            answer = rest[: p.start()]
            plot_text = rest[p.end():]
        else:
            answer = rest
    else:
        # no Answer marker: treat whatever follows the scratchpad as the answer
        m2 = re.search(r"###\s*Scratchpad", text)
        if m2:
            scratch = text[: m2.start()]
            answer = text[m2.end():]
    spec = extract_json(plot_text)
    out = {
        "scratchpad": scratch.strip(),
        "answer": answer.strip(),
        "plot_text": plot_text.strip(),
        "spec": spec,
        "spec_errors": validate_spec(spec) if spec else ["no plot spec"],
    }
    return out


def summarise_answer(text: str, limit: int = 90) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= limit else t[: limit - 1] + "…"
