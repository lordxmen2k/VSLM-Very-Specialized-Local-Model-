"""The contract between the speed test and the processes it measures.

The speed test runs each micro-batch shape in a process of its own, because
the caching allocator does not give back memory it has cached and cannot
always reuse a block for a request of a different size. A process that has
already run a wider configuration measures the next one through its own
fragmentation, so measuring several shapes in one process measures the sum
of them rather than each one.

The parent and the child exchange one line of text, so the format lives here
and can be tested without a GPU.
"""
from __future__ import annotations

SPEED_ROW_PREFIX = "SPEEDROW"

# the shapes to measure. all of them are the same effective batch of 16, so
# all of them see the same number of examples per optimiser step.
SPEED_CONFIGS = [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)]


def format_row(bs: int, ga: int, s_step: float, tok_s: float, ex_s: float,
               alloc: float, resv: float) -> str:
    """The line a measuring process prints."""
    return (f"{SPEED_ROW_PREFIX}|{bs}|{ga}|{bs * ga}|{s_step:.4f}|{tok_s:.1f}"
            f"|{ex_s:.4f}|{alloc:.4f}|{resv:.4f}")


def format_oom() -> str:
    return f"{SPEED_ROW_PREFIX}|oom"


def parse_speed_row(line: str | None) -> dict | None:
    """Parse the machine-readable row a measuring process prints."""
    if not line or not line.startswith(SPEED_ROW_PREFIX):
        return None
    parts = line.split("|")
    if len(parts) > 1 and parts[1] == "oom":
        return {"oom": True}
    try:
        return {"oom": False,
                "bs": int(parts[1]), "ga": int(parts[2]), "eff": int(parts[3]),
                "s_step": float(parts[4]), "tok_s": float(parts[5]),
                "ex_s": float(parts[6]), "alloc": float(parts[7]),
                "resv": float(parts[8])}
    except (IndexError, ValueError):
        return None
