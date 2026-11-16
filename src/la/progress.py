"""The progress line for a long generation.

A generation that prints nothing looks exactly like a hang. This lives
apart from the generation so it can be tested without a GPU.
"""

HEARTBEAT_SECONDS = 60


def progress_line(label: str, done: int, total: int, elapsed: float) -> str:
    """The line printed while a generation is in flight."""
    rate = done / max(elapsed, 1e-9)
    # Before the first batch there is no rate to speak of, and a confident
    # number for it is worse than none.
    if done == 0:
        left = "?"
    else:
        left = f"{(total - done) / max(rate, 1e-9) / 60:.0f}"
    return (f"  {label}: {done}/{total}  ({rate:.1f}/s, "
            f"about {left} min left, {elapsed/60:.0f} min elapsed)")
