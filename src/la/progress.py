"""The progress line for a long generation.

A generation that prints nothing looks exactly like a hang. This lives
apart from the generation so it can be tested without a GPU.
"""

HEARTBEAT_SECONDS = 60


def progress_line(label: str, done: int, total: int, elapsed: float) -> str:
    """The line printed while a generation is in flight."""
    rate = done / max(elapsed, 1e-9)
    left = (total - done) / max(rate, 1e-9)
    return (f"  {label}: {done}/{total}  ({rate:.1f}/s, "
            f"about {left/60:.0f} min left, {elapsed/60:.0f} min elapsed)")
