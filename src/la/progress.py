"""The progress line for a long generation.

A generation that prints nothing looks exactly like a hang. This lives
apart from the generation so it can be tested without a GPU.
"""

HEARTBEAT_SECONDS = 60


def progress_line(label: str, done: int, total: int, elapsed: float,
                  batch: int = 0, batches: int = 0) -> str:
    """The line printed while a generation is in flight.

    The batch number is in the line because the problem count only moves
    when a whole batch finishes. A batch of eight takes minutes, so the
    count alone sits at zero the entire time and looks frozen.
    """
    rate = done / max(elapsed, 1e-9)
    # A rate of zero, or one so small that the estimate is meaningless, is
    # worse to print than no estimate at all. Dividing by it produced
    # "about 11000000000 min left" on the first heartbeat of every run.
    if rate < 0.05:
        left = "?"
    else:
        left = f"{(total - done) / rate / 60:.0f}"
    where = ""
    if batches:
        where = f", batch {batch}/{batches}"
    return (f"  {label}: {done}/{total}{where}  ({rate:.1f}/s, "
            f"about {left} min left, {elapsed/60:.0f} min elapsed)")
