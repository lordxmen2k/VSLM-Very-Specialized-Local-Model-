"""VSLM — a linear algebra model you can train on one 12 GB card.

Importing this package quiets one warning that PyTorch emits on every CUDA
build without triton:

    W torch/utils/flop_counter.py:29] triton not found; flop counting will
    not work for triton kernels

It is a logging warning raised at import time of ``torch.utils.flop_counter``,
not at run time, and not by anything in this project. The counter it guards
is a diagnostic: ``FlopCounterMode`` intercepts operators and sums
shape-based FLOP formulas. This project never uses it, and PyTorch's own
documentation for it says "the workload still executes normally".

Triton has no official Windows wheel, so on Windows the warning appears
unavoidably and says nothing true about this project. Setting the level on
that one logger removes the noise without touching any other warning.

The logger is set here because importing ``la`` happens before anything in
``la.train`` imports torch, and the warning is emitted when torch first pulls
the flop counter in.
"""
import logging

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
