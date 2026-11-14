#!/usr/bin/env python3
"""Micro-benchmark: solve the single worst line of a puzzle with the DP,
the one the old enumeration approach could never finish."""
from __future__ import annotations

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fast_line_solver import solve_line  # noqa: E402
from blowup_experiment import _load_notebook_namespace, variant_count  # noqa: E402


def main(pid: int):
    ns = _load_notebook_namespace()
    rows, cols = ns["get_actual_puzzle"](pid)
    n_rows, n_cols = len(rows), len(cols)

    lines = [(r, n_cols) for r in rows] + [(c, n_rows) for c in cols]
    worst = max(lines, key=lambda rc: variant_count(rc[0], rc[1]))
    clue, width = worst
    vc = variant_count(clue, width)

    print(f"Worst line: clue={clue}")
    print(f"width={width}, theoretical variants to enumerate (OLD): {vc:,}")
    print(f"  (that's ~10^{int(math.log10(vc))}; at 900 bytes/variant it needs "
          f"~{vc*900/1e18:.1f} exabytes of RAM)")

    cur = [0] * width
    t = time.perf_counter()
    out = solve_line(width, clue, cur)
    dt = time.perf_counter() - t
    forced = sum(1 for x in out if x != 0)
    print(f"\nNEW DP solve_line: {dt*1000:.3f} ms")
    print(f"  forced {forced}/{width} cells on the very first pass (no clues known)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 293872)
