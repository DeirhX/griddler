#!/usr/bin/env python3
"""Prove the DP line solver == the notebook's exhaustive enumeration.

For thousands of random (length, clue, known-cells) cases we compute the forced
cells two ways:
  * BEFORE: enumerate every arrangement with the notebook's Seq.variants and
    intersect them (a cell is forced iff identical across all arrangements).
  * AFTER:  fast_line_solver.solve_line (the O(n*k) DP).
Any mismatch is a bug and is printed.
"""
from __future__ import annotations

import contextlib
import io
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fast_line_solver import solve_line  # noqa: E402
from blowup_experiment import _load_notebook_namespace  # noqa: E402

NS = _load_notebook_namespace()
Seq = NS["Seq"]


def brute_forced(n, blocks, cur):
    """Return (feasible, forced) using the notebook's exhaustive enumeration.
    forced is a list in {1,-1,0}: 0 where arrangements disagree."""
    seq = Seq(list(blocks))
    with contextlib.redirect_stdout(io.StringIO()):
        arrangements = [v.unpack() for v in seq.variants(n, list(cur))]
    if not arrangements:
        return False, None
    forced = []
    for c in range(n):
        vals = {int(a[c]) for a in arrangements}
        forced.append(next(iter(vals)) if len(vals) == 1 else 0)
    return True, forced


def random_clue(n):
    """A random clue that can fit in n cells (possibly empty)."""
    blocks = []
    remaining = n
    while remaining > 0 and random.random() < 0.6:
        b = random.randint(1, max(1, remaining))
        if b + (1 if blocks else 0) > remaining:
            break
        blocks.append(b)
        remaining -= b + 1
    return blocks


def random_solution_cur(n, blocks):
    """Pick a real arrangement, then reveal a random subset of its cells."""
    seq = Seq(list(blocks))
    with contextlib.redirect_stdout(io.StringIO()):
        arrangements = [v.unpack() for v in seq.variants(n, [0] * n)]
    if not arrangements:
        return None
    sol = random.choice(arrangements)
    cur = [0] * n
    for c in range(n):
        if random.random() < 0.35:
            cur[c] = int(sol[c])
    return cur


def run(trials=4000, seed=0):
    random.seed(seed)
    mismatches = 0
    checked = 0
    for _ in range(trials):
        n = random.randint(1, 15)
        blocks = random_clue(n)

        # Two flavours of cur: derived-from-solution (always feasible) and fully random.
        if random.random() < 0.5:
            cur = random_solution_cur(n, blocks)
            if cur is None:
                continue
        else:
            cur = [random.choice([0, 0, 0, 1, -1]) for _ in range(n)]

        feasible, forced = brute_forced(n, blocks, cur)
        out = solve_line(n, blocks, cur)
        checked += 1

        if not feasible:
            if out is not None:
                mismatches += 1
                print(f"MISMATCH (brute infeasible, DP feasible): n={n} blocks={blocks} cur={cur} -> {out}")
            continue
        if out is None:
            mismatches += 1
            print(f"MISMATCH (brute feasible, DP infeasible): n={n} blocks={blocks} cur={cur}")
            continue
        # Compare forced cells. DP output must equal cur with forced cells applied.
        expected = [forced[c] if forced[c] != 0 else cur[c] for c in range(n)]
        if out != expected:
            mismatches += 1
            print(f"MISMATCH: n={n} blocks={blocks} cur={cur}\n  brute={expected}\n  dp   ={out}")

    print(f"\nChecked {checked} cases, {mismatches} mismatches.")
    return mismatches


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    raise SystemExit(1 if run(trials) else 0)
