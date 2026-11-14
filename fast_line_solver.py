#!/usr/bin/env python3
"""Polynomial line solver for nonograms.

Replaces the notebook's exponential "enumerate every variant and average"
(Seq.gather_variants + Variants) with an O(n*k) dynamic program that computes
exactly the same information: which cells are forced filled / forced empty given
a line's clue and the cells already known.

Cell convention matches the notebook:
    1  = filled
   -1  = empty
    0  = unknown

`solve_line(n, blocks, cur)` returns a new cur with every deducible cell filled
in, or None if the clue is impossible given `cur` (a contradiction).

No enumeration, no per-variant objects, no unbounded memory.
"""
from __future__ import annotations


def _empty_prefix(n: int, cur: list[int]) -> list[int]:
    """emp[i] = number of known-empty (-1) cells in cur[0:i]. Lets us test
    'range [s,e) contains no forced-empty cell' in O(1)."""
    emp = [0] * (n + 1)
    for i in range(n):
        emp[i + 1] = emp[i] + (1 if cur[i] == -1 else 0)
    return emp


def _forward(n: int, blocks: list[int], cur: list[int],
             emp: list[int] | None = None) -> list[list[bool]]:
    """f[j][i] = blocks[0:j] can be placed within the first i cells, 'closed'
    (cell i-1 is either empty or the last cell of block j-1), consistent with cur.
    """
    if emp is None:
        emp = _empty_prefix(n, cur)
    k = len(blocks)
    f = [[False] * (n + 1) for _ in range(k + 1)]
    f[0][0] = True
    # No blocks placed yet: the whole prefix must be empty-able.
    for i in range(1, n + 1):
        f[0][i] = f[0][i - 1] and cur[i - 1] != 1
    for j in range(1, k + 1):
        L = blocks[j - 1]
        fj = f[j]
        fj1 = f[j - 1]
        for i in range(1, n + 1):
            # Option A: cell i-1 is a gap.
            res = cur[i - 1] != 1 and fj[i - 1]
            # Option B: block j-1 occupies [s, i) (no forced-empty inside).
            if not res:
                s = i - L
                if s >= 0 and emp[i] - emp[s] == 0:
                    if s == 0:
                        res = j == 1
                    else:
                        res = cur[s - 1] != 1 and fj1[s - 1]
            fj[i] = res
    return f


def _backward(n: int, blocks: list[int], cur: list[int]) -> list[list[bool]]:
    """b[j][i] = blocks[j:k] can be placed within cells [i:n]. Computed by
    running the forward DP on the reversed line/clue."""
    k = len(blocks)
    fr = _forward(n, blocks[::-1], cur[::-1])
    b = [[False] * (n + 1) for _ in range(k + 1)]
    for j in range(k + 1):
        row = b[j]
        fr_row = fr[k - j]
        for i in range(n + 1):
            row[i] = fr_row[n - i]
    return b


def solve_line(n: int, blocks: list[int], cur: list[int]) -> list[int] | None:
    """Return cur with all forced cells resolved, or None on contradiction."""
    blocks = list(blocks)
    k = len(blocks)
    emp = _empty_prefix(n, cur)
    f = _forward(n, blocks, cur, emp)
    if not f[k][n]:
        return None  # clue cannot be satisfied given current cells

    b = _backward(n, blocks, cur)

    can_empty = [False] * n

    # A cell can be empty if the blocks can be split so that j of them fit
    # strictly before it and the remaining k-j strictly after it.
    for c in range(n):
        if cur[c] == 1:
            continue
        for j in range(k + 1):
            if f[j][c] and b[j][c + 1]:
                can_empty[c] = True
                break

    # A cell can be filled if some block can cover it with a valid arrangement
    # on both sides. Mark covered ranges with a difference array (O(1) per
    # placement) instead of writing every cell.
    diff = [0] * (n + 1)
    for j in range(k):
        L = blocks[j]
        for s in range(0, n - L + 1):
            e = s + L
            if emp[e] - emp[s] != 0:
                continue
            if s == 0:
                left_ok = j == 0
            else:
                left_ok = cur[s - 1] != 1 and f[j][s - 1]
            if not left_ok:
                continue
            if e == n:
                right_ok = j == k - 1
            else:
                right_ok = cur[e] != 1 and b[j + 1][e + 1]
            if not right_ok:
                continue
            diff[s] += 1
            diff[e] -= 1

    can_fill = [False] * n
    running = 0
    for c in range(n):
        running += diff[c]
        can_fill[c] = running > 0

    out = list(cur)
    for c in range(n):
        if can_fill[c] and not can_empty[c]:
            out[c] = 1
        elif can_empty[c] and not can_fill[c]:
            out[c] = -1
        elif not can_fill[c] and not can_empty[c]:
            return None  # cell can be neither -> contradiction
    return out
