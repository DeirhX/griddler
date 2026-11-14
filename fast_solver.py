#!/usr/bin/env python3
"""Fast nonogram solver: DP line solving + dirty-line propagation + backtracking.

Addresses the notebook solver's structural weaknesses:
  * #1-3  no variant enumeration -- uses fast_line_solver.solve_line (O(n*k))
  * #5    a dirty-line queue: only re-solve lines whose cells actually changed,
          instead of re-scoring and re-sorting every line every step
  * #4    correct backtracking: deep-copies only the tiny int8 grid (not the
          shared clue/cache objects), branches on the most-constrained line, and
          tries BOTH filled and empty (the notebook tried filled twice)

Grid convention: numpy int8, 1=filled, -1=empty, 0=unknown.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from fast_line_solver import solve_line


class FastField:
    def __init__(self, rows, cols):
        self.row_clues = [list(r) for r in rows]
        self.col_clues = [list(c) for c in cols]
        self.R = len(rows)
        self.C = len(cols)
        self.grid = np.zeros((self.R, self.C), dtype=np.int8)
        # instrumentation
        self.line_solves = 0
        self.nodes = 0

    def _clue(self, is_row, idx):
        return self.row_clues[idx] if is_row else self.col_clues[idx]

    def _line(self, grid, is_row, idx):
        return (grid[idx, :] if is_row else grid[:, idx]).tolist()

    def propagate(self, grid, dirty: deque) -> bool:
        """Run constraint propagation to a fixpoint. Returns False on contradiction."""
        queued = set(dirty)
        while dirty:
            is_row, idx = dirty.popleft()
            queued.discard((is_row, idx))
            n = self.C if is_row else self.R
            cur = self._line(grid, is_row, idx)
            self.line_solves += 1
            out = solve_line(n, self._clue(is_row, idx), cur)
            if out is None:
                return False
            for p in range(n):
                if out[p] != cur[p]:
                    if is_row:
                        grid[idx, p] = out[p]
                    else:
                        grid[p, idx] = out[p]
                    # a change in this line dirties the crossing (perpendicular) line
                    perp = (False, p) if is_row else (True, p)
                    if perp not in queued:
                        queued.add(perp)
                        dirty.append(perp)
        return True

    def _pick_branch_cell(self, grid):
        """Most-constrained line: fewest unknowns but at least one. Branch on its
        first unknown cell. Far better than the notebook's top-left scan."""
        best_idx = best_is_row = None
        best_unknown = None
        for i in range(self.R):
            unk = int((grid[i, :] == 0).sum())
            if unk > 0 and (best_unknown is None or unk < best_unknown):
                best_unknown, best_idx, best_is_row = unk, i, True
        for j in range(self.C):
            unk = int((grid[:, j] == 0).sum())
            if unk > 0 and (best_unknown is None or unk < best_unknown):
                best_unknown, best_idx, best_is_row = unk, j, False
        if best_idx is None:
            return None
        if best_is_row:
            p = int(np.where(grid[best_idx, :] == 0)[0][0])
            return best_idx, p
        p = int(np.where(grid[:, best_idx] == 0)[0][0])
        return p, best_idx

    def _search(self, grid):
        self.nodes += 1
        if (grid != 0).all():
            return grid
        cell = self._pick_branch_cell(grid)
        if cell is None:
            return grid
        i, j = cell
        for val in (1, -1):
            g2 = grid.copy()
            g2[i, j] = val
            if self.propagate(g2, deque([(True, i), (False, j)])):
                res = self._search(g2)
                if res is not None:
                    return res
        return None

    def solve(self):
        dirty = deque([(True, i) for i in range(self.R)]
                      + [(False, j) for j in range(self.C)])
        if not self.propagate(self.grid, dirty):
            return None
        result = self._search(self.grid)
        if result is not None:
            self.grid = result
        return result


def _runs(line):
    out = []
    count = 0
    for v in line:
        if v == 1:
            count += 1
        elif count:
            out.append(count)
            count = 0
    if count:
        out.append(count)
    return out


def verify(grid, row_clues, col_clues) -> bool:
    """Independent check that the solved grid actually matches every clue."""
    for i, clue in enumerate(row_clues):
        if _runs(grid[i, :].tolist()) != [x for x in clue if x]:
            return False
    for j, clue in enumerate(col_clues):
        if _runs(grid[:, j].tolist()) != [x for x in clue if x]:
            return False
    return bool((grid != 0).all())


if __name__ == "__main__":
    import os
    import sys
    import time

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "tools"))
    from blowup_experiment import _load_notebook_namespace

    ns = _load_notebook_namespace()
    arg = sys.argv[1] if len(sys.argv) > 1 else "293872"
    if arg.endswith(".txt"):
        rows, cols = ns["get_test_puzzle"](arg)
    else:
        rows, cols = ns["get_actual_puzzle"](int(arg))

    field = FastField(rows, cols)
    t = time.perf_counter()
    result = field.solve()
    dt = time.perf_counter() - t

    solved = result is not None
    ok = verify(field.grid, field.row_clues, field.col_clues) if solved else False
    print(f"\n{field.R}x{field.C}  solved={solved}  verified={ok}")
    print(f"time={dt*1000:.1f} ms  line_solves={field.line_solves:,}  search_nodes={field.nodes:,}")
