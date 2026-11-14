#!/usr/bin/env python3
"""Headless harness to observe how the solver scales on large grids.

It reuses the notebook's own code by exec-ing the relevant `# %%` cells from
Griddler.py (no duplication, no drift), stubbing matplotlib so we don't need a
GUI/backend. Then it can:

  * download a puzzle from griddlers.net (via the notebook's get_actual_puzzle)
  * report the *theoretical* per-line variant explosion (safe, no enumeration)
  * optionally run the real solver under a wall-clock budget (may be killed)

Usage:
  python tools/blowup_experiment.py download <id>
  python tools/blowup_experiment.py analyze  <id>
  python tools/blowup_experiment.py solve    <id> [max_seconds]
"""
from __future__ import annotations

import io
import math
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NB_PY = os.path.join(REPO, "Griddler.py")


def _load_notebook_namespace() -> dict:
    """Exec the class-definition and downloader cells from Griddler.py."""
    # Stub matplotlib so the import at the top of the solver cell succeeds
    # without pulling in a real backend.
    mpl = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda *a, **k: (None, None)
    pyplot.show = lambda *a, **k: None
    mpl.pyplot = pyplot
    sys.modules.setdefault("matplotlib", mpl)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)

    with io.open(NB_PY, "r", encoding="utf-8") as fh:
        text = fh.read()

    # Split on cell markers ("# %%" at column 0).
    cells = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if current:
                cells.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        cells.append("\n".join(current))

    ns: dict = {"__name__": "griddler_nb"}
    for cell in cells:
        if "class Field" in cell or "def get_actual_puzzle" in cell:
            exec(compile(cell, NB_PY, "exec"), ns)
    return ns


def variant_count(seq_list, width) -> int:
    """Number of arrangements of a clue in a line, closed form (no enumeration)."""
    space = width - (len(seq_list) - 2 + sum(seq_list))
    needles = len(seq_list)
    if space < 1:
        return 0
    return math.factorial(space + needles - 1) // (
        math.factorial(space - 1) * math.factorial(needles)
    )


def cmd_download(ns: dict, pid: int):
    rows, cols = ns["get_actual_puzzle"](pid)
    print(f"OK: {len(rows)} rows x {len(cols)} cols")
    return rows, cols


def cmd_analyze(ns: dict, pid: int):
    rows, cols = ns["get_actual_puzzle"](pid)
    n_rows, n_cols = len(rows), len(cols)
    print(f"\nGrid: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} cells")

    # width of a row = number of columns; width of a col = number of rows
    line_counts = []
    for r in rows:
        line_counts.append(("row", variant_count(r, n_cols)))
    for c in cols:
        line_counts.append(("col", variant_count(c, n_rows)))

    counts = [c for _, c in line_counts]
    total = sum(counts)
    biggest = max(counts)
    print(f"Lines: {len(counts)}")
    print(f"Total variants across all lines (initial, no clues known): {total:,}")
    print(f"Largest single line: {biggest:,}")
    print(f"Mean per line: {total // len(counts):,}")
    for thr in (1_000, 1_000_000, 1_000_000_000, 1_000_000_000_000):
        n = sum(1 for c in counts if c > thr)
        print(f"  lines with > {thr:>15,} variants: {n}")

    top = sorted(counts, reverse=True)[:10]
    print("Top 10 line variant counts:")
    for c in top:
        print(f"  {c:,}")


def _current_rss_bytes() -> int:
    """Process resident set size on Windows via psapi (no psutil dependency)."""
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    return counters.WorkingSetSize


def _start_watchdog(start: float, max_seconds: float, max_bytes: int):
    """Daemon that hard-exits the process on time or memory blow-up."""
    import threading

    def _watch():
        while True:
            time.sleep(0.5)
            elapsed = time.perf_counter() - start
            try:
                rss = _current_rss_bytes()
            except Exception:
                rss = 0
            if elapsed > max_seconds:
                print(f"\n[WATCHDOG] TIME cap hit: {elapsed:.1f}s > {max_seconds}s. "
                      f"RSS={rss/1e9:.2f} GB. Killing.", flush=True)
                os._exit(42)
            if rss > max_bytes:
                print(f"\n[WATCHDOG] MEMORY cap hit: {rss/1e9:.2f} GB > "
                      f"{max_bytes/1e9:.1f} GB at {elapsed:.1f}s. Killing.", flush=True)
                os._exit(43)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()


def cmd_solve(ns: dict, pid: int, max_seconds: float):
    rows, cols = ns["get_actual_puzzle"](pid)
    Seq, Field = ns["Seq"], ns["Field"]
    field = Field([Seq(r) for r in rows], [Seq(c) for c in cols])

    start = time.perf_counter()
    steps = 0
    max_bytes = 10 * 1024**3  # 10 GB hard cap
    _start_watchdog(start, max_seconds, max_bytes)
    print(f"\nSolving with {max_seconds:.0f}s / {max_bytes/1e9:.0f}GB hard caps "
          f"(watchdog will hard-kill)...\n", flush=True)
    try:
        for advanced, _ in field.solve_step():
            steps += 1
            elapsed = time.perf_counter() - start
            if elapsed > max_seconds:
                print(f"\n[BUDGET EXCEEDED] {elapsed:.1f}s, {steps} steps, still not solved")
                break
            if not advanced:
                break
    except MemoryError:
        print(f"\n[MEMORY ERROR] after {time.perf_counter() - start:.1f}s, {steps} steps")
        return
    print(f"solved={field.is_solved()} steps={steps} time={time.perf_counter() - start:.1f}s")
    print(f"Variant.unpacks_done={ns['Variant'].unpacks_done:,}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd, pid = argv[0], int(argv[1])
    ns = _load_notebook_namespace()
    if cmd == "download":
        cmd_download(ns, pid)
    elif cmd == "analyze":
        cmd_analyze(ns, pid)
    elif cmd == "solve":
        cmd_solve(ns, pid, float(argv[2]) if len(argv) > 2 else 60.0)
    else:
        print(f"unknown command: {cmd}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
