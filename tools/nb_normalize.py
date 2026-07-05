#!/usr/bin/env python3
"""Normalize Jupyter notebooks so git diffs stay small and stable.

Two modes:
  * git clean filter: reads a notebook from stdin, writes normalized JSON to
    stdout. Wire it up with `.gitattributes` (see repo) so it runs on `git add`.
  * CLI: `python tools/nb_normalize.py NOTEBOOK.ipynb [...]` rewrites files in
    place (handy for a one-off cleanup or a pre-commit sweep).

What it does:
  * clears code-cell outputs and resets execution_count to null
  * normalizes `source`/`text` from a single blob into a line list, which is
    what Jupyter writes natively, so tools that emit one giant string no longer
    produce wall-of-text diffs
  * drops volatile metadata (kernel timing, language_info.version, cell ids that
    only add churn) while leaving kernelspec intact so the notebook still opens

Stdlib only. No jupyter/nbformat/nbstripout required.
"""
from __future__ import annotations

import io
import json
import sys


def _as_line_list(value):
    """Return notebook multiline field as a list of lines, Jupyter-style."""
    if isinstance(value, list):
        text = "".join(value)
    elif isinstance(value, str):
        text = value
    else:
        return value
    if text == "":
        return []
    return text.splitlines(keepends=True)


def _clean_cell(cell: dict) -> dict:
    if "source" in cell:
        cell["source"] = _as_line_list(cell["source"])

    # Cell ids churn every time a tool (e.g. jupytext) rewrites the notebook and
    # carry no meaning for a single-file project, so drop them entirely.
    cell.pop("id", None)

    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

    meta = cell.get("metadata")
    if isinstance(meta, dict):
        for volatile in ("execution", "collapsed", "scrolled", "ExecuteTime"):
            meta.pop(volatile, None)

    return cell


def normalize(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if isinstance(cell, dict):
            _clean_cell(cell)

    meta = nb.get("metadata")
    if isinstance(meta, dict):
        lang = meta.get("language_info")
        if isinstance(lang, dict):
            # Version churns with every interpreter bump; kernelspec is enough.
            lang.pop("version", None)

    return nb


def dumps(nb: dict) -> str:
    # indent=1 matches nbformat's canonical on-disk formatting.
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def _run_stdin() -> int:
    raw = sys.stdin.buffer.read().decode("utf-8")
    nb = json.loads(raw)
    sys.stdout.buffer.write(dumps(normalize(nb)).encode("utf-8"))
    return 0


def _run_files(paths: list[str]) -> int:
    for path in paths:
        with io.open(path, "r", encoding="utf-8") as fh:
            nb = json.load(fh)
        with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(dumps(normalize(nb)))
    return 0


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--stdin"]
    if args:
        return _run_files(args)
    return _run_stdin()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
