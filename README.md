# griddler

A probabilistic nonogram/griddler solver. The implementation lives in
`Griddler.ipynb`, paired with a plain-Python `Griddler.py` (jupytext `percent`
format) so it can be edited and reviewed as normal source.

## Setup

```bash
pip install -r requirements.txt

# One-time per clone: enable the notebook diff normalizer (config isn't versioned)
git config filter.nbnormalize.clean "python tools/nb_normalize.py --stdin"
```

## Working with the notebook

`Griddler.ipynb` and `Griddler.py` are paired via `jupytext.toml`. Edit either
side, then propagate changes with:

```bash
jupytext --sync Griddler.py     # or Griddler.ipynb
```

Prefer editing `Griddler.py` for code changes (clean diffs), and the notebook
when you want live output/plots.

## Why the notebook doesn't blow up

- `tools/nb_normalize.py` is wired as a git *clean* filter (see `.gitattributes`).
  On `git add` it strips cell outputs, resets `execution_count`, drops volatile
  metadata and cell ids, and normalizes `source` to line-lists. Re-running the
  notebook therefore never shows up as a diff.
- It is stdlib-only and also runs as a CLI: `python tools/nb_normalize.py *.ipynb`.
