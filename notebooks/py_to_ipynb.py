#!/usr/bin/env python3
"""Build an .ipynb from a percent-format script (``# %%`` / ``# %% [markdown]``).

The .py file stays the source of truth; re-run after editing it.

    python notebooks/py_to_ipynb.py notebooks/AAOD_error_attribution_PPE.py
    python notebooks/py_to_ipynb.py notebooks/AOD_DUST_ERROR_PPE.py --execute
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat as nbf

CELL_RE = re.compile(r"^# %%(?P<md>\s*\[markdown\])?.*$")
KERNELSPEC = {"display_name": ".venv-1 (3.13.15)", "language": "python", "name": "python3"}


def split_cells(text: str):
    cells, kind, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip("\n")
        if kind is None and not body.strip():
            return
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
            cells.append(nbf.v4.new_markdown_cell("\n".join(lines).strip()))
        elif body.strip():
            cells.append(nbf.v4.new_code_cell(body))

    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            flush()
            kind, buf = ("markdown" if m.group("md") else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("script", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--execute", action="store_true",
                   help="Execute in place (cwd = notebooks/) so outputs are stored")
    args = p.parse_args(argv)

    nb = nbf.v4.new_notebook()
    cells = split_cells(args.script.read_text())
    # Jupyter does not display figures that were closed, and a bare notebook
    # has no inline backend until this cell runs.
    cells.insert(0, nbf.v4.new_code_cell("%matplotlib inline"))
    nb["cells"] = cells
    nb.metadata["kernelspec"] = KERNELSPEC
    nb.metadata["language_info"] = {"name": "python"}
    out = args.out or args.script.with_suffix(".ipynb")
    if args.execute:
        from nbconvert.preprocessors import ExecutePreprocessor

        ExecutePreprocessor(timeout=-1, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(args.script.resolve().parent)}}
        )
    nbf.write(nb, out)
    print(f"Wrote {out} ({len(nb['cells'])} cells{', executed' if args.execute else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
