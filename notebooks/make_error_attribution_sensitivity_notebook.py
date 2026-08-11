#!/usr/bin/env python
"""
Generator that converts `error_attribution_sensitivity_tests.py`
into a Jupyter notebook (`error_attribution_sensitivity_tests.ipynb`).

The companion script uses `# %%` comments as code-cell separators. Running
this generator produces a runnable notebook with the same logic split into
cells (setup + one cell per sensitivity test + summary).

Run with the project `.venv`:

    python notebooks/make_error_attribution_sensitivity_notebook.py

"""

import json
from pathlib import Path


def parse_py_to_notebook(py_path: Path) -> dict:
    """Parse a Python script with `# %%` cell markers into a notebook dict."""
    text = py_path.read_text(encoding='utf-8')
    lines = text.splitlines()

    cells = []
    current = []
    for line in lines:
        if line.strip().startswith('# %%'):
            # Flush the previous cell
            if current:
                cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'source': current})
                current = []
            # The marker itself is not included; the following lines go into the next cell
        else:
            current.append(line)
    if current:
        cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'source': current})

    # Ensure final newline consistency for each cell source
    for cell in cells:
        if cell['source'] and not cell['source'][-1].endswith('\n'):
            cell['source'][-1] += '\n'

    # Add a markdown intro cell at the top
    intro = [
        '# Error attribution sensitivity tests\n',
        '\n',
        'This notebook tests a series of literature recommendations against the current '
        'AeroCom error-decomposition methodology (Zhong et al. 2023, Sci. Adv.).\n',
        '\n',
        'Run all cells in order. Each sensitivity test is in its own cell so the '
        'results are easy to inspect.\n'
    ]
    cells.insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': intro})

    notebook = {
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.13.1'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5,
        'cells': cells,
    }
    return notebook


if __name__ == '__main__':
    here = Path(__file__).parent
    py_path = here / 'error_attribution_sensitivity_tests.py'
    nb_path = here / 'error_attribution_sensitivity_tests.ipynb'
    if not py_path.exists():
        raise FileNotFoundError(py_path)
    notebook = parse_py_to_notebook(py_path)
    nb_path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    print(f'Wrote notebook: {nb_path}')
    print(f'Cells: {len(notebook["cells"])}')
