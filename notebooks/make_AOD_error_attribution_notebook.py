"""Generate AOD_error_attribution.ipynb from AOD_error_attribution.py.

This mirrors the existing make_figure3_notebook.py generator and keeps the
notebook in sync with the Python source.  Re-run this script after editing the
.py file to update the notebook.
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, '/scistor/guest/gbb083/AeroCom/.nb_tools')
import nbformat as nbf

script_path = Path('/scistor/guest/gbb083/AeroCom/notebooks/AOD_error_attribution.py')
out_path = Path('/scistor/guest/gbb083/AeroCom/notebooks/AOD_error_attribution.ipynb')

text = script_path.read_text()

# Top docstring becomes the intro markdown cell.
match = re.match(r'"""(.*?)"""', text, re.DOTALL)
intro_md = match.group(1).strip() if match else ''

# Code body after the docstring.
code_text = text[match.end():].strip() if match else text

# Section headers: three-line blocks with a numbered title.
# The separator length may vary, so accept a run of dashes.
section_pattern = re.compile(
    r'^(# -{70,90}\n#\s+(\d+)\.\s+(.*?)\n# -{70,90})$',
    re.MULTILINE,
)
parts = section_pattern.split(code_text)
pre_section = parts[0].strip()

cells = []
if intro_md:
    cells.append(nbf.v4.new_markdown_cell(intro_md))

if pre_section:
    cells.append(nbf.v4.new_code_cell(pre_section))

for i in range(1, len(parts), 4):
    if i + 3 >= len(parts):
        break
    title = parts[i + 2].strip()
    content = parts[i + 3].strip()
    cells.append(nbf.v4.new_markdown_cell(f'## {title}'))
    if content:
        cells.append(nbf.v4.new_code_cell(content))

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}

with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f'Created notebook: {out_path}')
print(f'Total cells: {len(cells)}')
