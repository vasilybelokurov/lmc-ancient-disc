"""Paper table (Section 2, tab:ages): ages [Gyr] of the formed mass per prior [Fe/H] range.

Row AMR (reference): 16th-84th percentile ages and the mean of the reference weighting (age-metallicity relation of Povick et al.
2024), results/age_weighting.json from scripts/paper_mass_method.py. Rows A, B (older), C (younger): the age windows of uniform
formed mass, lmcold.mass_conversion.AGE_RANGES.

Writes results/table_ages.tex (the rows; copied to the paper's tables/table_ages.tex).
Run: .venv/bin/python scripts/manifest.py run paper_table_ages.py
"""
import json

from lmcold.mass_conversion import AGE_RANGES
from lmcold.sources import ROOT

# %% [markdown]
# ### Load the data
# The ages of the reference weighting per [Fe/H] range (scripts/paper_mass_method.py).

# %%
w = json.loads((ROOT / 'results/age_weighting.json').read_text())
RANGES = ['<-1.5', '-1.5..-1', '-1..-0.5', '>-0.5']   # keys of age_weighting.json, in the order of AGE_RANGES
# %% [markdown]
# ### Write the rows
# One row for the reference weighting, then one per age prior; the file holds only the rows, the paper's table the rest.

# %%
# AMR cells: 'p16--p84 (mean)' ages [Gyr] per [Fe/H] range, one decimal.
amr = [f'{w[k]["p16_p84"][0]:.1f}--{w[k]["p16_p84"][1]:.1f} ({w[k]["mean"]:.1f})' for k in RANGES]
rows = ['AMR (reference) & ' + ' & '.join(amr) + r'\\']
# Prior rows: the age window 'lo--hi' [Gyr] of each [Fe/H] range from AGE_RANGES; LaTeX row ends with \\.
for prior, label in (('A', 'A'), ('B', 'B (older)'), ('C', 'C (younger)')):
    rows.append(label + ' & ' + ' & '.join(f'{lo:g}--{hi:g}' for lo, hi in AGE_RANGES[prior]) + r'\\')
(ROOT / 'results/table_ages.tex').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
