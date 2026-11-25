"""Paper table (Section 2, tab:cuts): the count rows of the selection-function table, BOSS and APOGEE-2.

Parent stars, observed, good spectra and parent in covered groups: results/boss_selection.json 'summary' (eligible, good, covered;
scripts/boss_selection.py) with the observed count from data/boss_weights.parquet (column observed, same script), and
results/apogee_selection.json 'summary' (candidates, observed, quality, covered; scripts/apogee_selection.py). Mass sample:
results/giant_counts.json (boss_members, apogee_members; scripts/paper_giant_counts.py). The cut rows above them are text.

Writes results/table_cuts.tex (the count rows; copied to the paper's tables/table_cuts.tex).
Run: .venv/bin/python scripts/manifest.py run paper_table_cuts.py
"""
import json

import pandas as pd

from lmcold.sources import DATA, ROOT
from lmcold.tex import count

# %% [markdown]
# ### Load the data
# The selection summaries of both surveys, the mass-sample counts and the observed flag of the BOSS parent.

# %%
# Count summaries: b for BOSS, a for APOGEE-2 (dicts of integers), m the mass-sample counts of both
b = json.loads((ROOT / 'results/boss_selection.json').read_text())['summary']
a = json.loads((ROOT / 'results/apogee_selection.json').read_text())['summary']
m = json.loads((ROOT / 'results/giant_counts.json').read_text())
# Number of BOSS parent stars with a spectrum (sum of the boolean column observed)
b_observed = int(pd.read_parquet(DATA / 'boss_weights.parquet', columns=['observed']).observed.sum())
# %% [markdown]
# ### Write the rows
# Label, BOSS count, APOGEE-2 count; lmcold.tex.count formats integers as the paper prints them.

# %%
# (label, BOSS count, APOGEE-2 count) of each row, in the order of the paper's table
rows = [('Parent stars', b['eligible'], a['candidates']), ('Observed', b_observed, a['observed']),
        ('Good spectra', b['good'], a['quality']), ('Parent in covered groups', b['covered'], a['covered']),
        ('Mass sample$^b$', m['boss_members'], m['apogee_members'])]
# LaTeX rows 'label & N_BOSS & N_APOGEE \\'; written to results/table_cuts.tex and printed
rows = [f'{label} & {count(nb)} & {count(na)} \\\\' for label, nb, na in rows]
(ROOT / 'results/table_cuts.tex').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
