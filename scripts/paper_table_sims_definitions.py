"""Paper table (Appendix, tab:sims_definitions): number of simulated analogues whose rotational support at R = 4 kpc reaches the
LMC's, per giant [Fe/H] group, for v_rot over sigma_z, sigma_R or sigma_RMS, fitted (fit) or from the mock giants' particles (part.),
and under any definition. Counts from results/sims_definitions.csv (scripts/paper_sims_kinematics.py, which also defines the LMC
thresholds).

Writes results/table_sims_definitions.tex (the rows; copied to the paper's tables/table_sims_definitions.tex).
Run: .venv/bin/python scripts/manifest.py run paper_table_sims_definitions.py
"""
import pandas as pd

from lmcold.sources import ROOT
from lmcold.tex import GROUP_LABEL

# %% [markdown]
# ### Load the data
# The counts of analogues per group and definition (scripts/paper_sims_kinematics.py).

# %%
# One row per giant group (population) with the number of analogues reaching the LMC under each definition (columns of COLS).
t = pd.read_csv(ROOT / 'results/sims_definitions.csv')

# %% [markdown]
# ### Write the rows
# One row per giant group: the counts for each definition, in the column order of the paper.

# %%
# Definitions in the paper's column order: v_rot over sigma_z, sigma_R, sigma_RMS, fitted or from particles; 'any' of them.
COLS = ['z', 'z particles', 'R', 'R particles', 'RMS', 'RMS particles', 'any']
# Row: group label without the \feh macro, then the integer counts; LaTeX row end.
rows = [' & '.join([GROUP_LABEL[r.population].replace(r'\feh', '')] + [str(int(r[c])) for c in COLS]) + r' \\'
        for _, r in t.iterrows()]
(ROOT / 'results/table_sims_definitions.tex').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
