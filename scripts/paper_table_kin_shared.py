"""Paper table (Section 4, tab:kin_shared): shared parameters of the joint (stage-2) fits.

Columns: Level 1 (symmetric disc, sky-aligned dispersions: results/kin_mcmc_data_centre.csv, stage 2 'shared') and Level 2
(velocity ellipsoid: results/kin_mcmc_data_ellipsoid.csv, stage 2 'shared'), each the posterior median with the 16th-84th
percentile half-range (as the caption states); then the range (min--max) of the stage-1 Level-2 medians of the five giant groups
(kin_mcmc_data_ellipsoid.csv), with the RR Lyrae stage-1 median in brackets (kin_mcmc_data_ellipsoid_outer.csv; none for v_los,0,
which the RR Lyrae do not constrain). Decimals per parameter in DEC (value, stage-1 range).

Writes results/table_kin_shared.tex (the rows; copied to the paper's tables/table_kin_shared.tex).
Run: .venv/bin/python scripts/manifest.py run paper_table_kin_shared.py
"""
import pandas as pd

from lmcold.sources import ROOT
from lmcold.tex import pm_sym

# %% [markdown]
# ### Load the data
# The posterior summaries of the Level-1 and Level-2 joint fits (2-9 deg) and of the RR Lyrae fit at 4-9 deg.

# %%
# One row per stage x population x parameter: median, p16, p84 (scripts/fit_kinematics_mcmc.py).
L1 = pd.read_csv(ROOT / 'results/kin_mcmc_data_centre.csv')  # Level 1: sky-aligned model, 2-9 deg
L2 = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid.csv')  # Level 2: velocity-ellipsoid model, 2-9 deg
RR = pd.read_csv(ROOT / 'results/kin_mcmc_data_ellipsoid_outer.csv')  # Level 2 at 4-9 deg (the RR Lyrae values)
# %% [markdown]
# ### Formats
# Row label and decimals of each shared parameter, and the two cell formats.

# %%
# parameter: (row label, decimals of the stage-2 values, decimals of the stage-1 range)
DEC = {'dx': (r'$\Delta x$ (deg)', 3, 2), 'dy': (r'$\Delta y$ (deg)', 3, 2),
       'pmra0': (r'$\mu_{\alpha*,0}$ (mas\,yr$^{-1}$)', 4, 3), 'pmdec0': (r'$\mu_{\delta,0}$ (mas\,yr$^{-1}$)', 3, 3),
       'vlos0': (r'$v_{\rm los,0}$ (\kms)', 1, 1), 'inc': (r'$i$ (deg)', 1, 1), 'theta': (r'$\theta$ (deg)', 1, 1)}


def shared(t, p, d):
    """Stage-2 shared value of p in table t as median +- half the 16th-84th range, or '--' if t does not fit p."""
    r = t[(t.stage == 2) & (t.population == 'shared') & (t.param == p)]
    return pm_sym(r['median'].iloc[0], (r.p84.iloc[0] - r.p16.iloc[0]) / 2, d) if len(r) else '--'


def num(x, d):
    """Plain number, in math mode if negative (as the paper prints minus signs)."""
    return f'${x:.{d}f}$' if round(x, d) < 0 else f'{x:.{d}f}'


# %% [markdown]
# ### Write the rows
# One row per shared parameter, then the m = 2 terms and the radial-motion terms of the Level-2 model.

# %%
rows = []  # LaTeX table rows
for p, (label, d, d1) in DEC.items():
    # Stage-1 Level-2 medians of the five giant groups, and the RR Lyrae stage-1 median of the outer run.
    g = L2[(L2.stage == 1) & L2.population.str.startswith('giants') & (L2.param == p)]['median']
    r = RR[(RR.stage == 1) & (RR.population == 'RR Lyrae') & (RR.param == p)]['median']
    # Stage-1 range 'min--max (RR Lyrae)'; no RR Lyrae value for vlos0 (no line-of-sight velocities).
    rng = f'{num(g.min(), d1)}--{num(g.max(), d1)}' + (f' ({num(r.iloc[0], d1)})' if p != 'vlos0' else '')
    # Columns: label, Level 1, Level 2, stage-1 range.
    rows.append(' & '.join([label, shared(L1, p, d), shared(L2, p, d), rng]) + r' \\')
# Streaming terms (Level 2 only; not in Level 1): m = 2 modulation of v_phi, then mean and m = 2 radial velocity.
rows.append(r'$m_{2c}$, $m_{2s}$ & -- & ' + ', '.join(shared(L2, p, 3) for p in ('m2c', 'm2s')) + r' & \\')
rows.append(r'$v_{R,0}$, $r_{2c}$, $r_{2s}$ & -- & ' + ', '.join(shared(L2, p, 3) for p in ('vr0', 'r2c', 'r2s')) + r' & \\')
# Write the rows (copied to the paper's tables/) and print them.
(ROOT / 'results/table_kin_shared.tex').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
