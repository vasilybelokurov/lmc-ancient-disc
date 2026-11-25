"""Paper figure (Sect. 2.3): the total stellar mass of the LMC, this work against published estimates.

One row per study: formed mass (triangles) and present mass in living stars and remnants (squares), as published (no IMF or
aperture conversion; lmcold.constants.LMC_MASS_LITERATURE). This work (results/giant_present_mass.csv): reference value, 16th-84th
bootstrap percentiles (thin bars) and range of the model alternatives (thick light bars), also shaded across all rows.
Writes figures/paper_mass_summary.png.

Input: results/giant_present_mass.csv (scripts/build_giant_mass.py, then scripts/build_present_mass.py), one row per reporting
[Fe/H] bin; its last row (the open bin [Fe/H] >= 0) holds the cumulative mass of all bins, i.e. the total within 12 deg. Columns
used: cumulative and present_cumulative [Msun], with suffixes _p16/_p84 (bootstrap) and _model_min/_model_max (alternatives).
Panel (single): x: stellar mass [10^9 Msun], log, 0.8-4.5; y: one row per study, this work on top (separated by a grey line),
then the published estimates in order of publication year. Blue triangles: formed mass; orange squares: present mass; open
symbols: published values. Paper: Sect. 2.3, main text, with Table 3 (scripts/paper_table_mass_summary.py).
Run: .venv/bin/python scripts/paper_mass_summary.py
Run time: 1 s on a MacBook (M-series), docs/run_times.txt.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmcold.constants import LMC_MASS_LITERATURE
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

# %% [markdown]
# ### Load the data
# The total formed and present mass of this work: the last row of `results/giant_present_mass.csv` (written by
# `scripts/build_present_mass.py`). The published masses are not read from a file: they are listed in
# `lmcold.constants.LMC_MASS_LITERATURE`.

# %%
# results/giant_present_mass.csv: formed and present mass with bootstrap percentiles and model ranges; last bin = total mass
t = pd.read_csv(ROOT / 'results/giant_present_mass.csv').iloc[-1]  # last bin: cumulative over all [Fe/H] = total mass

# %% [markdown]
# ### This work and the published masses
# The total of this work and the published estimates, ordered by year.

# %%
blue, orange = '#2878b5', '#d75a20'
lit = sorted(LMC_MASS_LITERATURE, key=lambda r: r['label'][-5:-1])  # by the year in the label, e.g. 'Kim et al. (1998)'
labels = ['This work'] + [r['label'] for r in lit]
y = np.arange(len(labels))[::-1]  # row positions, top row (this work) highest
# %% [markdown]
# ### The figure

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, axes = panel_grid(1, 1, width=3.4, left=1.45, right=.1, bottom=.45, top=.45, aspect=.95)
ax = axes[0, 0]
for pre, colour, mk, dy, lab in [('cumulative', blue, '^', .12, 'formed'), ('present_cumulative', orange, 's', -.12, 'present')]:
    # Reference value, bootstrap 16th/84th percentiles and model range, in 10^9 Msun.
    v, lo, hi, mlo, mhi = (t[pre + s] / 1e9 for s in ('', '_p16', '_p84', '_model_min', '_model_max'))
    ax.axvspan(mlo, mhi, color=colour, alpha=.12, lw=0)  # model range shaded across all rows
    ax.plot([mlo, mhi], [y[0] + dy] * 2, color=colour, lw=4, alpha=.35, solid_capstyle='butt')
    ax.errorbar(v, y[0] + dy, xerr=[[v - lo], [hi - v]], fmt=mk, ms=5, color=colour, lw=1, label=lab)
for yy, r in zip(y[1:], lit):
    formed = r['kind'] == 'formed'  # published as formed mass; otherwise present mass
    ax.plot(r['mass'] / 1e9, yy + (.12 if formed else -.12), '^' if formed else 's', ms=5, mfc='white',
            color=blue if formed else orange)
ax.set(xscale='log', xlim=(.8, 4.5), ylim=(-.7, len(labels) - .3), xlabel=r'Stellar mass ($10^9\,M_\odot$)')
ax.set_xticks([1, 1.5, 2, 3, 4], ['1', '1.5', '2', '3', '4'])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks(y, labels, fontsize=7)
ax.axhline(y[0] - .5, color='.6', lw=.6)
h, l = ax.get_legend_handles_labels()  # present (left) then formed (right), above their columns
ax.legend(h[::-1], l[::-1], loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False, fontsize=7)
fig.savefig(ROOT / 'figures/paper_mass_summary.png', dpi=250)
