"""Paper figure (appendix): how the rotation and the rotational support of the giant [Fe/H] bins depend on the disc model.

Maximum-likelihood fits (lmcold.kinfit.fit, proper motions and line-of-sight velocities, 2 < rho < 9 deg, release giants in
their measured bins) of each bin with every model variant, from the tables of the diagnostic scripts:
- 'sym' (Level 1) and 'stream' (Level 2), 'tilt + stream' (inner ring rho < 4.5 deg in its own plane): results/kin_tilted_ring.csv
  (scripts/fig_kin_tilted_ring.py);
- 'qlos + stream' (internal motion scaled by 1 + qlos along the line of sight): results/kin_los_scale.csv, velocities 'all'
  (scripts/fig_kin_los_scale.py);
- 'turn' and 'turn + stream' (inclination and line of nodes changing in time): results/kin_turning.csv (scripts/fig_kin_turning.py).
Black: the Level-2 posterior medians of results/kin_support.csv (the main-text values). x: the median [Fe/H] of each bin
(results/kin_support.csv), each model offset slightly for clarity.
Two equal panels: (a) v0 [km/s], (b) v0 / sigma_los. Writes figures/paper_kin_sensitivity.png and prints the table.
Run: .venv/bin/python scripts/paper_kin_sensitivity.py   (after the three diagnostic scripts; run time not recorded)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from lmcold.layout import panel_grid
from lmcold.sources import ROOT

# model label: (table, model name in it, colour, line style)
MODELS = {'Level 1 (symmetric)': ('kin_tilted_ring', 'sym', '0.6', '-'),
          'Level 2 (ML)': ('kin_tilted_ring', 'stream', 'C0', '-'),
          'tilted inner ring + streaming': ('kin_tilted_ring', 'tilt + stream', 'C2', '-'),
          'LOS scale + streaming': ('kin_los_scale', 'qlos + stream', 'C1', '-'),
          'turning disc': ('kin_turning', 'turn', 'C4', '--'),
          'turning disc + streaming': ('kin_turning', 'turn + stream', 'C3', '--')}

# %% [markdown]
# ### Load the data
# The Level-2 posterior values and median [Fe/H] of each group from `results/kin_support.csv` (written by
# `scripts/paper_kin_support.py`). The tables of the model variants (`results/kin_tilted_ring.csv`, `results/kin_los_scale.csv`,
# `results/kin_turning.csv`, from the diagnostic scripts) are read in the loop below, one per entry of `MODELS`.

# %%
sup = pd.read_csv(ROOT / 'results/kin_support.csv')  # Level-2 posteriors per group: v0, ratio, feh_med, ...

# %% [markdown]
# ### The model variants
# The maximum-likelihood $v_0$ and $\sigma_{\rm los}$ of each giant bin in every model variant, collected from the tables of the
# diagnostic scripts, with the bin's median [Fe/H]; the printed table gives $v_0/\sigma_{\rm los}$.

# %%
# giant bins only (kin_support.csv also has other populations)
sup = sup[sup.population.str.startswith('giants')]
rows = []
for lab, (tab, model, _, _) in MODELS.items():
    # the table of this model variant
    t = pd.read_csv(ROOT / f'results/{tab}.csv')
    # kin_los_scale.csv: only the fits with the velocities of both surveys
    if 'velocities' in t:
        t = t[t.velocities == 'all']
    # rows of this model, giant bins only
    t = t[(t.model == model) & t['sample'].isin(sup.population)]
    # v0, sigma_los [km/s] and their ratio per bin
    rows += [dict(model=lab, population=r['sample'], v0=r.v0, sigma_los=r.sigma_los, ratio=r.v0 / r.sigma_los) for _, r in t.iterrows()]
# add the median [Fe/H] [dex] of each bin
t = pd.DataFrame(rows).merge(sup[['population', 'feh_med']], on='population')
# v0 / sigma_los, one column per model
print(t.pivot(index='population', columns='model', values='ratio').round(2).to_string())

# %% [markdown]
# ### The figure
# (a) $v_0$ and (b) $v_0/\sigma_{\rm los}$ against [Fe/H], one line per model (offset slightly in [Fe/H]); black squares: the
# Level-2 posteriors.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# two equal panels, (a) v0 and (b) v0 / sigma_los
fig, ax = panel_grid(1, 2, width=6.0, left=0.55, right=0.1, hgap=0.6, top=0.6)
for j, (lab, (_, _, colour, ls)) in enumerate(MODELS.items()):
    # rows of this model, ordered in [Fe/H]
    s = t[t.model == lab].sort_values('feh_med')
    for a, q in zip(ax[0], ('v0', 'ratio')):
        # offset in [Fe/H] by 0.01 dex per model so points do not overlap
        a.plot(s.feh_med + 0.01 * (j - 2.5), s[q], 'o', ls=ls, color=colour, ms=3, lw=0.8, label=lab)
# Level-2 posterior medians with their 16th-84th percentile errors
for a, q, lo, hi in zip(ax[0], ('v0', 'ratio'), ('v0_p16', 'ratio_lo'), ('v0_p84', 'ratio_hi')):
    a.errorbar(sup.feh_med, sup[q], yerr=[sup[q] - sup[lo], sup[hi] - sup[q]], fmt='s', color='k', ms=3.5, lw=0.8,
               label='Level 2 (posterior; main text)')
ax[0, 0].set(xlabel='[Fe/H]', ylabel=r'$v_0$ [km s$^{-1}$]')
ax[0, 1].set(xlabel='[Fe/H]', ylabel=r'$v_0/\sigma_{\rm los}$')
# panel letters inside the top-left corners
for a, letter in zip(ax[0], 'ab'):
    a.text(0.04, 0.92, f'({letter})', transform=a.transAxes, fontsize=7)
# one legend for all lines, above panel (a)
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=4, fontsize=5, frameon=False)
fig.savefig(ROOT / 'figures/paper_kin_sensitivity.png')
