"""Paper figure (Section 6.5, Fig. sims_settle): how quickly the inner dark-matter halo of the TNG50 progenitors lines up with the
stellar disc and becomes round, and when its quadrupole falls below the level at which cohorts keep their orbital planes.

Input: results/sims_halo_settle.csv and results/sims_halo_cohorts.csv (scripts/sims_halo_settle.py; 60 progenitors, dark matter only,
z = 4-1); results/sims_heating_coherence.csv for the coherence of randomly oriented orbits, 1 / sqrt(median cohort size) of
Fig. sims_heating c (C_RANDOM).
Statistic: per-galaxy median in 0.5-Gyr lookback bins, then the median and 16th-84th percentiles over galaxies (bins with >= 20).
(a) angle psi between the halo minor axis and the stellar spin; (b) axis ratios b/a, c/a; (c) eps2 and its tilt and in-plane
elongation parts at 2 kpc. Dotted lines: the values beyond which the cohorts born there lose a small fraction of their available
coherence over the next 0.4-0.5 Gyr, (C0 - C1) / (C0 - C_RANDOM): psi < 10 deg, c/a > 0.9, b/a > 0.95, eps2 < 0.003 (one variable
at a time; printed with the medians on both sides).
Writes figures/paper_sims_settle.png; prints the medians per bin.
Run: .venv/bin/python scripts/manifest.py run paper_sims_settle.py   (after scripts/sims_halo_settle.py; 2 s in the recorded run of
docs/run_times.txt)"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lmcold.layout import panel_grid
from lmcold.sources import ROOT


# %% [markdown]
# ### Load the data
# The figure reads three result tables, all made from the 60 TNG50 progenitors, and no simulation (the simulations are needed only
# to remake the tables): `results/sims_halo_settle.csv` and `results/sims_halo_cohorts.csv` (`scripts/sims_halo_settle.py`: the
# dark-matter halo at each tracked snapshot, and at each cohort's birth with the cohort's loss of coherence) and
# `results/sims_heating_coherence.csv` (`scripts/build_sims_heating.py`: the cohort sizes that set the coherence of randomly oriented
# orbits).

# %%
T = pd.read_csv(ROOT / 'results/sims_halo_settle.csv', dtype={'halo': str})   # misalignment, axis ratios, eps2 per progenitor and snapshot
L = pd.read_csv(ROOT / 'results/sims_halo_cohorts.csv', dtype={'halo': str})   # per cohort: halo state at birth; C0, C1, loss
coh = pd.read_csv(ROOT / 'results/sims_heating_coherence.csv', dtype={'halo': str})   # 100-Myr cohorts: size n0 and C against time

# %% [markdown]
# ### The halo measurements and the thresholds
# `T`: the misalignment, axis ratios and quadrupole of the dark matter of each progenitor at each tracked snapshot. For each quantity
# of `QUIET`, the median fraction of coherence lost by the cohorts born on the quiet side of the line and by the others (`L`).

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
EPS_QUIET = 0.003   # eps2 below which cohorts keep their orbital planes (Fig. sims_scramble c)
QUIET = {'misalign': (0, 10), 'c_a': (0.9, 1), 'b_a': (0.95, 1), 'eps2': (0, EPS_QUIET)}   # the side on which cohorts keep their planes

# C_RANDOM: coherence of randomly oriented orbits, 1 / sqrt(median cohort size n0) over the progenitors' cohorts ('full' set)
T['tb'] = (T.lookback * 2).round() / 2   # Gyr: lookback time rounded to 0.5-Gyr bins
C_RANDOM = 1 / np.sqrt(coh[coh.set == 'full'].groupby(['halo', 's_birth']).n0.first().median())
# frac: fraction of the available coherence lost by each cohort, (C0 - C1) / (C0 - C_RANDOM)
L['frac'] = L.loss / (L.C0 - C_RANDOM)
# for each quantity: median fraction lost by the cohorts born on the quiet side (lo, hi] and by the others
for q, (lo, hi) in QUIET.items():
    k = (L[q] > lo) & (L[q] <= hi)
    print(f'cohorts with {lo} < {q} <= {hi}: median fraction lost {L.frac[k].median():.3f} (n {k.sum()}); others {L.frac[~k].median():.3f}')
print(f'{len(L)} cohorts, C_RANDOM {C_RANDOM:.3f}')



# %% [markdown]
# ### The figure
# `band` draws, for one quantity, the median over the galaxies of each galaxy's median in 0.5-Gyr lookback bins and its 16th-84th
# percentiles (bins with at least 20 galaxies), and prints them.

# %%
def band(a, q, c, lab, ls='-'):
    """Draw on axis a the median over the galaxies of each galaxy's median of column q of T per 0.5-Gyr lookback bin, with its
    16th-84th percentiles shaded (colour c, label lab, line style ls; bins with at least 20 galaxies), and print the values."""
    # per bin tb and galaxy: the galaxy's median; then, per bin, the distribution over galaxies
    g = T.groupby(['tb', 'halo'])[q].median().groupby(level=0)
    # m: median, lo/hi: 16th/84th percentiles, n: number of galaxies
    s = pd.DataFrame({'m': g.median(), 'lo': g.quantile(.16), 'hi': g.quantile(.84), 'n': g.size()})
    # bins with at least 20 galaxies
    s = s[s.n >= 20]
    # shaded 16th-84th percentile band and the median line
    a.fill_between(s.index, s.lo, s.hi, color=c, alpha=0.15, lw=0)
    a.plot(s.index, s.m, ls, c=c, lw=1.4, label=lab)
    # printed from the earliest bin to the latest
    print(f'{q}: ' + '; '.join(f'{t:.1f} Gyr {r.m:.4f} ({r.lo:.4f}-{r.hi:.4f}) n {r.n}' for t, r in s.iloc[::-1].iterrows()))


# one row of three equal panels
fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.1, hgap=0.62, top=0.62)
ax = ax[0]
# (a) angle between the halo minor axis and the stellar spin [deg]; dotted: the 10-deg threshold of QUIET
band(ax[0], 'misalign', 'k', 'TNG50 DM, median and 16-84 per cent')
ax[0].axhline(10, ls=':', c='0.3', lw=0.8)
ax[0].text(7.7, 8, 'below: cohorts keep their planes', fontsize=5, color='0.3', ha='right', va='top')
ax[0].set(ylabel='halo minor axis to stellar spin [deg]', ylim=(0, 90))
# (b) axis ratios b/a and c/a with their thresholds 0.95 and 0.9
band(ax[1], 'b_a', 'C0', 'intermediate-to-major $b/a$')
band(ax[1], 'c_a', 'C3', 'minor-to-major $c/a$')
ax[1].axhline(0.95, ls=':', c='C0', lw=0.8); ax[1].axhline(0.9, ls=':', c='C3', lw=0.8)
ax[1].text(7.7, 0.957, 'above the line of each colour: cohorts keep their planes', fontsize=5, color='0.3', ha='right', va='bottom')
ax[1].set(ylabel='DM axis ratio within 3 kpc', ylim=(0.5, 1.02))
# (c) the tilt and in-plane elongation parts of eps2, and eps2 itself, with the threshold EPS_QUIET
band(ax[2], 'eps_tilt', 'C0', r'tilt, $\epsilon_{\rm tilt}$', '--')
band(ax[2], 'eps_elong', 'C3', r'elongation in the plane, $\epsilon_{\rm elong}$', '--')
band(ax[2], 'eps2', 'k', r'both, $\epsilon_2$ (as in Fig. 19c)')
ax[2].axhline(EPS_QUIET, ls=':', c='0.3', lw=0.8)
ax[2].text(7.7, EPS_QUIET / 1.15, r'$\epsilon_2$ below this: cohorts' + '\nkeep their orbital planes', fontsize=5, color='0.3', ha='right',
           va='top')
ax[2].set(yscale='log', ylim=(1e-3, 0.03), ylabel=r'DM $\epsilon_2=|\Phi_2|/|\Phi_0|$ at 2 kpc')
# log y ticks labelled with plain numbers
ax[2].set_yticks([0.001, 0.002, 0.005, 0.01, 0.02], ['0.001', '0.002', '0.005', '0.01', '0.02']); ax[2].minorticks_off()
# common x axis (lookback time decreasing to the right), titles and legends above the panels
for a, t in zip(ax, ('(a) the halo lines up', '(b) the halo rounds slowly', '(c) the quadrupole decays')):
    a.set(xlabel='lookback time [Gyr]', xlim=(12.4, 7.6))
    a.set_title(t, loc='left', fontsize=8)
    h, l = a.get_legend_handles_labels()
    a.legend(h, l, frameon=False, fontsize=5, ncol=1 if a is ax[0] else 2, loc='lower left', bbox_to_anchor=(-0.02, 1.09),
             handlelength=1.8, columnspacing=0.7, handletextpad=0.4, borderaxespad=0)
fig.savefig(ROOT / 'figures/paper_sims_settle.png', dpi=300)
