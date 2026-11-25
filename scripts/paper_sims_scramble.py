"""Paper figure (Section 6.4, Fig. sims_scramble): what turns the orbital planes of the early TNG50 cohorts, and the numbers of the text.

Inputs: results/sims_scramble.csv (scripts/sims_scramble.py: 92 cohorts of 25 Myr in 16 progenitors, test particles in parts of the gravity);
results/sims_halo_cohorts.csv (scripts/sims_halo_settle.py: 623 cohorts of 60 progenitors, eps2 at birth and loss over the next track
step); results/sims_heating_coherence.csv for C_RANDOM = 1 / sqrt(median cohort size) of Fig. sims_heating c.
Loss of a cohort in a variant = C at the birth snapshot - C at time t (the real stars: C_meas).
(a) C against time since the birth snapshot, in bins 0 / 0.05-0.2 / 0.2-0.35 / 0.35-0.5 Gyr: the median over galaxies of each galaxy's
    median (16th-84th percentiles shaded for the real stars): real stars, all gravity (A), frozen at birth (E), axisymmetric (D0),
    only the DM non-round (G_DM), only gas and stars non-round (G_bar).
(b) mean loss at 0.2-0.35 Gyr against the angle of the halo minor axis to the cohort's spin at birth (bins 0-15-30-45-90 deg; 16th-84th
    percentiles of 500 bootstrap resamples of the cohorts), frozen at birth: real stars, E_split, H2, H1, D0_fr.
(c) fraction of the available coherence lost over the next 0.40-0.50 Gyr, (C0 - C1) / (C0 - C_RANDOM), against eps2 at birth (bins
    0-0.003-0.006-0.01-0.015-0.1): median and 16th-84th percentiles.
Printed (text of Section 6.4): validation of A against the real stars (t > 0); each variant's loss at 0.2-0.35 Gyr summed over the
cohorts as a fraction of the real stars' (and of E_split's), with 16th-84th percentiles of 1000 bootstrap resamples of the galaxies;
the panel-(b) bin means; the panel-(c) bins.
Writes figures/paper_sims_scramble.png.
Run: .venv/bin/python scripts/manifest.py run paper_sims_scramble.py   (after scripts/sims_scramble.py and sims_halo_settle.py; 1 s in
the recorded run of docs/run_times.txt)"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lmcold.layout import panel_grid
from lmcold.sources import ROOT


# %% [markdown]
# ### Load the data
# The figure reads three result tables, all made from the TNG50 progenitors, and no simulation (the simulations are needed only to
# remake the tables): `results/sims_scramble.csv` (`scripts/sims_scramble.py`: the early cohorts' real stars and their test-particle
# re-runs), `results/sims_halo_cohorts.csv` (`scripts/sims_halo_settle.py`: the dark-matter halo at each cohort's birth and the
# cohort's loss of coherence) and `results/sims_heating_coherence.csv` (`scripts/build_sims_heating.py`: the cohort sizes that set
# the coherence of randomly oriented orbits).

# %%
a = pd.read_csv(ROOT / 'results/sims_scramble.csv', dtype={'halo': str})   # coherence C per cohort, variant and time since birth
L = pd.read_csv(ROOT / 'results/sims_halo_cohorts.csv', dtype={'halo': str})   # per cohort: eps2, misalignment at birth; C0, C1, loss
coh = pd.read_csv(ROOT / 'results/sims_heating_coherence.csv', dtype={'halo': str})   # 100-Myr cohorts: size n0 and C against time

# %% [markdown]
# ### The test-particle runs and the cohort tables
# `a`: the coherence $C$ of each cohort's real stars (`C_meas`) and of its test particles in each variant of the gravity (`C_sim`),
# against the time since the birth snapshot. The real stars become a variant of their own; the loss is $C$ at the birth snapshot minus
# $C$ at time $t$. `L` gives $\epsilon_2$ at birth and the loss over the next tracked step of each cohort; `coh` the median cohort size,
# from which $C_{\rm random}$.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# Real stars as an extra variant 'real' (C_sim := C_meas, from the rows of variant A); C0 = C at the birth snapshot (t = 0).
u = pd.concat([a, a[a.variant == 'A'].assign(variant='real', C_sim=lambda d: d.C_meas)])
u = u.join(u[u.t == 0].set_index(['variant', 'halo', 's_birth']).C_sim.rename('C0'), on=['variant', 'halo', 's_birth'])
u['loss'] = u.C0 - u.C_sim   # loss of coherence since the birth snapshot, per cohort, variant and time
# C of randomly oriented orbits: 1 / sqrt(median cohort size n0) of the full 100-Myr cohorts.
C_RANDOM = 1 / np.sqrt(coh[coh.set == 'full'].groupby(['halo', 's_birth']).n0.first().median())


# %% [markdown]
# ### Validation and the loss of each variant
# The test particles in all of the gravity against the real stars; then each variant's loss after 0.2-0.35 Gyr, summed over the
# cohorts, as a fraction of the real stars' loss, with 16th-84th percentiles of a bootstrap over the galaxies.

# %%
pa = a[(a.variant == 'A') & (a.t > 0)]   # all-gravity test particles after birth, compared with the real stars
print(f'{a.halo.nunique()} galaxies, {a.groupby("halo").s_birth.nunique().sum()} cohorts, median {a.groupby(["halo", "s_birth"]).n0.first().median():.0f} '
      f'stars; A - real at t > 0 ({len(pa)}): median {np.median(pa.C_sim - pa.C_meas):+.3f}, 16-84 {np.percentile(pa.C_sim - pa.C_meas, 16):+.3f} '
      f'{np.percentile(pa.C_sim - pa.C_meas, 84):+.3f}; Pearson r {np.corrcoef(pa.C_sim, pa.C_meas)[0, 1]:.2f}')
W = u[(u.t > 0.2) & (u.t <= 0.35)].dropna(subset=['loss'])   # the snapshots 0.2-0.35 Gyr after birth
S = W.groupby(['variant', 'halo']).loss.sum().unstack(0)            # summed loss per galaxy and variant
gal = S.index.values             # galaxy (halo) ids
rng = np.random.default_rng(1)   # fixed seed
# 1000 bootstrap resamples of the galaxies: loss summed over the drawn galaxies, per variant.
boot = [S.loc[gal[rng.integers(0, len(gal), len(gal))]].sum() for _ in range(1000)]
print('loss at 0.2-0.35 Gyr summed over cohorts, fraction of the real (16-84 bootstrap over galaxies) / of E_split:')
# Each variant's summed loss as a fraction of the real stars' (and of E_split's), with the bootstrap 16th-84th percentiles.
for v in S.columns:
    f = np.array([b[v] / b['real'] for b in boot])
    print(f'  {v:8s} {S[v].sum() / S.real.sum():.2f} ({np.percentile(f, 16):.2f}-{np.percentile(f, 84):.2f})  / E_split '
          f'{S[v].sum() / S.E_split.sum():.2f}')


# %% [markdown]
# ### (a) Which mass turns the orbits
# $C$ against time since the birth snapshot: median over the galaxies of each galaxy's median, for the real stars and five variants.

# %%
fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.1, hgap=0.62, top=0.78)
ax = ax[0]   # the three panels (a), (b), (c)
u['tb'] = pd.cut(u.t, [-1e-9, 0.05, 0.2, 0.35, 0.5 + 1e-6], labels=False)   # Gyr: time bins 0 / 0.05-0.2 / 0.2-0.35 / 0.35-0.5
# Legend labels and colours of the variants shown in (a).
LAB = {'real': 'real stars', 'A': 'all gravity', 'E': 'all, frozen at birth', 'D0': 'axisymmetric only', 'G_DM': 'only DM non-round',
       'G_bar': 'only gas, stars non-round'}
COL = {'real': 'k', 'A': 'C0', 'E': 'C1', 'D0': 'C4', 'G_DM': 'C3', 'G_bar': 'C2'}
for v in LAB:
    # Median C and t per (time bin, galaxy), grouped by time bin; the curve is the median over galaxies.
    g = u[u.variant == v].groupby(['tb', 'halo']).agg(C=('C_sim', 'median'), t=('t', 'median')).groupby(level=0)
    ax[0].plot(g.t.median(), g.C.median(), '-o', c=COL[v], lw=1.6 if v == 'real' else 1.0, ms=2.5, label=LAB[v])
    # Band: 16th-84th percentiles over galaxies, real stars only.
    if v == 'real':
        ax[0].fill_between(g.t.median(), g.C.quantile(.16), g.C.quantile(.84), color='k', alpha=0.12, lw=0)
ax[0].set(xlabel='time since birth [Gyr]', ylabel='orbital-plane coherence $C$', xlim=(0, 0.45), ylim=(0.3, 0.8))
ax[0].set_title('(a) which mass scrambles', loc='left', fontsize=8)


# %% [markdown]
# ### (b) Which shape of the halo
# Mean loss after 0.2-0.35 Gyr in bins of the angle between the halo's minor axis and the cohort's spin at birth, in the gravity frozen
# at birth, for the full halo and three changes to it; bars: 16th-84th percentiles of a bootstrap over the cohorts.

# %%
W = W.assign(mb=pd.cut(W.misalign, [0, 15, 30, 45, 90]))   # deg: bins of the halo minor axis to the cohort's spin at birth
# Legend labels and colours of the variants shown in (b).
BL = {'real': 'real stars', 'E_split': 'full halo, frozen', 'H2': 'halo aligned with stars', 'H1': 'round about minor axis',
      'D0_fr': 'round about stellar spin'}
BC = {'real': 'k', 'E_split': 'C1', 'H2': 'C0', 'H1': 'C3', 'D0_fr': 'C4'}
print('(b) mean loss at 0.2-0.35 Gyr by misalignment at birth [deg] (16-84 bootstrap over cohorts):')
rng = np.random.default_rng(0)   # fixed seed of the bootstrap over cohorts
for j, v in enumerate(BL):
    rows = []
    # Per misalignment bin: median angle [deg], mean loss, 16th-84th percentiles of 500 bootstrap means, number of cohorts.
    for mb, g in W[W.variant == v].groupby('mb', observed=True):
        bs = [g.loss.values[rng.integers(0, len(g), len(g))].mean() for _ in range(500)]
        rows.append((g.misalign.median(), g.loss.mean(), *np.percentile(bs, [16, 84]), len(g)))
    x, y, lo, hi, n = map(np.array, zip(*rows))
    # Variants offset by 1 deg in x for legibility.
    ax[1].errorbar(x + (j - 2) * 1.0, y, [y - lo, hi - y], fmt='-o', c=BC[v], lw=1.6 if v == 'real' else 1.0, ms=2.5, elinewidth=0.8,
                   capsize=1.5, label=BL[v])
    print(f'  {v:8s} ' + ', '.join(f'{xx:.0f} deg: {yy:.3f} ({l:.3f}-{h:.3f}) n {nn}' for xx, yy, l, h, nn in zip(x, y, lo, hi, n)))
ax[1].set(xlabel='halo minor axis to stellar spin at birth [deg]', ylabel='loss of $C$ after 0.2-0.35 Gyr', xlim=(0, 75), ylim=(0, 0.32))
ax[1].set_title('(b) which shape scrambles', loc='left', fontsize=8)


# %% [markdown]
# ### (c) The halo quadrupole sets the loss
# The fraction of the available coherence lost over the next tracked step, $(C_0-C_1)/(C_0-C_{\rm random})$, in bins of
# $\epsilon_2$ at birth: median and 16th-84th percentiles.

# %%
L['eb'] = pd.cut(L.eps2, [0, 0.003, 0.006, 0.01, 0.015, 0.1])   # bins of eps2 at 2 kpc at birth (dimensionless)
L['frac'] = L.loss / (L.C0 - C_RANDOM)   # fraction of the available coherence lost over the next tracked step
# Per eps2 bin: median eps2, median and 16th-84th percentiles of frac, number of cohorts, median lookback time [Gyr].
g = L.groupby('eb', observed=True)
s = pd.DataFrame({'x': g.eps2.median(), 'm': g.frac.median(), 'lo': g.frac.quantile(.16), 'hi': g.frac.quantile(.84), 'n': g.size(),
                  'lb': g.lookback.median()})
# Band and median line against eps2 (log axis).
ax[2].fill_between(s.x, s.lo, s.hi, color='k', alpha=0.12, lw=0)
ax[2].plot(s.x, s.m, 'k-o', lw=1.6, ms=2.5, label='TNG50: median, 16-84 per cent')
print(f'(c) fraction of the available coherence lost in 0.4-0.5 Gyr by eps2 at birth (C_RANDOM {C_RANDOM:.3f}): ' + '; '.join(
    f'{r.x:.4f}: {r.m:.3f} ({r.lo:.3f}-{r.hi:.3f}) n {r.n} lookback {r.lb:.1f}' for r in s.itertuples()))
ax[2].set(xscale='log', xlim=(1.5e-3, 0.03), ylim=(-0.05, 1.05), xlabel=r'DM $\epsilon_2=|\Phi_2|/|\Phi_0|$ at 2 kpc, at birth',
          ylabel='fraction of $C$ lost in 0.4-0.5 Gyr')
ax[2].set_xticks([0.002, 0.005, 0.01, 0.02], ['0.002', '0.005', '0.01', '0.02']); ax[2].minorticks_off()
ax[2].set_title('(c) the halo quadrupole sets the loss', loc='left', fontsize=8)
# Legends above each panel; errorbar containers are replaced by their line handle.
for a_ in ax:
    h, l = a_.get_legend_handles_labels()
    a_.legend([x[0] if hasattr(x, 'lines') else x for x in h], l, frameon=False, fontsize=5, ncol=2 if len(l) > 1 else 1,
              loc='lower left', bbox_to_anchor=(-0.02, 1.09), handlelength=1.8, columnspacing=0.7, handletextpad=0.4, borderaxespad=0)
fig.savefig(ROOT / 'figures/paper_sims_scramble.png', dpi=300)
