"""Letter figure: how fast the early simulated cohorts lose their rotation, what does it, and why only early.

Statistic: per galaxy the median in each time bin, then the median over galaxies; the shading
spans the 16th-84th percentiles, darkest at the median and fading towards both percentiles (density_shade).
(a) How fast and what does it: TNG50 cohorts born at z = 1.5-4 (results/sims_scramble.csv, scripts/sims_scramble.py: the 25 Myr of
    stars before a birth snapshot, 92 cohorts in 16 progenitors; as Fig. sims_scramble a): v_phi/sigma_z about the cohort's spin at
    its birth snapshot, held fixed (vs_meas_birth, vs_sim_birth), of the members at r < 3 kpc and cylindrical R > 0.5 kpc at that
    snapshot, followed wherever they go, against time since the birth snapshot in bins 0 / 0.05-0.2 / 0.2-0.35 / 0.35-0.5 Gyr: the
    real stars, and the same stars re-run as test particles in parts of the gravity: all gravity frozen at birth (E); only the dark
    matter non-round, the rest made axisymmetric (G_DM); everything axisymmetric about the cohort's spin (D0). (At formation, from
    BirthVel, the same galaxies' stars of these epochs have 2.05, 16-84 per cent 1.48-2.58: Fig. sims_birth.) The TNG50
    star-forming gas (results/sims_heating_gas.csv, scripts/build_sims_heating.py: 60 progenitors, 0.5 < R < 3 kpc, about its own
    spin; blue) at z >= Z_GAS, per galaxy the median over those snapshots; shading as for the stars, over these 60 values. Dotted: the LMC's [Fe/H] < -1.5 giants (results/kin_support_ellipsoid.csv, ratio_z). Inner tick labels on the time axis: t / T_orb, T_orb the
    median orbital period at 2 kpc of the progenitors at the early cohorts' births (results/sims_vcirc.csv, scripts/
    build_sims_vcirc.py).
(b) What sets the loss: the real stars of (a), v_phi/sigma_z (fixed birth axis) in T_AFTER Gyr after the birth snapshot (per cohort the
    median), against the dark-matter quadrupole eps2 = |Phi_2| / |Phi_0| at 2 kpc of the cohort's galaxy at its birth snapshot
    (results/sims_halo_settle.csv, scripts/sims_halo_settle.py: AGAMA multipole of the dark matter in the disc frame; the l = 2 terms
    non-axisymmetric about the disc, |m| = 1 tilt and |m| = 2 in-plane elongation); quartiles of eps2, median and 16th-84th
    percentiles over cohorts. Dashed: the cohorts' median at the birth snapshot; dotted: the LMC.
(c) How the halo evolves: against lookback time (0.5-Gyr bins; per galaxy the median, then the median over the 60 progenitors of
    results/sims_halo_settle.csv): the angle psi between the halo's minor axis (dark matter within 3 kpc) and the stellar spin (left
    axis, 16th-84th percentiles shaded), and on the right axis eps2 and its part from the halo's elongation in the disc plane
    (eps_elong; the other part, the tilt of the flattening to the disc, eps_tilt, follows psi and is printed for the text).
Writes figures/paper_letter_scramble.png.
Run: .venv/bin/python scripts/manifest.py run paper_letter_scramble.py
"""
# %% [markdown]
# ### Settings and helpers
# The time bins, and two helpers: `gal_median` takes the median per galaxy and then over galaxies, and `density_shade` draws the
# spread between the 16th and 84th percentiles, darkest at the median.

# %%
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from lmcold.layout import panel_grid
from lmcold.sources import ROOT

T_AFTER = (0.2, 0.35)       # Gyr after the birth snapshot: v_phi/sigma_z in (b), about 1-1.5 orbits
TIME_EDGES = (0.0, 0.05, 0.2, 0.35, 0.5)   # Gyr since the birth snapshot: bins of (a)
N_EPS = 4                  # (b): cohorts in quantiles of eps2
EPOCH = 0.5                # Gyr, lookback-time bins of (c)
Z_GAS = 2.0                # star-forming gas in (a) at z >= Z_GAS (lookback > 10.4 Gyr), when the oldest cohorts formed
RES = ROOT / 'results'
GAS_COLOUR = '#2b6cb0'      # blue of the gas shading in (a) (PI); the birth colour of Fig. sims_birth
LEG = dict(fontsize=5, frameon=False, handlelength=1.5, columnspacing=0.8)   # legends in the top right corners (PI)


def gal_median(d, x, y):
    """Per-galaxy median of y in each value of column x (the plotted time), then the median and 16th/84th percentiles over galaxies.

    d: table with columns x, y and halo (one row per cohort, variant and time); returns a DataFrame with columns t (the values of x),
    m (median), lo, hi (16th, 84th percentiles over the galaxies), one row per value of x."""
    g = d.groupby([x, 'halo'])[y].median().groupby(level=0)
    return pd.DataFrame({'m': g.median(), 'lo': g.quantile(.16), 'hi': g.quantile(.84)}).rename_axis('t').reset_index()


def density_shade(a, x, samples, colour, peak, ylim):
    """Shade the spread of the values in samples[i] (one array per x[i], x increasing) between their 16th and 84th percentiles,
    darkest at the median and fading towards both percentiles: opacity peak * (1 - |2F - 1| / 0.68), F the fraction of values below
    (from the percentiles, linear between them), with the percentiles linear in x (in log x on a log axis) between the nodes, on a
    grid over ylim. (A kernel density of 16 values is nearly flat inside this range: its smoothing width is about the range itself.)

    a: axes to draw on; x: the n positions on the horizontal axis; samples: n arrays of values (axis units of y); colour: matplotlib
    colour; peak: largest opacity (0-1); ylim: (bottom, top) of the shaded grid, in the units of y."""
    q = np.linspace(.16, .84, 35)
    Q = np.array([np.quantile(v, q) for v in samples])
    f = np.log if a.get_xscale() == 'log' else (lambda z: z)
    xs = np.linspace(f(x[0]), f(x[-1]), 400)
    y = np.linspace(*ylim, 600)
    img = np.zeros((len(y), len(xs)))
    for j, Qx in enumerate(np.array([np.interp(xs, f(np.asarray(x)), Q[:, i]) for i in range(len(q))]).T):
        F = np.interp(y, Qx, q, left=np.nan, right=np.nan)
        img[:, j] = np.nan_to_num(1 - np.abs(2 * F - 1) / 0.68)
    cmap = LinearSegmentedColormap.from_list('', [to_rgba(colour, 0), to_rgba(colour, peak)])
    a.pcolormesh(np.exp(xs) if f is np.log else xs, y, img, cmap=cmap, vmin=0, vmax=1, shading='nearest', lw=0, antialiased=False, zorder=0)


# %% [markdown]
# ### Load the data
# The figure reads five result tables and no simulation; the TNG50 progenitors are needed only to remake the four simulation tables:
# `results/sims_scramble.csv` (`scripts/sims_scramble.py`: the early cohorts' real stars and their test-particle re-runs),
# `results/sims_heating_gas.csv` (`scripts/build_sims_heating.py`: the star-forming gas of 60 progenitors),
# `results/sims_halo_settle.csv` (`scripts/sims_halo_settle.py`: shape and quadrupole of the dark matter of the same 60 progenitors)
# and `results/sims_vcirc.csv` (`scripts/build_sims_vcirc.py`: their orbital period at 2 kpc). The LMC value is the [Fe/H] < -1.5
# giants' $v_{\rm rot}/\sigma_z$ from the stage-1 ellipsoid fits, `results/kin_support_ellipsoid.csv`.

# %%
sc = pd.read_csv(RES / 'sims_scramble.csv', dtype={'halo': str}).replace([np.inf, -np.inf], np.nan)   # v_phi/sigma_z per cohort, variant, time; inf: 1-2 particles left
gas = pd.read_csv(RES / 'sims_heating_gas.csv', dtype={'halo': str}).query('z >= @Z_GAS')   # star-forming gas per progenitor and snapshot, z >= Z_GAS
T = pd.read_csv(RES / 'sims_halo_settle.csv', dtype={'halo': str})   # dark-matter misalignment, axis ratios, eps2 per progenitor and snapshot
t_orb = pd.read_csv(RES / 'sims_vcirc.csv').t_orb.median()   # median orbital period at 2 kpc at the early cohorts' births [Gyr]
lmc = pd.read_csv(RES / 'kin_support_ellipsoid.csv').set_index('population').ratio_z['giants [Fe/H] < -1.5']   # LMC v_rot/sigma_z

# %% [markdown]
# ### (a) How fast the rotation is lost, and what does it
# $v_\phi/\sigma_z$ of the early TNG50 cohorts against time since birth (`sc`, from `results/sims_scramble.csv`):
# the real stars, and the same stars moved as test particles in the gravity frozen at birth, with only the dark matter aspherical,
# and made axisymmetric about the cohort's spin. Also the star-forming gas (`gas`) and the LMC's [Fe/H] < -1.5 giants (`lmc`).
# The orbital period at 2 kpc (`t_orb`) labels the time axis.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, ax = panel_grid(1, 3, width=7.0, left=0.45, right=0.5, hgap=0.55, top=0.3)
ax = ax[0]

# (a) how fast and what does it: v_phi/sigma_z of the early cohorts about their own spin, real stars and test particles in parts of
# the gravity; the star-forming gas they formed from and the LMC's oldest giants
sc = pd.concat([sc, sc[sc.variant == 'A'].assign(variant='real', vs_sim_birth=lambda d: d.vs_meas_birth)]).rename(columns={'t': 'dt'})
sc['tb'] = sc.groupby(pd.cut(sc.dt, np.r_[-1e-9, TIME_EDGES[1:-1], TIME_EDGES[-1] + 1e-6])).dt.transform('median')   # plotted at the bins' median time
# variant of results/sims_scramble.csv: legend label, colour, line width ('real': the real stars, vs_meas_birth)
VAR = {'real': ('stars', 'k', 1.6), 'E': ('frozen gravity', 'C1', 1.0), 'G_DM': ('DM aspherical only', 'C3', 1.0),
       'D0': ('axisymmetric', 'C4', 1.0)}
a = ax[0]
g = (gas.vphi / gas.sz).groupby(gas.halo).median()   # star-forming gas v_phi/sigma_z, per galaxy the median over its snapshots
density_shade(a, (-0.03, 0.46), (g, g), GAS_COLOUR, 0.6, (0, 3.5))
a.fill_between([], [], color=GAS_COLOUR, alpha=0.45, lw=0, label='gas')   # legend entry for the shading
print(f'(a) TNG50 star-forming gas at z >= {Z_GAS}: {g.median():.2f} ({g.quantile(.16):.2f}-{g.quantile(.84):.2f}), {len(g)} galaxies')
for v, (lab, col, lw) in VAR.items():
    s = gal_median(sc[sc.variant == v], 'tb', 'vs_sim_birth')
    if v == 'real':
        per = sc[sc.variant == v].groupby(['tb', 'halo']).vs_sim_birth.median().dropna()
        density_shade(a, s.t.values, [per[t].values for t in s.t], 'k', 0.35, (0, 3.5))
    a.plot(s.t, s.m, '-o', color=col, ms=2.5, lw=lw, label=lab)
    print(f'(a) {v}: ' + '; '.join(f'{r.t:.2f} Gyr: {r.m:.2f}' for r in s.itertuples()))
a.axhline(lmc, color='k', lw=0.8, ls=':')
a.text(0.45, lmc - 0.05, r'LMC, [Fe/H] $<-1.5$', ha='right', va='top', fontsize=6)
a.set(xlabel='time since birth [Gyr]', ylabel=r'$v_\phi/\sigma_z$', xlim=(-0.03, 0.46), ylim=(-0.5, 3.5))   # below 0: room for the t/T_orb labels
a.tick_params(axis='x', which='both', direction='out', top=False)   # time since birth: ticks out, below the axis
orb = a.secondary_xaxis('bottom', functions=(lambda x: x / t_orb, lambda x: x * t_orb))   # t/T_orb: ticks in, labels inside
orb.tick_params(axis='x', which='both', direction='in', pad=-11, labelsize=6)
orb.tick_params(axis='x', which='minor', length=0)
orb.spines['bottom'].set_visible(False)
orb.set_xticks([0.5, 1, 1.5, 2])
orb.set_xticks([0.02 / t_orb], [r'$t/T_{\rm orb}$'], minor=True)   # the scale's name where its 0 would be; T_orb printed for the caption
print(f'T_orb {t_orb:.3f} Gyr, LMC v/sigma {lmc:.2f}')

# %% [markdown]
# ### (b) What sets the loss
# The $v_\phi/\sigma_z$ that each cohort keeps 0.2-0.35 Gyr after birth, against the dark-matter quadrupole $\epsilon_2$ at 2 kpc of its
# galaxy at its birth (`T`, from `results/sims_halo_settle.csv`), in quartiles of $\epsilon_2$.

# %%
# (b) what sets the loss: v_phi/sigma_z of the same cohorts after T_AFTER Gyr against the halo's quadrupole at their birth snapshot
real = sc[sc.variant == 'real']
after = real[(real.dt > T_AFTER[0]) & (real.dt <= T_AFTER[1])].groupby(['halo', 's_birth']).vs_sim_birth.median().rename('vs')
k = after.reset_index().merge(T, left_on=['halo', 's_birth'], right_on=['halo', 'snap'], how='inner')
g = k.groupby(pd.qcut(k.eps2, N_EPS), observed=True)
# per quartile of eps2: its median eps2 (x), the median and 16th/84th percentiles of v_phi/sigma_z after T_AFTER, and the cohorts
q = pd.DataFrame({'x': g.eps2.median(), 'm': g.vs.median(), 'lo': g.vs.quantile(.16), 'hi': g.vs.quantile(.84), 'n': g.size()})
a = ax[1]
a.set_xscale('log')
density_shade(a, q.x.values, [x.vs.values for _, x in g], 'k', 0.35, (-0.3, 3.5))
a.plot(q.x, q.m, 'k-o', lw=1.6, ms=2.5, label=f'stars after {T_AFTER[0]:.1f}-{T_AFTER[1]:.2f} Gyr')
v0 = gal_median(real, 'tb', 'vs_sim_birth').m.iloc[0]
a.axhline(v0, color='k', lw=0.8, ls='--', label='at birth')
a.axhline(lmc, color='k', lw=0.8, ls=':', label=r'LMC, [Fe/H] $<-1.5$')
a.set(xscale='log', xlim=(0.003, 0.025), ylim=(-0.5, 3.5), xlabel=r'DM quadrupole $\epsilon_2$ at 2 kpc, at birth', ylabel=r'$v_\phi/\sigma_z$')
a.set_xticks([0.003, 0.005, 0.01, 0.02], ['0.003', '0.005', '0.01', '0.02'])
a.minorticks_off()
print(f'(b) {len(k)} cohorts; Spearman of v/sigma after with eps2 {k.vs.corr(k.eps2, method="spearman"):.2f}, eps_tilt '
      f'{k.vs.corr(k.eps_tilt, method="spearman"):.2f}, psi {k.vs.corr(k.misalign, method="spearman"):.2f}, lookback '
      f'{k.vs.corr(k.lookback, method="spearman"):.2f}; quartiles: ' + '; '.join(f'{r.x:.4f}: {r.m:.2f} ({r.lo:.2f}-{r.hi:.2f}) n {r.n}' for r in q.itertuples()))
a.legend(loc='upper right', **LEG)


# %% [markdown]
# ### (c) How the halo settles
# Against lookback time: the angle $\psi$ between the minor axis of the dark matter within 3 kpc and the stellar spin, and
# $\epsilon_2$ with its part from the halo's elongation in the disc plane. The figure is saved at the end of this cell.

# %%
def by_epoch(d, y):
    """Per-galaxy median of y in 0.5-Gyr bins of lookback time, then the median and 16th/84th percentiles over galaxies."""
    g = d.assign(tb=(d.lookback / EPOCH).round() * EPOCH).groupby(['tb', 'halo'])[y].median().groupby(level=0)
    return pd.DataFrame({'m': g.median(), 'lo': g.quantile(.16), 'hi': g.quantile(.84)})


# (c) how the halo evolves: misalignment and quadrupole against time
a = ax[2]
psi = by_epoch(T, 'misalign')   # deg: angle between the halo's minor axis and the stellar spin, per 0.5-Gyr bin of lookback time
tb = (T.lookback / EPOCH).round() * EPOCH
per = T.groupby([tb, 'halo']).misalign.median()
density_shade(a, psi.index.values, [per[t].values for t in psi.index], 'C3', 0.35, (0, 80))
a.plot(psi.index, psi.m, '-o', color='C3', lw=1.4, ms=2.5, label=r'misalignment $\psi$')
a.set(xlim=(12.7, 7.3), ylim=(0, 80), xlabel='lookback time [Gyr]', ylabel=r'$\psi$ [deg]')
r = a.twinx()
EPS = {'eps2': ('k', '-', r'DM quadrupole $\epsilon_2$'), 'eps_elong': ('C0', '--', r'in-plane part $\epsilon_{\rm elong}$')}   # tilt part: printed
for qn, (col, ls, lab) in EPS.items():
    e = by_epoch(T, qn)
    r.plot(e.index, e.m, ls, color=col, lw=1.2, label=lab)
for qn in ('eps2', 'eps_tilt', 'eps_elong'):
    e = by_epoch(T, qn)
    print(f'(c) {qn}: ' + '; '.join(f'{t:.1f}: {v:.4f}' for t, v in e.m.items()))
r.set(yscale='log', ylim=(1e-3, 0.04))
r.set_ylabel(r'DM quadrupole $\epsilon_2$ at 2 kpc')
print('(c) psi: ' + '; '.join(f'{t:.1f}: {v:.0f}' for t, v in psi.m.items()))
h3 = [h for x in (a, r) for h in x.get_legend_handles_labels()[0]]
a.legend(h3, [h.get_label() for h in h3], loc='upper right', **LEG)

for a, title in zip(ax, ('(a) loss of rotation', '(b) role of halo shape', '(c) halo settling')):
    a.set_title(title, loc='left', fontsize=8)   # above the panel, as Fig. sims_sidm
ax[0].legend(loc='upper right', ncol=2, **LEG)
fig.savefig(ROOT / 'figures/paper_letter_scramble.png', dpi=250)
print('wrote figures/paper_letter_scramble.png')
