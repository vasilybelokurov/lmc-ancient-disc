"""Rotational support of the LMC giants and RR Lyrae against metallicity (paper, kinematics section).

Values: stage-1 posteriors of the L2 model (lmcold.kinfit: thin disc with free centre, bulk motion, inclination, line of nodes,
tanh rotation curve v0 tanh(R / R0), isotropic dispersions, outliers, plus m = 2 streaming; proper motions and, for the giants,
line-of-sight velocities) from scripts/fit_kinematics_mcmc.py 'data stream', each population fitted on its own at 2 < rho < 9 deg:
medians and 16th-84th percentiles. Giants: all release giants in their measured [Fe/H] bin (lmcold.samples.giants(); BOSS on the
APOGEE scale). In the bins from [Fe/H] = -1.1 up the BOSS [Fe/H] errors blur the narrow bins (scripts/paper_kin_feh_mixing.py),
so APOGEE giants alone are fitted there as the reference with true [Fe/H]: maximum likelihood (lmcold.kinfit.fit) of v0, sigma,
sigma_los, k and the outlier fraction with the rest held at the bin's posterior medians; errors from the inverse Hessian.
RR Lyrae: proper motions only, so no sigma_los; open star: their v0 in the joint (stage-2) fit, with the shared geometry; plotted at
the median photometric [Fe/H] of the members (data/rrab.parquet).
Proper-motion dispersion sigma, for every population: stage 1 of the 'outer' run (fit_kinematics_mcmc.py data stream outer, stars at
4 < rho < 9 deg), with v0 / sigma (ratio_pm) from the draws of that run. One Gaia error scale k per population fits the stars beyond
4 deg but not the crowded inner disc, where Gaia underestimates its errors more; over 2-9 deg the inner stars set k and, for the
RR Lyrae (errors ~64 km/s per component), push sigma to its lower bound (scripts/paper_kin_dispersion_cut.py).
x: median [Fe/H] of each sample's stars at 2 < rho < 9 deg; horizontal bars: 16th-84th percentiles of their [Fe/H].

Writes figures/paper_kin_support.png and results/kin_support.csv (population, n, n_los, feh_med; v0, sigma (proper motions),
sigma_los with p16 and p84; v0 / sigma_los (ratio) and v0 / sigma (ratio_pm) median, p16, p84 from the posterior draws
(data/kin_mcmc_data_stream_samples.npz, data/kin_mcmc_data_stream_outer_samples.npz); APOGEE reference values and errors where fitted). Panels (1 x 3, equal; legend
above (a)): (a) v0 [km/s], (b) sigma_los (filled) and sigma from the proper motions (open; RR Lyrae: star) [km/s], (c) v0 / sigma_los (filled)
and v0 / sigma (open; RR Lyrae: star), both from the draws.

Run: .venv/bin/python scripts/paper_kin_support.py   (after fit_kinematics_mcmc.py data stream, and data stream outer;
     9 s in the run of 2026-11-22, docs/run_times.txt)
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_kinematics_mcmc import ring
from lmcold import kinfit, samples
from lmcold.layout import panel_grid
from lmcold.sources import DATA, ROOT

# Catalogue columns passed to kinfit.data, in its argument order (deg, mas/yr, km/s).
COLS = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'vlos', 'evlos')
OWN = ('v0', 'sigma', 'sigma_los', 'k', 'logit_fout')  # refitted for the APOGEE giants of a bin; the rest held at the bin's posterior medians
APOGEE_FROM = -1.1  # [Fe/H]: bins with this lower edge or above get the APOGEE reference

# %% [markdown]
# ### Load the data
# The posterior summaries of the Level-2 model with sky-aligned dispersions (`scripts/fit_kinematics_mcmc.py`): stage 1 of each
# group at $2<\rho<9^\circ$ and the joint RR Lyrae $v_0$ (`results/kin_mcmc_data_stream.csv`), stage 1 at $4<\rho<9^\circ$
# (`results/kin_mcmc_data_stream_outer.csv`); the giant catalogues of Zenodo bundle 1 through `lmcold.samples.giants()`, split into
# the five [Fe/H] bins; the RR Lyrae through `lmcold.samples.rrab()` and, for their [Fe/H], `data/rrab.parquet`. The posterior draws
# (`data/kin_mcmc_data_stream_samples.npz`, `data/kin_mcmc_data_stream_outer_samples.npz`) are read one run at a time in the loop
# of the support ratios below.

# %%
post = pd.read_csv(ROOT / 'results/kin_mcmc_data_stream.csv').query('stage == 1')  # stage-1 summaries, 2-9 deg
outer = pd.read_csv(ROOT / 'results/kin_mcmc_data_stream_outer.csv').query('stage == 1')  # stage-1 summaries, 4-9 deg
# Joint (stage-2) v0 of the RR Lyrae, 2-9 deg.
rr2 = pd.read_csv(ROOT / 'results/kin_mcmc_data_stream.csv').query('stage == 2 & population == "RR Lyrae" & param == "v0"').iloc[0]
giants = samples.giants()  # (label, catalogue dict) per [Fe/H] bin, from data/giants_apogee.parquet and giants_boss.parquet
rrab = samples.rrab()  # RR Lyrae members with Gaia astrometry (data/rrab.parquet)
rr = pd.read_parquet(DATA / 'rrab.parquet')  # the full RR Lyrae table, with the photometric [Fe/H] column feh

# %% [markdown]
# ### The posteriors
# Lookup of the stage-1 summaries: each group at $2<\rho<9^\circ$ (run 'stream') and at $4<\rho<9^\circ$ (run 'stream_outer'), from
# which the proper-motion dispersion $\sigma_{\rm PM}$ is taken.

# %%
# q(pop, n, c): column c (median, p16, p84) of parameter n of group pop, 2-9 deg run; qo: the same, 4-9 deg run;
# pm(pop): the proper-motion dispersion sigma [km/s] and its percentiles from the 4-9 deg run.
q = lambda pop, n, c='median': float(post[(post.population == pop) & (post.param == n)][c].iloc[0])  # noqa: E731
qo = lambda pop, n, c='median': float(outer[(outer.population == pop) & (outer.param == n)][c].iloc[0])  # noqa: E731
pm = lambda pop: {f'sigma{s}': qo(pop, 'sigma', c) for s, c in (('', 'median'), ('_p16', 'p16'), ('_p84', 'p84'))}  # noqa: E731
# %% [markdown]
# ### The giant bins and the RR Lyrae
# For each [Fe/H] bin: star counts, the [Fe/H] spread, $v_0$, $\sigma_{\rm los}$ and $\sigma_{\rm PM}$; from [Fe/H] = `APOGEE_FROM` up, the
# maximum-likelihood fit of the APOGEE giants alone (the rest held at the bin's posterior medians). Then the RR Lyrae, with proper
# motions only.

# %%
rows = []
for (lo, _), (label, c) in zip(samples.FEH_BINS, giants):
    # The bin's giants at 2 < rho < 9 deg; counts, and the median and 16th-84th percentiles of their [Fe/H] [dex].
    c = ring(c)
    pop = f'giants {label}'
    r = dict(population=pop, n=len(c['ra']), n_los=int(np.isfinite(c['vlos']).sum()), feh_med=np.median(c['feh']),
             feh_p16=np.percentile(c['feh'], 16), feh_p84=np.percentile(c['feh'], 84))
    # v0 and sigma_los [km/s] (2-9 deg run) and sigma (4-9 deg run), each with p16 and p84.
    for n in ('v0', 'sigma_los'):
        r |= {n: q(pop, n), f'{n}_p16': q(pop, n, 'p16'), f'{n}_p84': q(pop, n, 'p84')}
    r |= pm(pop)
    if lo >= APOGEE_FROM:
        # Start from the bin's posterior medians wherever the summary has the parameter, kinfit.start() elsewhere.
        p = kinfit.start()
        for n in kinfit.PARAMS:
            if ((post.population == pop) & (post.param == n)).any():
                p[kinfit.PARAMS.index(n)] = q(pop, n)
        # ML fit of OWN on the APOGEE giants of the bin; their values and inverse-Hessian errors stored as *_apogee, *_apogee_err.
        a = c['survey'] == 'apogee'
        pa, ea, _ = kinfit.fit(kinfit.data(*(c[k][a] for k in COLS)), free=OWN, p0=p)
        r |= dict(n_apogee=int(a.sum()), feh_med_apogee=np.median(c['feh'][a]))
        r |= {f'{n}_apogee': pa[kinfit.PARAMS.index(n)] for n in ('v0', 'sigma_los')}
        r |= {f'{n}_apogee_err': ea[kinfit.PARAMS.index(n)] for n in ('v0', 'sigma_los')}
    rows.append(r)
# RR Lyrae members: count at 2-9 deg, photometric [Fe/H] spread, v0 (stage 1) and the 4-9 deg sigma; no line-of-sight data.
rr = rr[rr.member]
rows.append(dict(population='RR Lyrae', n=len(ring(rrab)['ra']), n_los=0, feh_med=float(rr.feh.median()),
                 feh_p16=float(rr.feh.quantile(0.16)), feh_p84=float(rr.feh.quantile(0.84)),
                 **{f'v0{s}': q('RR Lyrae', 'v0', c) for s, c in (('', 'median'), ('_p16', 'p16'), ('_p84', 'p84'))}, **pm('RR Lyrae')))
# %% [markdown]
# ### The support ratios
# $v_0/\sigma_{\rm los}$ and $v_0/\sigma_{\rm PM}$ draw by draw from the stage-1 posterior samples; the RR Lyrae $v_0$ of the joint
# fit; the APOGEE ratios with propagated errors. The table goes to `results/kin_support.csv`.

# %%
t = pd.DataFrame(rows)
# v0 / sigma_los (2-9 deg run) and v0 / sigma (outer run) draw by draw from the stage-1 posterior samples (keys '1|<i>|<param>_0',
# populations in the run's order).
for col, n, run in (('ratio', 'sigma_los', 'stream'), ('ratio_pm', 'sigma', 'stream_outer')):
    draws = np.load(DATA / f'kin_mcmc_data_{run}_samples.npz')
    for i, pop in enumerate(draws['names']):
        if pop in set(t.population) and f'1|{i}|{n}_0' in draws:
            r = (draws[f'1|{i}|v0_0'] / draws[f'1|{i}|{n}_0']).reshape(-1)
            t.loc[t.population == pop, [col, f'{col}_lo', f'{col}_hi']] = np.percentile(r, [50, 16, 84])
# APOGEE ratio v0 / sigma_los with the relative errors added in quadrature.
t['ratio_apogee'] = t.v0_apogee / t.sigma_los_apogee
t['ratio_apogee_err'] = t.ratio_apogee * np.hypot(t.v0_apogee_err / t.v0_apogee, t.sigma_los_apogee_err / t.sigma_los_apogee)
t.to_csv(ROOT / 'results/kin_support.csv', index=False, float_format='%.4f')
print(t[['population', 'n', 'feh_med', 'feh_p16', 'feh_p84', 'v0', 'sigma', 'sigma_los', 'ratio', 'ratio_pm', 'v0_apogee', 'sigma_los_apogee', 'ratio_apogee']].round(2).to_string(index=False))

# %% [markdown]
# ### The figure
# Three panels against [Fe/H]: $v_0$, the two dispersions and the two support ratios. Black: all giants; red squares: APOGEE alone;
# blue stars: RR Lyrae.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# Three equal panels; g: the giant bins, rl: the RR Lyrae row.
fig, ax = panel_grid(1, 3, width=7.0, left=0.5, right=0.12, hgap=0.55, top=0.4, bottom=0.42)
g, rl = t[t.population != 'RR Lyrae'], t[t.population == 'RR Lyrae']
# Asymmetric error bars from the percentiles: err for the summary columns (p16, p84), rerr for the ratios (lo, hi).
err = lambda s, n: [s[n] - s[f'{n}_p16'], s[f'{n}_p84'] - s[n]]  # noqa: E731
rerr = lambda s, n: [s[n] - s[f'{n}_lo'], s[f'{n}_hi'] - s[n]]  # noqa: E731
fx = lambda s: [s.feh_med - s.feh_p16, s.feh_p84 - s.feh_med]  # noqa: E731  [Fe/H] spread of the sample
# Plot styles: black for all giants, blue stars for the RR Lyrae.
kw = dict(color='k', ms=4, lw=0.8)
rkw = dict(fmt='*', color='C0', ms=7, lw=0.8)
# (a) v0 [km/s]: all giants, APOGEE alone (red squares), RR Lyrae stage 1, and the RR Lyrae joint-fit v0 (open star, shifted 0.04
# dex).
ax[0, 0].errorbar(g.feh_med, g.v0, xerr=fx(g), yerr=err(g, 'v0'), fmt='o', label='giants (APOGEE + BOSS)', **kw)
ax[0, 0].errorbar(g.feh_med_apogee, g.v0_apogee, yerr=g.v0_apogee_err, fmt='s', color='C3', mfc='none', ms=4, lw=0.8,
                  label='giants, APOGEE alone')
ax[0, 0].errorbar(rl.feh_med, rl.v0, xerr=fx(rl), yerr=err(rl, 'v0'), label='RR Lyrae', **rkw)
ax[0, 0].errorbar(rl.feh_med + 0.04, rr2['median'], yerr=[[rr2['median'] - rr2.p16], [rr2.p84 - rr2['median']]], mfc='none',
                  label='RR Lyrae, joint geometry', **rkw)
# (b) sigma_los (filled) and the 4-9 deg proper-motion sigma (open) [km/s].
ax[0, 1].errorbar(g.feh_med, g.sigma_los, xerr=fx(g), yerr=err(g, 'sigma_los'), fmt='o', **kw)
ax[0, 1].errorbar(g.feh_med, g.sigma, yerr=err(g, 'sigma'), fmt='o', mfc='none', **kw)
ax[0, 1].errorbar(g.feh_med_apogee, g.sigma_los_apogee, yerr=g.sigma_los_apogee_err, fmt='s', color='C3', mfc='none', ms=4, lw=0.8)
ax[0, 1].errorbar(rl.feh_med, rl.sigma, xerr=fx(rl), yerr=err(rl, 'sigma'), **rkw)
# (c) v0 / sigma_los (filled) and v0 / sigma (open), from the posterior draws.
ax[0, 2].errorbar(g.feh_med, g.ratio, xerr=fx(g), yerr=rerr(g, 'ratio'), fmt='o', **kw)
ax[0, 2].errorbar(g.feh_med, g.ratio_pm, yerr=rerr(g, 'ratio_pm'), fmt='o', mfc='none', **kw)
ax[0, 2].errorbar(g.feh_med_apogee, g.ratio_apogee, yerr=g.ratio_apogee_err, fmt='s', color='C3', mfc='none', ms=4, lw=0.8)
ax[0, 2].errorbar(rl.feh_med, rl.ratio_pm, xerr=fx(rl), yerr=rerr(rl, 'ratio_pm'), **rkw)
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=4, frameon=False, fontsize=5)  # above the panel
for a, lab in zip(ax[0], (r'$v_0$ [km s$^{-1}$]', r'$\sigma_{\rm los}$ (filled), $\sigma_{\rm PM}$ (open, star) [km s$^{-1}$]',
                          r'$v_0 / \sigma_{\rm los}$ (filled), $v_0 / \sigma_{\rm PM}$ (open, star)')):
    a.set(xlabel='[Fe/H]', ylabel=lab)
ax[0, 2].set_ylim(bottom=0)  # rotational support from zero
for a, letter, (lx, ly) in zip(ax[0], 'abc', ((0.9, 0.05), (0.9, 0.92), (0.04, 0.92))):  # panel letters in empty corners
    a.text(lx, ly, f'({letter})', transform=a.transAxes, fontsize=7)
fig.savefig(ROOT / 'figures/paper_kin_support.png', dpi=300)
