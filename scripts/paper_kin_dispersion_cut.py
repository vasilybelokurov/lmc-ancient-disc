"""Does the proper-motion dispersion depend on where the inner edge of the kinematic sample is cut? Diagnostic for the RR Lyrae, whose
dispersion fell to its 5 km/s lower bound in the 2 < rho < 9 deg fits (scripts/fit_kinematics_mcmc.py).

With one Gaia error scale k per population, the likelihood mostly constrains sigma^2 + k^2 e^2 per star. The RR Lyrae errors are
e ~ 64 km/s per component (giants 12-15), so a few per cent in k is the whole of sigma. If Gaia underestimates its errors more in the
crowded inner disc, the inner stars set k and the outer stars' errors are overstated (earlier fits
found k ~ 2 in the bar, ~1 beyond 3.5 kpc; outer-disc sigma 34-55 km/s for k = 1.2-0.9).

Data: every population of scripts/fit_kinematics_mcmc.py (giants in five [Fe/H] bins, PM + LOS; RR Lyrae, PM only) at inner < rho < 9 deg
for each inner edge in INNER, maximum likelihood of the Level-2 model with everything free (fit_kinematics_mcmc.ml, model 'stream');
errors of sigma and k from the inverse Hessian with the geometry held (kinfit.fit).
Mock: the RR Lyrae positions and Gaia covariances at 2 < rho < 9 deg, velocities drawn from their own fit beyond 4 deg with
sigma = MOCK_SIGMA and an error scale that depends on the position, MOCK_K[0] inside 4 deg and MOCK_K[1] beyond (the values the data
give), no outliers; fitted like the data. One k per population must then return too small a sigma when the inner stars are included
and the true one when they are not.

Writes results/kin_dispersion_cut.csv (sample 'data' or 'mock', population, inner, n, sigma, sigma_err, k, k_err, v0) and
figures/paper_kin_dispersion_cut.png: 2 x 2 equal panels; x the inner edge [deg]; top sigma from the proper motions [km/s], bottom k;
left the data (one line per population), right the mock (truth dashed). Paper figure (appendix app:kin).

Run: .venv/bin/python scripts/paper_kin_dispersion_cut.py   (about 8 min: 481 s in the run of 2026-11-22, docs/run_times.txt)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_kinematics_mcmc import COLS, ml, ring
from lmcold import kinfit, samples
from lmcold.constants import KAPPA, LMC
from lmcold.footprint import angular_distance
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

INNER = (2., 3., 4., 5., 6.)  # deg, inner edges of the sample
MOCK_SIGMA = 30.0  # km/s   # true proper-motion dispersion of the mock
MOCK_K = (1.39, 1.13)  # Gaia error scale inside and beyond 4 deg (RR Lyrae fits of the data at 2-4 and 4-9 deg)
COLOURS = ['#d7191c', '#fdae61', '#a6d96a', '#1a9641', '#2878b5', '#7b3294']  # giant bins metal-poor first, RR Lyrae


def fits(sample, name, c):
    """Maximum-likelihood rows of catalogue dict c (2 < rho < 9 deg) for every inner edge of INNER."""
    rho = angular_distance(c['ra'], c['dec'], LMC['ra0'], LMC['dec0'])
    # rho: angular distance of each star from the adopted LMC centre [deg]; one row per inner edge.
    out = []
    for inner in INNER:
        d = kinfit.data(*(np.asarray(c[k])[rho > inner] for k in COLS))
        # Full ML fit of the stars beyond `inner`; then errors of v0, sigma, k and the outlier fraction from the inverse Hessian,
        # geometry held.
        p = ml(d, 'stream')
        _, e, _ = kinfit.fit(d, free=('v0', 'sigma', 'k', 'logit_fout'), p0=p)
        g = lambda n, v=p: v[kinfit.PARAMS.index(n)]  # noqa: E731
        out.append(dict(sample=sample, population=name, inner=inner, n=int((rho > inner).sum()), sigma=g('sigma'),
                        sigma_err=g('sigma', e), k=g('k'), k_err=g('k', e), v0=g('v0')))
        print(out[-1], flush=True)
    return out


# %% [markdown]
# ### Load the data
# The giant catalogues of Zenodo bundle 1 through `lmcold.samples.giants()`, merged and split into the five [Fe/H] bins, and the
# RR Lyrae members through `lmcold.samples.rrab()` (from `data/rrab.parquet`).

# %%
giants = samples.giants()  # (label, catalogue dict) per [Fe/H] bin, from data/giants_apogee.parquet and giants_boss.parquet
rr = samples.rrab()  # RR Lyrae members with Gaia astrometry (data/rrab.parquet)

# %% [markdown]
# ### The data
# Every group (giants: proper motions and line-of-sight velocities; RR Lyrae: proper motions only) fitted by maximum likelihood for
# each inner edge of the sample in `INNER`.

# %%
# Missing observables as NaN arrays: no distances for anyone, no line-of-sight velocities for the RR Lyrae; all at 2 < rho < 9 deg.
nan = lambda c: np.full(len(c['ra']), np.nan)  # noqa: E731
pops = [(f'giants {lab}', ring(c | dict(dmu=nan(c)))) for lab, c in giants]
pops.append(('RR Lyrae', ring(rr | dict(vlos=nan(rr), evlos=nan(rr), dmu=nan(rr)))))
# One row per group and inner edge.
rows = [r for name, c in pops for r in fits('data', name, c)]

# %% [markdown]
# ### The mock RR Lyrae
# The RR Lyrae positions and Gaia covariances, velocities from their own fit beyond $4^\circ$ with $\sigma_{\rm PM}$ = `MOCK_SIGMA`,
# and an error scale $k$ = `MOCK_K[0]` inside $4^\circ$ and `MOCK_K[1]` beyond; fitted like the data. The table goes to
# `results/kin_dispersion_cut.csv`.

# %%
# Mock RR Lyrae: truth from their own fit beyond 4 deg, with sigma and the position-dependent k replaced.
c = pops[-1][1]
rho = angular_distance(c['ra'], c['dec'], LMC['ra0'], LMC['dec0'])
# c: the RR Lyrae at 2 < rho < 9 deg; q: their ML vector from the stars beyond 4 deg, with sigma set to MOCK_SIGMA [km/s].
q = ml(kinfit.data(*(np.asarray(c[k])[rho > 4.] for k in COLS)), 'stream')
q[kinfit.PARAMS.index('sigma')] = MOCK_SIGMA
d = kinfit.data(*(c[k] for k in COLS))
# Model proper motions [mas/yr] and distances D [kpc] of each RR Lyrae for vector q.
mra, mde, _, D = (np.asarray(v) for v in kinfit.model(q, d['rhat'], d['ea'], d['ed']))
k = np.where(rho < 4., MOCK_K[0], MOCK_K[1])
# Per-star covariance (N, 2, 2) [(mas/yr)^2]: Gaia covariance scaled by k^2 plus the intrinsic dispersion converted to mas/yr at D.
cov = k[:, None, None] ** 2 * np.asarray(d['cov']) + ((MOCK_SIGMA / (KAPPA * D)) ** 2)[:, None, None] * np.eye(2)
# Correlated Gaussian draws (seed 1) added to the model proper motions; the mock is fitted like the data.
noise = np.einsum('nij,nj->ni', np.linalg.cholesky(cov), np.random.default_rng(1).normal(size=(len(D), 2)))
rows += fits('mock', 'RR Lyrae', c | dict(pmra=mra + noise[:, 0], pmdec=mde + noise[:, 1]))
t = pd.DataFrame(rows)
t.to_csv(ROOT / 'results/kin_dispersion_cut.csv', index=False, float_format='%.4f')

# %% [markdown]
# ### The figure
# Top: $\sigma_{\rm PM}$; bottom: $k$; left: data, one line per group; right: the mock, truth dashed.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# 2 x 2 equal panels: rows sigma and k, columns data and mock; each group's points shifted slightly in x so they do not overlap.
fig, ax = panel_grid(2, 2, width=6.0, left=0.6, right=0.1, hgap=0.5, vgap=0.1, top=0.55, aspect=0.75)
# Column j: data (one colour per group, COLOURS) or the mock (the RR Lyrae colour); rows: sigma, k.
for j, sample in enumerate(('data', 'mock')):
    s = t[t['sample'] == sample]
    for (name, g), colour in zip(s.groupby('population', sort=False), COLOURS if sample == 'data' else COLOURS[-1:]):
        for i, q in enumerate(('sigma', 'k')):
            ax[i, j].errorbar(g.inner + 0.04 * (COLOURS.index(colour) - 2.5), g[q], yerr=g[f'{q}_err'], fmt='o-',
                              color=colour, ms=3, lw=0.8, label=name.replace('giants ', 'giants, '))
    # Column title above the top panel; x label on the bottom row only.
    ax[0, j].text(0.5, 1.04, sample, transform=ax[0, j].transAxes, ha='center')
    ax[1, j].set_xlabel('inner edge of the sample [deg]')
    ax[0, j].tick_params(labelbottom=False)
# Mock truth: sigma and the two values of k, dashed.
ax[0, 1].axhline(MOCK_SIGMA, color='k', ls='--', lw=0.6)
for kk in MOCK_K:
    ax[1, 1].axhline(kk, color='k', ls='--', lw=0.6)
# Axis labels and ranges; the legend above the top-left panel, outside the data.
ax[0, 0].set_ylabel(r'$\sigma_{\rm PM}$ [km s$^{-1}$]')
ax[1, 0].set_ylabel('$k$')
for a in ax[0]:
    a.set_ylim(0, 60)
for a in ax[1]:
    a.set_ylim(0.9, 1.7)
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.12), ncol=3, fontsize=5, frameon=False)
fig.savefig(ROOT / 'figures/paper_kin_dispersion_cut.png')
