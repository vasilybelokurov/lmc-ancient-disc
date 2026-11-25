"""Is the fitted m = 2 'streaming' a tilted inner disc? Fits with a tilted ring (kinfit.TILT: the stars within kinfit.RHO_TILT
= 4.5 deg of the adopted centre in their own plane, inclination inc + dinc and nodes theta + dtheta, through the same centre)
instead of, and together with, in-plane streaming (kinfit.STREAM).

The bar region of the LMC is tilted relative to the outer disc (red clump distances, Choi et al. 2018, ApJ 866, 90; vertical
velocities, Jimenez-Arranz et al. 2025, A&A 698, A88), and the streaming fitted to our data is locked to the line of nodes (v_phi
minimum within 4 deg of the nodes in the metal-rich bins), not to the bar (24 deg from the nodes in the plane): the signature of a
plane that changes with radius fitted with one plane.

Fits (lmcold.kinfit.fit, stars at 2 < rho < 9 deg as in scripts/fit_kinematics_mcmc.py, centre, bulk proper motion, i, theta,
rotation, dispersions free), each sample with four models: 'sym' (one plane, no streaming; centre fixed at the adopted position
first, then free), 'stream' (+ STREAM), 'tilt' (+ TILT) and 'tilt + stream' (+ both), each started from 'sym'; 'tilt' fits start
from dinc = 0 and +-10 deg and keep the best. Samples:
- giants (each [Fe/H] bin), proper motions and line-of-sight velocities;
- RR Lyrae, proper motions ('RR Lyrae PM') and proper motions plus distance moduli ('RR Lyrae PM + distances');
- 'mock tilt': drawn (k = 1, no outliers, Gaia and velocity errors) at the stars of the giants -0.8 < [Fe/H] < -0.5 from that
  bin's 'tilt' fit (a tilted ring, no streaming): does fitting 'stream' to it produce node-locked m = 2 and move the centre?
- 'mock stream': the same from that bin's 'stream' fit (streaming, one plane): does 'tilt' appear when there is none?

Writes results/kin_tilted_ring.csv (sample, model, nll, every parameter of kinfit.PARAMS and its error *_err; truth_* for the
mocks) and figures/fig_kin_tilted_ring.png: four equal panels per sample (x axis): (a) likelihood gain -dlnL over 'sym' of the
other three models; (b) centre offset east and north [deg] of each model; (c) streaming amplitude |m2| = hypot(m2c, m2s) of
'stream' and 'tilt + stream'; (d) inner-ring tilt dinc and dtheta [deg] of 'tilt' and 'tilt + stream'. Mock truths as red crosses; legends above
panels (a), (c), (d) and below (b); fits with a centre outside panel (b) are listed in it.
Diagnostic figure (not in the paper). The 'sym' (Level 1), 'stream' (Level 2) and 'tilt + stream' rows of the five giant bins are
read by scripts/paper_kin_sensitivity.py (paper appendix figure paper_kin_sensitivity); the table also gives numbers of the KB page
'Rotational support, centre and sensitivity' (centre and geometry, alternative model i) through scripts/paper_text_kin_support.py.

Inputs: lmcold.samples (data/rrab.parquet, key release_rrab; release_giants_apogee and release_giants_boss via samples.giants()),
all in Zenodo bundle 1.

Run: .venv/bin/python scripts/fig_kin_tilted_ring.py   (run time not recorded)
"""
# %% [markdown]
# ### Settings
# The catalogue columns the fits read, the four models (sets of parameters freed on top of the 'sym' ones), the giant bin from
# which the mocks are drawn and the starting tilts of the inner ring.

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_kinematics_mcmc import ring
from lmcold import kinfit, samples
from lmcold.constants import KAPPA
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

# Columns passed to kinfit.data: position [deg], proper motion and errors [mas/yr], their correlation, line-of-sight velocity and
# error [km/s], distance modulus [mag] (NaN where a sample has none).
COLS = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'vlos', 'evlos', 'dmu')
# model name: parameters freed in addition to the 'sym' ones (STREAM: m2c, m2s, vr0, r2c, r2s; TILT: dinc, dtheta [deg]).
MODELS = {'sym': (), 'stream': kinfit.STREAM, 'tilt': kinfit.TILT, 'tilt + stream': kinfit.TILT + kinfit.STREAM}
MOCK_BIN = 3  # giants -0.8 < [Fe/H] < -0.5: the most precise bin
DINC_STARTS = (0.0, 10.0, -10.0)  # deg, starting inner-ring tilts of the 'tilt' fits


# %% [markdown]
# ### Helpers
# `fits` fits the four models to one sample, `mock_of` draws a mock catalogue from a fitted parameter vector at the positions of
# the real stars, and `add` stores the fitted rows.

# %%
def fits(c, free):
    """Fit the four models of MODELS to one sample.

    c : dict of ndarray, the catalogue (columns COLS); free : tuple of str, the parameters free in 'sym' (kinfit.PARAMS names).
    Returns a list of (model, nll, p, err): -lnL at the optimum, and the parameter vector and its inverse-Hessian errors (each of
    length len(kinfit.PARAMS); angles in deg, velocities in km/s). Prints one line per model.
    """
    # Likelihood inputs of the sample (kinfit.data arrays)
    d = kinfit.data(*(c[k] for k in COLS))
    p, _, _ = kinfit.fit(d, free=free[2:])  # centre fixed at the adopted position first
    # Then the centre (dx, dy, the first two of free) freed too; kinfit.fit returns (p, err, nll)
    sym = kinfit.fit(d, free=free, p0=p)
    out = [('sym', sym[2], sym[0], sym[1])]
    # The other three models, each started from 'sym'; 'tilt' models tried from every DINC_STARTS, best -lnL kept
    for name, extra in list(MODELS.items())[1:]:
        best = None
        for dinc in (DINC_STARTS if 'tilt' in name else (0.0,)):
            p0 = sym[0].copy()
            p0[kinfit.PARAMS.index('dinc')] = dinc
            r = kinfit.fit(d, free=free + extra, p0=p0)
            best = r if best is None or r[2] < best[2] else best
        out.append((name, best[2], best[0], best[1]))
    for name, nll, q, _ in out:  # q indices: kinfit.PARAMS (0 dx, 1 dy, 4 inc, 5 theta, 17 m2c, 18 m2s, 22 dinc, 23 dtheta)
        print(f'  {name:14s} -dlnL {sym[2] - nll:8.1f}  dx {q[0]:.3f} dy {q[1]:.3f} i {q[4]:.1f} theta {q[5]:.1f} '
              f'dinc {q[22]:.1f} dtheta {q[23]:.1f} |m2| {np.hypot(q[17], q[18]):.3f}', flush=True)
    return out


def mock_of(c, q):
    """Mock of catalogue c: proper motions and velocities drawn from parameter vector q, with k = 1 and no outliers.

    c : dict of ndarray (columns COLS); q : ndarray, kinfit.PARAMS vector (the fitted model to draw from).
    Returns (mock catalogue dict, the parameter vector actually used: q with k = 1 and logit_fout = -8, the mock's truth).
    """
    q = q.copy()
    q[kinfit.PARAMS.index('k')], q[kinfit.PARAMS.index('logit_fout')] = 1.0, -8.0  # Gaia errors as quoted; outlier fraction ~3e-4
    # Likelihood inputs at the real stars (positions, errors)
    d = kinfit.data(*(c[k] for k in COLS))
    # Model mean proper motions [mas/yr], line-of-sight velocity [km/s] and disc distance D [kpc] of each star.
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(q, d['rhat'], d['ea'], d['ed']))
    # Fixed seed: the same mock on every run
    rng = np.random.default_rng(5)
    # Proper motions: Gaia covariance plus sigma (q[8], km/s) as a proper motion at D; velocities: e_vlos and sigma_los (q[12]).
    cov = np.asarray(d['cov']) + ((q[8] / (KAPPA * D)) ** 2)[:, None, None] * np.eye(2)
    # Correlated proper-motion noise (N, 2) [mas/yr] via the Cholesky factor of each star's covariance
    noise = np.einsum('nij,nj->ni', np.linalg.cholesky(cov), rng.normal(size=(len(D), 2)))
    # Velocities [km/s] drawn only where the real star has one
    vlos = np.where(np.isfinite(c['vlos']), mlos + rng.normal(size=len(D)) * np.hypot(c['evlos'], q[12]), np.nan)
    return dict(c, pmra=mra + noise[:, 0], pmdec=mde + noise[:, 1], vlos=vlos), q


# %% [markdown]
# ### Load the data
# The giants of the five [Fe/H] bins (`lmcold.samples.giants()`, APOGEE and BOSS merged) and the RR Lyrae members
# (`lmcold.samples.rrab()`), each cut to 2 < $\rho$ < 9 deg from the adopted centre (`ring`). The giants have no distance moduli
# and the RR Lyrae no line-of-sight velocities: those columns are filled with NaN, which `kinfit.data` reads as missing.

# %%
# Free parameters of 'sym': with line-of-sight velocities (systemic vlos0 and dispersion sigma_los added) or with RR Lyrae
# distance moduli (zero point dmu0 and scatter sigma_mu added) on top of the proper-motion model kinfit.BASE.
LOS, DIST = kinfit.BASE + ('vlos0', 'sigma_los'), kinfit.BASE + ('dmu0', 'sigma_mu')
nan = lambda c: np.full(len(c['ra']), np.nan)  # noqa: E731
# Giant bins with NaN distance moduli; RR Lyrae with NaN velocities; both cut to 2 < rho < 9 deg by ring()
giants = [(f'giants {lab}', ring(c | dict(dmu=nan(c)))) for lab, c in samples.giants()]
rr = samples.rrab()
rr = ring(rr | dict(vlos=nan(rr), evlos=nan(rr)))
# (sample name, catalogue, 'sym' free parameters) of each data fit
runs = [(name, c, LOS) for name, c in giants] + [('RR Lyrae PM', rr | dict(dmu=nan(rr)), kinfit.BASE),
                                                  ('RR Lyrae PM + distances', rr, DIST)]
rows, fitted = [], {}  # output rows; fitted[sample][model] = parameter vector (the mocks are drawn from these)


def add(name, out, truth=None):
    """Append the rows of fits() output `out` for sample `name` (truth: the mock's parameter vector)."""
    # Keep each model's parameter vector; then one row per model with parameters, errors and (mocks) truths
    fitted[name] = {m: q for m, _, q, _ in out}
    for model, nll, p, e in out:
        rows.append(dict(sample=name, model=model, nll=nll, **dict(zip(kinfit.PARAMS, p)),
                         **{f'{n}_err': v for n, v in zip(kinfit.PARAMS, e)},
                         **({} if truth is None else {f'truth_{n}': v for n, v in zip(kinfit.PARAMS, truth)})))


# %% [markdown]
# ### Fits of the data and the mocks
# The four models fitted to each giant bin and to the RR Lyrae, then two mocks drawn at the stars of the giants
# -0.8 < [Fe/H] < -0.5: one from that bin's 'tilt' fit and one from its 'stream' fit, each fitted with the same four models.

# %%
# Fit the four models to every data sample
for name, c, free in runs:
    print(name, flush=True)
    add(name, fits(c, free))
# Mocks drawn at the stars of the MOCK_BIN giants from that bin's 'tilt' and 'stream' fits, then fitted alike
base_name, base = giants[MOCK_BIN]
for mname, model in (('mock tilt', 'tilt'), ('mock stream', 'stream')):
    print(mname, flush=True)
    mock, truth = mock_of(base, fitted[base_name][model])
    add(mname, fits(mock, LOS), truth)
# All rows (samples x models) to results/kin_tilted_ring.csv
t = pd.DataFrame(rows)
t.to_csv(ROOT / 'results/kin_tilted_ring.csv', index=False, float_format='%.5f')

# %% [markdown]
# ### The figure
# Four panels with one column of points per sample: likelihood gain over 'sym', centre offset, streaming amplitude and inner-ring
# tilt; the mock truths as red crosses.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# Samples in fitting order (x axis); four equal panels
names = list(t['sample'].unique())
fig, ax = panel_grid(1, 4, width=9.0, left=0.5, right=0.2, hgap=0.55, top=0.45, bottom=1.25)
mk = dict(zip(MODELS, 'o^sD'))  # marker per model; points of one sample offset by 0.15 in x per model
for k, name in enumerate(names):
    # Rows of one sample indexed by model; x: sample index offset by model
    s = t[t['sample'] == name].set_index('model')
    for j, m in enumerate(MODELS):
        r, x = s.loc[m], k + 0.15 * (j - 1.5)
        lab = m if k == 0 else None
        # (a) likelihood gain over 'sym'; (b) fitted centre offset [deg], open for 'sym', colour by sample
        if m != 'sym':
            ax[0, 0].plot(x, s.loc['sym', 'nll'] - r.nll, mk[m], color='k', ms=3, label=lab)
        ax[0, 1].plot(r.dx, r.dy, mk[m], color=plt.cm.viridis(k / (len(names) - 1)), ms=3, mfc='none' if m == 'sym' else None)
        # (c) streaming amplitude |m2| of the streaming models; (d) inner-ring tilts dinc (black), dtheta (red) [deg]
        if 'stream' in m:
            ax[0, 2].plot(x, np.hypot(r.m2c, r.m2s), mk[m], color='k', ms=3, label=lab)
        if 'tilt' in m:
            ax[0, 3].plot(x, r.dinc, mk[m], color='k', ms=3, label=f'{m}: dinc' if k == 0 else None)
            ax[0, 3].plot(x, r.dtheta, mk[m], color='tab:red', ms=3, label=f'{m}: dtheta' if k == 0 else None)
    # Mock truths (the same in every row of the mock) as red crosses
    if name.startswith('mock'):
        q = s.iloc[0]
        ax[0, 1].plot(q.truth_dx, q.truth_dy, 'rx', ms=5)
        ax[0, 2].plot(k, np.hypot(q.truth_m2c, q.truth_m2s), 'rx', ms=5)
        ax[0, 3].plot([k, k], [q.truth_dinc, q.truth_dtheta], 'rx', ms=5)
    # Legend entry for the sample's colour in panel (b)
    ax[0, 1].plot([], [], 's', color=plt.cm.viridis(k / (len(names) - 1)), ms=3, label=name)
# Sample names as x tick labels on panels (a), (c), (d)
for a in (ax[0, 0], ax[0, 2], ax[0, 3]):
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=60, ha='right', fontsize=5)
    a.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=4, frameon=False, ncol=2)  # above the panel
# Axis labels and limits; panel (b): east to the left, the adopted centre as a black cross
ax[0, 0].set(ylabel=r'$-\Delta\ln L$ over sym', yscale='symlog')
ax[0, 1].set(xlim=(1.3, -0.3), ylim=(-0.8, 0.8), xlabel='offset east [deg]', ylabel='offset north [deg]')
ax[0, 1].plot(0, 0, 'k+', ms=8)
ax[0, 1].legend(loc='upper left', bbox_to_anchor=(0, -0.2), fontsize=4, frameon=False, ncol=2)  # below the axis label
# Fits whose centre falls outside the panel are listed on it.
out = t[(t.dx.abs() > 1.3) | (t.dy.abs() > 0.8)]
ax[0, 1].text(0.02, 0.02, '\n'.join(f'off panel: {r.sample}, {r.model} ({r.dx:.2f}, {r.dy:.2f})' for r in out.itertuples()),
              transform=ax[0, 1].transAxes, fontsize=4, va='bottom')
ax[0, 2].set(ylabel='|m2| [fraction of vrot]')
ax[0, 3].set(ylabel='inner-ring tilt [deg]')
for a in (ax[0, 2], ax[0, 3]):
    a.axhline(0, color='0.6', lw=0.5)
fig.savefig(ROOT / 'figures/fig_kin_tilted_ring.png', dpi=200)
