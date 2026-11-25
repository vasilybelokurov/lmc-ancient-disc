"""Is the fitted m = 2 'streaming' a proper-motion / line-of-sight scale mismatch, or a difference between the APOGEE and BOSS
velocities?

The streaming fitted to the giants (kinfit.STREAM, proper motions + line-of-sight velocities) is locked to the line of nodes:
v_phi is lowest along the nodes, where line-of-sight velocities measure the rotation, and highest along the minor axis, where
proper motions measure it. A line-of-sight velocity field whose internal part is a constant factor (1 + qlos) times the one the
proper motions imply (kinfit 'qlos'; e.g. a distance-scale error, since proper motions give velocities proportional to the
adopted D0) would look like that. And the two spectroscopic surveys cover different fields, so a velocity offset between them
could fake structure.

Fits (lmcold.kinfit.fit, giants at 2 < rho < 9 deg as in scripts/fit_kinematics_mcmc.py; centre, bulk proper motion, i, theta,
rotation, dispersions, vlos0, sigma_los free) of each [Fe/H] bin with the velocities of both surveys ('all'), APOGEE only and BOSS
only (the other survey's velocities dropped, all proper motions kept), with four models: 'sym' (one plane, no streaming; centre
fixed at the adopted position first, then free), 'stream' (+ STREAM), 'qlos' (+ qlos) and 'qlos + stream' (+ both), each started
from 'sym'. Mocks at the stars of the giants -0.8 < [Fe/H] < -0.5 (both surveys), drawn (k = 1, no outliers, Gaia and velocity
errors) from that bin's 'qlos' fit ('mock qlos': does a scale mismatch make node-locked streaming?) and 'stream' fit ('mock
stream': does streaming make a spurious qlos?). Data sets with fewer than MIN_LOS velocities are skipped.

Writes results/kin_los_scale.csv (sample, velocities, model, n_los, nll, every parameter of kinfit.PARAMS and its error *_err;
truth_* for the mocks) and figures/fig_kin_los_scale.png: four equal panels per data set (x axis): (a) -dlnL over 'sym' of the
other three models; (b) |m2| = hypot(m2c, m2s) of 'stream' and 'qlos + stream'; (c) qlos of 'qlos' and 'qlos + stream';
(d) centre offset east [deg] of each model. Mock truths as red crosses; legends above the panels. Diagnostic figure (not in the
paper). The 'qlos + stream' rows (velocities 'all') of the five giant bins are read by scripts/paper_kin_sensitivity.py (paper
appendix figure paper_kin_sensitivity) and give the numbers of alternative model ii on the KB page 'Rotational support, centre and
sensitivity' (scripts/paper_text_kin_support.py).

Inputs: lmcold.samples (release_giants_apogee and release_giants_boss via samples.giants(), Zenodo bundle 1).

Run: .venv/bin/python scripts/fig_kin_los_scale.py   (run time not recorded)
"""
# %% [markdown]
# ### Settings
# The catalogue columns the fits read, the parameters free in every model, the four models, the giant bin from which the mocks are
# drawn and the fewest velocities a data set needs.

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
# error [km/s].
COLS = ('ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'vlos', 'evlos')
FREE = kinfit.BASE + ('vlos0', 'sigma_los')  # free in 'sym': proper-motion model plus systemic velocity and its dispersion
# model name: parameters freed in addition to FREE (STREAM: m2c, m2s, vr0, r2c, r2s; qlos: the line-of-sight deviations from the
# mean motion scaled by 1 + qlos).
MODELS = {'sym': (), 'stream': kinfit.STREAM, 'qlos': ('qlos',), 'qlos + stream': ('qlos',) + kinfit.STREAM}
MOCK_BIN = 3  # giants -0.8 < [Fe/H] < -0.5: the most precise bin
MIN_LOS = 300  # fewest line-of-sight velocities for a data set to be fitted


# %% [markdown]
# ### Helpers
# `fits` fits the four models to one data set, `mock_of` draws a mock catalogue from a fitted parameter vector at the positions of
# the real stars, and `add` stores the fitted rows.

# %%
def fits(c):
    """Fit the four models of MODELS to one data set (proper motions and line-of-sight velocities).

    c : dict of ndarray, the catalogue (columns COLS).
    Returns a list of (model, nll, p, err): -lnL at the optimum, and the parameter vector and its inverse-Hessian errors (each of
    length len(kinfit.PARAMS); angles in deg, velocities in km/s). Prints one line per model.
    """
    # the arrays kinfit needs (unit vectors, Gaia covariances, velocities)
    d = kinfit.data(*(c[k] for k in COLS))
    p, _, _ = kinfit.fit(d, free=FREE[2:])  # centre fixed at the adopted position first
    # 'sym': all of FREE, started from the fixed-centre fit; sym = (p, err, nll)
    sym = kinfit.fit(d, free=FREE, p0=p)
    out = [('sym', sym[2], sym[0], sym[1])]
    # the other three models, each started from 'sym'
    for name, extra in list(MODELS.items())[1:]:
        p, e, nll = kinfit.fit(d, free=FREE + extra, p0=sym[0])
        out.append((name, nll, p, e))
    for name, nll, q, _ in out:  # q indices: kinfit.PARAMS (0 dx, 1 dy, 4 inc, 5 theta, 6 v0, 17 m2c, 18 m2s, 24 qlos)
        print(f'  {name:14s} -dlnL {sym[2] - nll:8.1f}  dx {q[0]:.3f} dy {q[1]:.3f} i {q[4]:.1f} theta {q[5]:.1f} '
              f'v0 {q[6]:.1f} qlos {q[24]:+.3f} m2c {q[17]:+.3f} m2s {q[18]:+.3f}', flush=True)
    return out


def mock_of(c, q):
    """Mock of catalogue c: proper motions and velocities drawn from parameter vector q, with k = 1 and no outliers.

    c : dict of ndarray (columns COLS); q : ndarray, kinfit.PARAMS vector (the fitted model to draw from).
    Returns (mock catalogue dict, the parameter vector actually used: q with k = 1 and logit_fout = -8, the mock's truth).
    """
    q = q.copy()
    q[kinfit.PARAMS.index('k')], q[kinfit.PARAMS.index('logit_fout')] = 1.0, -8.0  # Gaia errors as quoted; outlier fraction ~3e-4
    # the catalogue's arrays, in the form kinfit.model needs
    d = kinfit.data(*(c[k] for k in COLS))
    # Model mean proper motions [mas/yr], line-of-sight velocity [km/s] and disc distance D [kpc] of each star.
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(q, d['rhat'], d['ea'], d['ed']))
    # fixed seed: the mocks are reproducible
    rng = np.random.default_rng(6)
    # Proper motions: Gaia covariance plus sigma (q[8], km/s) as a proper motion at D; velocities: e_vlos and sigma_los (q[12]).
    cov = np.asarray(d['cov']) + ((q[8] / (KAPPA * D)) ** 2)[:, None, None] * np.eye(2)
    noise = np.einsum('nij,nj->ni', np.linalg.cholesky(cov), rng.normal(size=(len(D), 2)))
    # velocities only where the star has a measured one; e_vlos and sigma_los added in quadrature
    vlos = np.where(np.isfinite(c['vlos']), mlos + rng.normal(size=len(D)) * np.hypot(c['evlos'], q[12]), np.nan)
    return dict(c, pmra=mra + noise[:, 0], pmdec=mde + noise[:, 1], vlos=vlos), q


rows, fitted = [], {}  # output rows; fitted[sample, velocities][model] = parameter vector (the mocks are drawn from these)


def add(name, vel, c, out, truth=None):
    """Append the rows of fits() output `out` for sample `name`, velocities `vel` (truth: the mock's parameter vector)."""
    # keep the fitted vectors: the mocks are drawn from the 'qlos' and 'stream' fits of MOCK_BIN
    fitted[name, vel] = {m: q for m, _, q, _ in out}
    for model, nll, p, e in out:
        # one row per model: counts, -lnL, every parameter and its error (and the truth for mocks)
        rows.append(dict(sample=name, velocities=vel, model=model, n_los=int(np.isfinite(c['vlos']).sum()), nll=nll,
                         **dict(zip(kinfit.PARAMS, p)), **{f'{n}_err': v for n, v in zip(kinfit.PARAMS, e)},
                         **({} if truth is None else {f'truth_{n}': v for n, v in zip(kinfit.PARAMS, truth)})))


# %% [markdown]
# ### Load the data
# The giants of the five [Fe/H] bins (`lmcold.samples.giants()`, APOGEE and BOSS merged; column `survey` names the source of each
# spectrum), each cut to 2 < $\rho$ < 9 deg from the adopted centre (`ring`).

# %%
# (label, catalogue dict) per giant bin, cut to 2 < rho < 9 deg
giants = [(f'giants {lab}', ring(c)) for lab, c in samples.giants()]

# %% [markdown]
# ### Fits of the data and the mocks
# Each giant bin is fitted three times: with the velocities of both surveys, of APOGEE only and of BOSS only (the other survey's
# velocities set to NaN, all proper motions kept). Then two mocks at the stars of the giants -0.8 < [Fe/H] < -0.5, drawn from that
# bin's 'qlos' and 'stream' fits (both surveys), each fitted with the same four models.

# %%
# each giant bin, with both surveys' velocities, then APOGEE only and BOSS only
for name, c in giants:
    for vel in ('all', 'apogee', 'boss'):
        # 'apogee'/'boss': velocities of the other survey set to NaN, proper motions all kept
        cv = c if vel == 'all' else dict(c, vlos=np.where(c['survey'] == vel, c['vlos'], np.nan))
        # number of finite line-of-sight velocities in this data set
        n_los = int(np.isfinite(cv['vlos']).sum())
        print(name, vel, n_los, flush=True)
        # fit only data sets with at least MIN_LOS velocities
        if n_los >= MIN_LOS:
            add(name, vel, cv, fits(cv))
# the bin the mocks are drawn from (MOCK_BIN = 3, -0.8 < [Fe/H] < -0.5)
base_name, base = giants[MOCK_BIN]
for mname, model in (('mock qlos', 'qlos'), ('mock stream', 'stream')):
    print(mname, flush=True)
    # mock drawn from that bin's fitted 'qlos' or 'stream' parameter vector (both surveys), then fitted with all four models
    mock, truth = mock_of(base, fitted[base_name, 'all'][model])
    add(mname, 'all', mock, fits(mock), truth)
# one row per (sample, velocities, model)
t = pd.DataFrame(rows)
t.to_csv(ROOT / 'results/kin_los_scale.csv', index=False, float_format='%.5f')

# %% [markdown]
# ### The figure
# Four panels with one column of points per data set: likelihood gain over 'sym', streaming amplitude, $q_{\rm los}$ and centre
# offset east; the mock truths as red crosses.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
# one group of four rows (one per model) per data set, in the order fitted
groups = list(t.groupby(['sample', 'velocities'], sort=False))
# x tick labels: 'sample (velocities)'
labels = [f'{s} ({v})' for (s, v), _ in groups]
# one row of four equal panels; bottom=1.4 leaves room for the rotated tick labels
fig, ax = panel_grid(1, 4, width=9.0, left=0.5, right=0.2, hgap=0.55, top=0.45, bottom=1.4)
mk = dict(zip(MODELS, 'o^sD'))  # marker per model; points of one data set offset by 0.15 in x per model
# k: data set (x position); j: model
for k, (_, s) in enumerate(groups):
    s = s.set_index('model')
    for j, m in enumerate(MODELS):
        # r: the model's row; x: data-set position plus offset; lab: legend label only for the first data set
        r, x, lab = s.loc[m], k + 0.15 * (j - 1.5), m if k == 0 else None
        # (a) likelihood gain over 'sym' (positive = better fit)
        if m != 'sym':
            ax[0, 0].plot(x, s.loc['sym', 'nll'] - r.nll, mk[m], color='k', ms=3, label=lab)
        # (b) streaming amplitude |m2| for the models with streaming
        if 'stream' in m:
            ax[0, 1].plot(x, np.hypot(r.m2c, r.m2s), mk[m], color='k', ms=3, label=lab)
        # (c) qlos with its error for the models with qlos
        if 'qlos' in m:
            ax[0, 2].errorbar(x, r.qlos, yerr=r.qlos_err, fmt=mk[m], color='k', ms=3, lw=0.6, label=lab)
        # (d) centre offset east dx [deg]; 'sym' as an open marker
        ax[0, 3].plot(x, r.dx, mk[m], color='k', ms=3, mfc='none' if m == 'sym' else None, label=lab)
    # mock data sets: the truth values (red crosses)
    if 'truth_qlos' in s and np.isfinite(s.truth_qlos.iloc[0]):
        q = s.iloc[0]
        ax[0, 1].plot(k, np.hypot(q.truth_m2c, q.truth_m2s), 'rx', ms=5)
        ax[0, 2].plot(k, q.truth_qlos, 'rx', ms=5)
        ax[0, 3].plot(k, q.truth_dx, 'rx', ms=5)
# axis labels, x tick labels and legends of the four panels
for a, lab in zip(ax[0], (r'$-\Delta\ln L$ over sym', '|m2| [fraction of vrot]', 'qlos', 'centre offset east [deg]')):
    a.set(ylabel=lab, xticks=range(len(labels)), xlim=(-0.5, len(labels) - 0.5))
    a.set_xticklabels(labels, rotation=70, ha='right', fontsize=4)
    a.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=4, frameon=False, ncol=2)  # above the panel
# symmetric log scale for the likelihood gains of panel (a)
ax[0, 0].set_yscale('symlog')
# zero lines in (b) and (c)
for a in ax[0, 1:3]:
    a.axhline(0, color='0.6', lw=0.5)
fig.savefig(ROOT / 'figures/fig_kin_los_scale.png', dpi=200)
