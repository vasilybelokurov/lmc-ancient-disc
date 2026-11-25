"""Is the fitted m = 2 'streaming' (or the proper-motion / line-of-sight scale term qlos) a changing disc orientation?

A disc whose plane turns (precession and nutation from the Milky Way and SMC torques; van der Marel et al. 2002, AJ 124, 2639,
2002AJ....124.2639V, fitted di/dt = -103 +- 61 deg/Gyr to carbon-star line-of-sight velocities) gives every star the velocity of its
point of the plane: out-of-plane motions that grow linearly with distance from the line of nodes (di/dt) or a solid-body turn
about the line of sight (dtheta/dt). kinfit TURN (didt, dthdt [deg/Gyr]) adds this to the model, equally in the proper motions
and the line-of-sight velocities.

Fits (lmcold.kinfit.fit, giants at 2 < rho < 9 deg as in scripts/fit_kinematics_mcmc.py, proper motions and line-of-sight
velocities of both surveys; centre, bulk proper motion, i, theta, rotation, dispersions, vlos0, sigma_los free) of each [Fe/H]
bin with six models, each started from 'sym' (one plane, no streaming; centre fixed at the adopted position first, then free):
'stream' (+ STREAM), 'turn' (+ TURN), 'turn + stream', 'qlos + stream' and 'turn + qlos + stream'. Two mocks at the stars of the
giants -0.8 < [Fe/H] < -0.5, drawn (k = 1, no outliers, Gaia and velocity errors) from that bin's 'stream' fit ('mock stream':
does streaming make a spurious turn?) and 'turn' fit ('mock turn': does a turn make spurious streaming?).

Writes results/kin_turning.csv (sample, model, n_los, nll, every parameter of kinfit.PARAMS and its error *_err; truth_* for the
mocks) and figures/fig_kin_turning.png: four equal panels per data set (x axis): (a) -dlnL over 'sym'; (b) di/dt and (c)
dtheta/dt [deg/Gyr] of the models with TURN; (d) |m2| = hypot(m2c, m2s) of the models with STREAM. Mock truths as red crosses;
legends above the panels. Diagnostic figure (not in the paper). The 'turn' and 'turn + stream' rows of the five giant bins are read
by scripts/paper_kin_sensitivity.py (paper appendix figure paper_kin_sensitivity) and give the numbers of alternative model iii on
the KB page 'Rotational support, centre and sensitivity' (scripts/paper_text_kin_support.py).

Inputs: lmcold.samples (release_giants_apogee and release_giants_boss via samples.giants(), Zenodo bundle 1).

Run: .venv/bin/python scripts/fig_kin_turning.py   (run time not recorded)
"""
# %% [markdown]
# ### Settings
# The catalogue columns the fits read, the parameters free in every model, the six models and the giant bin from which the
# mocks are drawn.

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
# model name: parameters freed in addition to FREE (STREAM: m2c, m2s, vr0, r2c, r2s; TURN: didt, dthdt [deg/Gyr]; qlos: the
# line-of-sight deviations from the mean motion scaled by 1 + qlos).
MODELS = {'sym': (), 'stream': kinfit.STREAM, 'turn': kinfit.TURN, 'turn + stream': kinfit.TURN + kinfit.STREAM,
          'qlos + stream': ('qlos',) + kinfit.STREAM, 'turn + qlos + stream': kinfit.TURN + ('qlos',) + kinfit.STREAM}
MOCK_BIN = 3  # giants -0.8 < [Fe/H] < -0.5: the most precise bin (as in scripts/fig_kin_los_scale.py)


# %% [markdown]
# ### Helpers
# `fits` fits the six models to one sample, `mock_of` draws a mock catalogue from a fitted parameter vector at the positions of
# the real stars, and `add` stores the fitted rows.

# %%
def fits(c):
    """Fit the six models of MODELS to one sample (proper motions and line-of-sight velocities).

    c : dict of ndarray, the catalogue (columns COLS).
    Returns a list of (model, nll, p, err): -lnL at the optimum, and the parameter vector and its inverse-Hessian errors (each of
    length len(kinfit.PARAMS); angles in deg, velocities in km/s, rates in deg/Gyr). Prints one line per model.
    """
    d = kinfit.data(*(c[k] for k in COLS))   # precomputed unit vectors, covariances etc. for the likelihood
    p, _, _ = kinfit.fit(d, free=FREE[2:])  # centre fixed at the adopted position first
    sym = kinfit.fit(d, free=FREE, p0=p)   # (p, err, nll) of 'sym' with the centre (dx, dy) free
    out = [('sym', sym[2], sym[0], sym[1])]
    # Each other model starts from the 'sym' optimum with its extra parameters freed.
    for name, extra in list(MODELS.items())[1:]:
        p, e, nll = kinfit.fit(d, free=FREE + extra, p0=sym[0])
        out.append((name, nll, p, e))
        # p indices: kinfit.PARAMS (0 dx, 4 inc, 5 theta, 6 v0, 17 m2c, 18 m2s, 24 qlos, 25 didt, 26 dthdt)
        print(f'  {name:22s} -dlnL {sym[2] - nll:8.1f}  dx {p[0]:.3f} i {p[4]:.1f} theta {p[5]:.1f} v0 {p[6]:.1f} '
              f'qlos {p[24]:+.3f} |m2| {np.hypot(p[17], p[18]):.3f} didt {p[25]:+.0f} dthdt {p[26]:+.0f}', flush=True)
    return out


def mock_of(c, q):
    """Mock of catalogue c: proper motions and velocities drawn from parameter vector q, with k = 1 and no outliers.

    c : dict of ndarray (columns COLS); q : ndarray, kinfit.PARAMS vector (the fitted model to draw from).
    Returns (mock catalogue dict, the parameter vector actually used: q with k = 1 and logit_fout = -8, the mock's truth).
    """
    q = q.copy()   # do not change the fitted vector stored in `fitted`
    q[kinfit.PARAMS.index('k')], q[kinfit.PARAMS.index('logit_fout')] = 1.0, -8.0  # Gaia errors as quoted; outlier fraction ~3e-4
    d = kinfit.data(*(c[k] for k in COLS))
    # Model mean proper motions [mas/yr], line-of-sight velocity [km/s] and disc distance D [kpc] of each star.
    mra, mde, mlos, D = (np.asarray(v) for v in kinfit.model(q, d['rhat'], d['ea'], d['ed']))
    rng = np.random.default_rng(7)   # fixed seed
    # Proper motions: Gaia covariance plus sigma (q[8], km/s) as a proper motion at D; velocities: e_vlos and sigma_los (q[12]).
    cov = np.asarray(d['cov']) + ((q[8] / (KAPPA * D)) ** 2)[:, None, None] * np.eye(2)
    # (N, 2) correlated noise [mas/yr]: Cholesky factor of each 2 x 2 covariance times two unit normals.
    noise = np.einsum('nij,nj->ni', np.linalg.cholesky(cov), rng.normal(size=(len(D), 2)))
    # Velocities only for stars with a measured one (NaN elsewhere, as in the data).
    vlos = np.where(np.isfinite(c['vlos']), mlos + rng.normal(size=len(D)) * np.hypot(c['evlos'], q[12]), np.nan)
    return dict(c, pmra=mra + noise[:, 0], pmdec=mde + noise[:, 1], vlos=vlos), q


rows, fitted = [], {}  # output rows; fitted[sample][model] = parameter vector (the mocks are drawn from these)


def add(name, c, out, truth=None):
    """Append the rows of fits() output `out` for sample `name` (truth: the mock's parameter vector)."""
    fitted[name] = {m: q for m, _, q, _ in out}
    # One row per model: number of line-of-sight velocities, -lnL, parameters, errors and (mocks) the truth.
    for model, nll, p, e in out:
        rows.append(dict(sample=name, model=model, n_los=int(np.isfinite(c['vlos']).sum()), nll=nll,
                         **dict(zip(kinfit.PARAMS, p)), **{f'{n}_err': v for n, v in zip(kinfit.PARAMS, e)},
                         **({} if truth is None else {f'truth_{n}': v for n, v in zip(kinfit.PARAMS, truth)})))


# %% [markdown]
# ### Load the data
# The giants of the five [Fe/H] bins (`lmcold.samples.giants()`, APOGEE and BOSS merged), each cut to 2 < $\rho$ < 9 deg from
# the adopted centre (`ring`).

# %%
# List of (name, catalogue dict) per [Fe/H] bin; ring keeps 2 < rho < 9 deg.
giants = [(f'giants {lab}', ring(c)) for lab, c in samples.giants()]

# %% [markdown]
# ### Fits of the data and the mocks
# The six models fitted to each giant bin (the number of line-of-sight velocities is printed), then two mocks drawn at the stars of
# the giants -0.8 < [Fe/H] < -0.5, from that bin's 'stream' and 'turn' fits, each fitted with the same six models.

# %%
# Data: the six models per giant bin.
for name, c in giants:
    print(name, int(np.isfinite(c['vlos']).sum()), flush=True)
    add(name, c, fits(c))
base_name, base = giants[MOCK_BIN]   # the bin whose stars carry the mocks
# Mocks: drawn from the bin's 'stream' or 'turn' fit, then fitted with the six models.
for mname, model in (('mock stream', 'stream'), ('mock turn', 'turn')):
    print(mname, flush=True)
    mock, truth = mock_of(base, fitted[base_name][model])
    add(mname, mock, fits(mock), truth)
t = pd.DataFrame(rows)   # one row per (sample, model)
t.to_csv(ROOT / 'results/kin_turning.csv', index=False, float_format='%.5f')

# %% [markdown]
# ### The figure
# Four panels with one column of points per sample: likelihood gain over 'sym', the two rates of change of the orientation and the
# streaming amplitude; the mock truths as red crosses, and the van der Marel et al. (2002) di/dt as a shaded band.

# %%
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
groups = list(t.groupby('sample', sort=False))   # (sample, rows) in the order fitted: five giant bins, then the two mocks
fig, ax = panel_grid(1, 4, width=9.0, left=0.5, right=0.2, hgap=0.55, top=0.55, bottom=1.2)
mk = dict(zip(MODELS, 'o^sDvP'))  # marker per model; points of one sample offset by 0.12 in x per model
# Sample k at x = k; legend labels only from the first sample.
for k, (_, s) in enumerate(groups):
    s = s.set_index('model')
    for j, m in enumerate(MODELS):
        r, x, lab = s.loc[m], k + 0.12 * (j - 2.5), m if k == 0 else None
        # (a) likelihood gain over 'sym'; (b), (c) turn rates with errors; (d) streaming amplitude.
        if m != 'sym':
            ax[0, 0].plot(x, s.loc['sym', 'nll'] - r.nll, mk[m], color='k', ms=3, label=lab)
        if 'turn' in m:
            ax[0, 1].errorbar(x, r.didt, yerr=r.didt_err, fmt=mk[m], color='k', ms=3, lw=0.6, label=lab)
            ax[0, 2].errorbar(x, r.dthdt, yerr=r.dthdt_err, fmt=mk[m], color='k', ms=3, lw=0.6, label=lab)
        if 'stream' in m:
            ax[0, 3].plot(x, np.hypot(r.m2c, r.m2s), mk[m], color='k', ms=3, label=lab)
    # Mock samples: their truth (same in every row) as red crosses.
    if 'truth_didt' in s and np.isfinite(s.truth_didt.iloc[0]):
        q = s.iloc[0]
        ax[0, 1].plot(k, q.truth_didt, 'rx', ms=5)
        ax[0, 2].plot(k, q.truth_dthdt, 'rx', ms=5)
        ax[0, 3].plot(k, np.hypot(q.truth_m2c, q.truth_m2s), 'rx', ms=5)
labels = [g for g, _ in groups]   # sample names for the x ticks
# Axis labels, sample names as rotated ticks, legends above the panels.
for a, lab in zip(ax[0], (r'$-\Delta\ln L$ over sym', 'di/dt [deg/Gyr]', r'd$\theta$/dt [deg/Gyr]', '|m2| [fraction of vrot]')):
    a.set(ylabel=lab, xticks=range(len(labels)), xlim=(-0.5, len(labels) - 0.5))
    a.set_xticklabels(labels, rotation=70, ha='right', fontsize=4)
    a.legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=4, frameon=False, ncol=2)  # above the panel
ax[0, 0].set_yscale('symlog')   # symmetric log scale: linear near zero, logarithmic beyond
# Zero lines of the two turn rates.
for a in ax[0, 1:3]:
    a.axhline(0, color='0.6', lw=0.5)
ax[0, 1].axhspan(-103 - 61, -103 + 61, color='C0', alpha=0.15, lw=0)  # van der Marel et al. 2002, di/dt = -103 +- 61 deg/Gyr
fig.savefig(ROOT / 'figures/fig_kin_turning.png', dpi=200)
