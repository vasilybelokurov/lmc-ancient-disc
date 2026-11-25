"""Does the disc-frame velocity-ellipsoid likelihood (lmcold.kinfit, ellipsoid=True) recover sigma_R, sigma_phi and sigma_z from mocks
that are not drawn from it? Acceptance test of the ellipsoid model of the kinematics section (appendix; KB chapter 6, 'Recovery of
the velocity ellipsoid from mocks', figure paper_kin_ellipsoid_mock; the recovery numbers there are written from
results/kin_ellipsoid_mock.csv by scripts/paper_text_kin_model.py).

Mocks: the physical mocks of scripts/fig_kin_mock_physical.py (draw(); positions drawn in 3D from an exponential disc, Gaia
covariances of the nearest RR Lyrae member, line-of-sight errors 2 km/s, 15 000 stars at 2 < rho < 9 deg), with constant
dispersions: 'cold thin' (20, 20, 20 km/s, thin, mean vc tanh(R/R0): the fitting model up to its rotation-curve shape), 'hot'
(40, 28, 30, thin, asymmetric-drift mean), 'hot, no drift' (40, 28, 30, mean vc tanh(R/R0)) and 'hot thick' (40, 28, 30, Gaussian
height dispersion 1 kpc, drift) and 'hot spheroid' (40, 28, 30, drift, positions from an oblate K0 spheroid of axis ratio Q_SPH:
the rotation-implied shape of the metal-poor giants, paper Section 5, whose thickness grows with R). None has streaming or a radial dispersion gradient; the fit has both free.
Fits: maximum likelihood of the Level-2 ellipsoid model with everything free as for the data (fit_kinematics_mcmc.ml, model
'ellipsoid', outer slope vslope of the rotation curve free): 'PM + LOS' (the giants' case) and 'PM' (proper motions only, the
RR Lyrae case; sigma_z then comes only from the proper motions); and 'PM + LOS, tanh': the same with vslope held at 0, a pure tanh
rotation curve, to show what the slope removes. Errors from the inverse Hessian.

Writes results/kin_ellipsoid_mock.csv (mock, data, truth and fitted sigma_R, sigma_phi, sigma_z with errors, dlnsig_dR, vslope, inc,
theta, v0) and figures/paper_kin_ellipsoid_mock.png: three equal panels (sigma_R, sigma_phi, sigma_z [km/s]) against the mock;
filled circles PM + LOS, open circles PM only, grey squares PM + LOS with a tanh curve, truth as red bars. Paper figure (appendix app:kin).

Inputs: the RR Lyrae members of lmcold.samples.rrab() (data/rrab.parquet, Zenodo bundle 1): their sky positions set the footprint
and their Gaia proper-motion covariances the mock errors. Run time: about 20 min on one machine (docs/release_scripts.md).

Run: .venv/bin/python scripts/paper_kin_ellipsoid_mock.py
"""
# %% [markdown]
# ### Settings
# The names of the five mocks (their dispersions and shapes are set in `fig_kin_mock_physical.MOCKS`), the axis ratio of the
# spheroid, and the three dispersions compared.

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_kin_mock_physical import C, HOT, MOCKS, draw
from fit_kinematics_mcmc import ml, ring
from lmcold import kinfit, samples
from lmcold.layout import panel_grid
from lmcold.sources import ROOT

NAMES = ('cold thin', 'hot', 'hot, no drift', 'hot thick', 'hot spheroid')   # the five mocks, in plotting order (module docstring)
Q_SPH = 0.5  # axis ratio of the 'hot spheroid' mock
COMPS = ('sigma_R', 'sigma_phi', 'sigma_z')  # km/s: the disc-frame dispersions compared with the truth
LABELS = dict(sigma_R=r'$\sigma_R$', sigma_phi=r'$\sigma_\phi$', sigma_z=r'$\sigma_z$')   # y-axis labels of the panels

# %% [markdown]
# ### Load the data
# The RR Lyrae members (`lmcold.samples.rrab()`, from `data/rrab.parquet`) at 2-9 deg: only their positions and Gaia proper-motion
# errors are used, as the footprint and error model of the mocks.

# %%
# RR Lyrae members (dict of columns): only positions [deg] and proper-motion errors [mas/yr] are used
rr = samples.rrab()
c = ring({k: rr[k] for k in C})   # positions and proper-motion errors (columns C) at 2 < rho < 9 deg

# %% [markdown]
# ### Draw the mocks and fit them
# Each mock takes the sky positions and Gaia errors of the RR Lyrae members (`fig_kin_mock_physical.draw`); the 'hot spheroid'
# draws its positions from a $K_0$ spheroid of axis ratio `Q_SPH`. Each is fitted three times by maximum likelihood: proper motions
# and line-of-sight velocities, proper motions only, and both with a pure tanh rotation curve. This step takes a while.

# %%
# One output row per (mock, kind of fit)
rows = []
for name in NAMES:
    # Spheroid: HOT dispersions with asymmetric drift, positions from the K0 spheroid of axis ratio Q_SPH;
    # the other mocks: (height dispersion h [kpc], dispersions [km/s], drift flag) from MOCKS; same seed 100 for all
    if name == 'hot spheroid':
        disp = HOT
        m = draw(c, 0.0, HOT, True, np.random.default_rng(100), q=Q_SPH)
    else:
        h, disp, drift = MOCKS[name]
        m = draw(c, h, disp, drift, np.random.default_rng(100))
    for data in ('PM + LOS', 'PM', 'PM + LOS, tanh'):
        # Line-of-sight velocities [km/s] kept for 'PM + LOS' fits, NaN (missing) for the proper-motion-only fit
        vlos = m['vlos'] if data.startswith('PM + LOS') else np.full(len(m['ra']), np.nan)
        d = kinfit.data(*(m[k] for k in C), vlos, m['evlos'])
        # Maximum-likelihood fit of the ellipsoid model; p: kinfit.PARAMS vector
        p = ml(d, 'ellipsoid')
        if data.endswith('tanh'):  # refit with a pure tanh rotation curve, everything else free as before
            p[kinfit.PARAMS.index('vslope')] = 0.0
            free_all = tuple(n for n in kinfit.PARAMS if n not in ('m1c', 'm1s', 'sigma', 'sigma_los', 'dmu0', 'sigma_mu', 'vslope')
                             + kinfit.TILT + ('qlos',) + kinfit.TURN)
            p = kinfit.fit(d, free=free_all, p0=p, ellipsoid=True)[0]
        # Errors: inverse Hessian of a refit of the rotation curve, k, f and the ellipsoid, started at the best fit p.
        free = [n for n in kinfit.PARAMS if n in ('v0', 'R0', 'k', 'logit_fout') + kinfit.ELLIPSOID]
        _, e, _ = kinfit.fit(d, free=tuple(free), p0=p, ellipsoid=True)
        # g(n): value of parameter n in p (or in the error vector e); the row holds truth, fit and errors [km/s]
        g = lambda n, v=p: v[kinfit.PARAMS.index(n)]  # noqa: E731
        rows.append(dict(mock=name, data=data, **{f'truth_{n}': s for n, s in zip(COMPS, disp)},
                         **{n: g(n) for n in COMPS + ('dlnsig_dR', 'vslope', 'inc', 'theta', 'v0')}, **{f'{n}_err': g(n, e) for n in COMPS}))
        print(rows[-1], flush=True)
# %% [markdown]
# ### The table and the figure
# The fitted dispersions with their inverse-Hessian errors go to `results/kin_ellipsoid_mock.csv`; the figure compares them with
# the true values.

# %%
# Table of all fits, written to results/kin_ellipsoid_mock.csv
t = pd.DataFrame(rows)
t.to_csv(ROOT / 'results/kin_ellipsoid_mock.csv', index=False, float_format='%.4f')

# Three equal panels (one per dispersion); x: mock index, the three kinds of fit offset by -0.15, 0, +0.15
plt.style.use(ROOT / 'src/lmcold/paper.mplstyle')
fig, ax = panel_grid(1, 3, width=7.0, left=0.55, right=0.1, hgap=0.5, top=0.45, bottom=0.75)
x = np.arange(len(NAMES))
for a, n, letter in zip(ax[0], COMPS, 'abc'):
    for data, dx, fmt, colour, mfc in (('PM + LOS', -0.15, 'o', 'k', 'k'), ('PM + LOS, tanh', 0.0, 's', '0.55', '0.55'),
                                       ('PM', 0.15, 'o', 'k', 'w')):
        # Fitted dispersion [km/s] with its inverse-Hessian error, mocks in NAMES order
        s = t[t.data == data].set_index('mock').loc[list(NAMES)]
        a.errorbar(x + dx, s[n], yerr=s[f'{n}_err'], fmt=fmt, color=colour, mfc=mfc, ms=4, lw=0.8, label=data)
    # True dispersion of each mock as a red bar
    tr = t[t.data == 'PM + LOS'].set_index('mock').loc[list(NAMES)][f'truth_{n}']
    a.hlines(tr, x - 0.3, x + 0.3, color='C3', lw=1.2, label='truth')
    # Axes: mock names as tick labels, y from 0 to 60 km/s; panel letter in the top left
    a.set(xticks=x, xlim=(-0.5, len(NAMES) - 0.5), ylim=(0, 60), ylabel=LABELS[n] + r' [km s$^{-1}$]')
    a.set_xticklabels(NAMES, rotation=30, ha='right')
    a.text(0.04, 0.92, f'({letter})', transform=a.transAxes, fontsize=7)
ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=4, fontsize=6, frameon=False)
fig.savefig(ROOT / 'figures/paper_kin_ellipsoid_mock.png')
