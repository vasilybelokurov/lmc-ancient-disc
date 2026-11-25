"""Robustness checks of the velocity-ellipsoid model for the giant [Fe/H] bins (paper, Sections 4.2 and 4.7; numbers of KB chapter 7
on an inner disc in its own plane, a line-of-sight scale and one Gaia error scale per group, written into results/macros_kinsupport.tex
by scripts/paper_text_kin_support.py).

Maximum-likelihood fits (lmcold.kinfit.fit, ellipsoid=True; proper motions and line-of-sight velocities; release giants in their
measured bins at 2 < rho < 9 deg), each bin on its own:
- 'reference': the Level-2 ellipsoid model as fitted for the paper (fit_kinematics_mcmc.ml, model 'ellipsoid').
- 'tilt': the reference plus an inner disc (rho < kinfit.RHO_TILT) in its own plane (dinc, dtheta free), all else free.
- 'qlos': the reference plus a scale (1 + qlos) of the internal motion along the line of sight, all else free.
- 'bright' and 'faint': the stars brighter and fainter than the bin's median G fitted separately, each with its own Gaia error
  scale k, rotation curve, ellipsoid and outlier fraction (OWN), the geometry, systemic motion and streaming held at the reference:
  a test of one k per group. If k depended on magnitude, the two halves would need different k and, with one k, the dispersions
  would absorb the difference.

Writes results/kin_ellipsoid_checks.csv: population, model, n, lnL (of the stars fitted), dlnL (model minus reference, same stars;
NaN for the halves), inc, theta, dinc, qlos, vrot4 = v0 tanh(4 kpc / R0) [km/s], sigma_R, sigma_phi, sigma_z [km/s], dlnsig_dR,
k, G_split.

Inputs: the giant catalogues of lmcold.samples.giants() (Zenodo bundle 1). No posterior samples: every fit here is a maximum-
likelihood fit started as in scripts/fit_kinematics_mcmc.py.

Run: .venv/bin/python scripts/kin_ellipsoid_checks.py
"""
# %% [markdown]
# ### Settings
# The parameters left free in the alternative models, and those refitted in each magnitude half.

# %%
import numpy as np
import pandas as pd

from fit_kinematics_mcmc import COLS, ml, ring
from lmcold import kinfit, samples
from lmcold.sources import ROOT

P = kinfit.PARAMS.index
# Free parameters of the ellipsoid model in the paper's fits (fit_kinematics_mcmc FIXED['ellipsoid'] held, vlos0 free).
FREE = tuple(n for n in kinfit.BASE + ('vlos0',) + kinfit.STREAM if n != 'sigma') + kinfit.ELLIPSOID + ('vslope',)
OWN = ('v0', 'R0', 'vslope', 'k', 'logit_fout') + kinfit.ELLIPSOID  # refitted in each magnitude half


# %% [markdown]
# ### One row of the output table
# `row` collects the geometry, rotation speed at 4 kpc and dispersions of one fit.

# %%
def row(pop, model, d, p, nll, n, **extra):
    """One output row (dict) for parameter vector p (shape (33,), kinfit.PARAMS order) of population pop.

    nll is the negative log likelihood at p, n the number of stars fitted; extra adds columns (dlnL, G_split). d is unused.
    vrot4 [km/s] is v0 tanh(R_SIG / R0), the rotation speed at kinfit.R_SIG = 4 kpc; dispersions in km/s, angles in deg.
    """
    g = lambda name: float(p[P(name)])  # noqa: E731
    return dict(population=pop, model=model, n=n, lnL=-nll, inc=g('inc'), theta=g('theta'), dinc=g('dinc'), qlos=g('qlos'),
                vrot4=g('v0') * np.tanh(kinfit.R_SIG / g('R0')), sigma_R=g('sigma_R'), sigma_phi=g('sigma_phi'),
                sigma_z=g('sigma_z'), dlnsig_dR=g('dlnsig_dR'), k=g('k')) | extra


# %% [markdown]
# ### Load the giants and fit each bin
# The giants of each [Fe/H] bin (Zenodo bundle 1, through `lmcold.samples.giants()`) at 2-9 deg: the reference fit, the two
# extended models started from it, and the two magnitude halves. The table goes to `results/kin_ellipsoid_checks.csv`.

# %%
rows = []
for label, cat in samples.giants():
    pop = f'giants {label}'
    c = ring(cat | dict(dmu=np.full(len(cat['ra']), np.nan)))
    d = kinfit.data(*(c[k] for k in COLS))
    p0 = ml(d, 'ellipsoid')   # reference: the maximum-likelihood start of the paper's MCMC fit
    nll0 = float(kinfit.neg_loglike(p0, d, True))
    rows.append(row(pop, 'reference', d, p0, nll0, len(c['ra']), dlnL=0.0))
    for model, extra in (('tilt', kinfit.TILT), ('qlos', ('qlos',))):
        p, _, nll = kinfit.fit(d, free=FREE + extra, p0=p0, ellipsoid=True)
        rows.append(row(pop, model, d, p, nll, len(c['ra']), dlnL=nll0 - nll))
        print(rows[-1], flush=True)
    gsplit = float(np.median(c['G']))   # Gaia G [mag]: median of the bin, the split between bright and faint halves
    for half, keep in (('bright', c['G'] < gsplit), ('faint', c['G'] >= gsplit)):
        h = {k: np.asarray(v)[keep] for k, v in c.items()}
        dh = kinfit.data(*(h[k] for k in COLS))
        p, _, nll = kinfit.fit(dh, free=OWN, p0=p0, ellipsoid=True)
        rows.append(row(pop, half, dh, p, nll, int(keep.sum()), dlnL=np.nan, G_split=gsplit))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / 'results/kin_ellipsoid_checks.csv', index=False, float_format='%.4f')
