"""Log-likelihood gain of the velocity-ellipsoid model over the sky-aligned dispersion model (paper, appendix app:kin_sky; the
Delta ln L values of KB chapter 6, 'Tests' and 'The sky-aligned dispersion model', written into results/macros_kinmodel.tex by
scripts/paper_text_kin_model.py).

For each population and radial range (2 < rho < 9 deg and OUTER_INNER < rho < 9 deg) the log likelihood of the population's own
stars (lmcold.kinfit.neg_loglike; ellipsoid=True for the ellipsoid model) is evaluated at NDRAW posterior draws of its stage-1 fit
(each population on its own) of each model, read from data/kin_mcmc_data_<run>_samples.npz (scripts/fit_kinematics_mcmc.py:
runs 'stream' and 'ellipsoid', and their '_outer' versions). Parameters not sampled stay at their kinfit.start() values, as in the
fits. No refitting: the best of the draws stands in for the maximum likelihood (it agrees with a maximum-likelihood refit to 1 in
ln L where both were computed).

Writes results/kin_model_comparison.csv: rho_inner [deg], population, n (stars), dlnL_max (best ellipsoid draw minus best sky-aligned
draw), dlnL_median (same with the median over draws), dk (extra parameters of the ellipsoid model: sigma_R, sigma_phi, sigma_z,
dlnsig_dR, vslope replace sigma, sigma_los; 3 for the giants, 4 for the RR Lyrae, which have no sigma_los).

Inputs: the posterior samples data/kin_mcmc_data_{stream,ellipsoid}[_outer]_samples.npz (Zenodo bundle 1) and the star catalogues
of lmcold.samples (giants and RR Lyrae, also bundle 1).

Run: .venv/bin/python scripts/kin_model_comparison.py   (after fit_kinematics_mcmc.py data stream|ellipsoid [outer])
"""
# %% [markdown]
# ### Settings
# The number of posterior draws at which the likelihood is evaluated, and their indices (the same for every fit).

# %%
import jax
import numpy as np
import pandas as pd

from fit_kinematics_mcmc import COLS, OUTER_INNER, ring
from lmcold import kinfit, samples
from lmcold.sources import DATA, ROOT

NDRAW = 300  # posterior draws per model and population
DRAWS = np.random.default_rng(0).choice(4000, NDRAW, replace=False)  # same draw indices for every fit (4 chains x 1000)


# %% [markdown]
# ### Log-likelihood at the posterior draws
# `lnl` reads the stage-1 draws of one population from a run's samples file and evaluates the log likelihood of its stars at each.

# %%
def lnl(run, i, d, ellipsoid):
    """ln L of data d at the NDRAW stage-1 draws of population i in run.

    Parameters
    ----------
    run : str
        Run name of fit_kinematics_mcmc.py without 'data_' ('stream', 'ellipsoid', 'stream_outer', 'ellipsoid_outer').
    i : int
        Population index in the run (giant bins 0-4 metal-poor first, RR Lyrae 5).
    d : dict
        kinfit.data() arrays of that population's stars.
    ellipsoid : bool
        Use the velocity-ellipsoid likelihood (True for the 'ellipsoid' runs).

    Returns
    -------
    ndarray, shape (NDRAW,)
        ln L of all stars of d at each draw.
    """
    z = np.load(DATA / f'kin_mcmc_data_{run}_samples.npz')   # keys 'stage|population|site', arrays of posterior draws
    # Parameter vectors, shape (NDRAW, len(kinfit.PARAMS)), starting at the kinfit.start() values (kept for unsampled parameters).
    p = np.tile(kinfit.start(), (NDRAW, 1))
    # Overwrite each sampled parameter of stage 1, population i, with its draws at the indices DRAWS.
    for key in z.files:
        if key.startswith(f'1|{i}|'):
            n = key.split('|')[2]   # site name; a population's own parameters carry the suffix _0
            p[:, kinfit.PARAMS.index(n[:-2] if n.endswith('_0') else n)] = z[key].reshape(-1)[DRAWS]
    # Compiled ln L of the stars d as a function of the parameter vector; evaluated at each draw.
    f = jax.jit(lambda q: -kinfit.neg_loglike(q, d, ellipsoid))
    return np.array([float(f(q)) for q in p])


# %% [markdown]
# ### Load the stars and compare the two models
# For each radial range (2-9 deg, and 4-9 deg for the '_outer' runs) the giants of the five [Fe/H] bins and the RR Lyrae are read
# through `lmcold.samples` (Zenodo bundle 1) and cut by `ring`; for each population the best and the median ln L over the draws
# of the two models are compared and written to `results/kin_model_comparison.csv`.

# %%
rows = []
# Inner radius [deg] of the ring and run-name suffix: 2-9 deg ('' runs) and OUTER_INNER-9 deg ('_outer' runs).
for inner, suffix in ((2.0, ''), (OUTER_INNER, '_outer')):
    # Giant bins (no distances: dmu NaN), then the RR Lyrae (no line-of-sight velocities: vlos, evlos NaN), cut to the ring.
    pops = [(f'giants {lab}', ring(cat | dict(dmu=np.full(len(cat['ra']), np.nan)), inner)) for lab, cat in samples.giants()]
    rr = samples.rrab()
    nan = np.full(len(rr['ra']), np.nan)
    pops.append(('RR Lyrae', ring(rr | dict(vlos=nan, evlos=nan, dmu=nan), inner)))
    for i, (name, c) in enumerate(pops):
        d = kinfit.data(*(c[k] for k in COLS))   # likelihood inputs of this population
        # ln L at the draws, shape (NDRAW,): sky-aligned dispersion model ('stream' runs) and ellipsoid model.
        sky, ell = lnl('stream' + suffix, i, d, False), lnl('ellipsoid' + suffix, i, d, True)
        # Gains of the ellipsoid model: best draw minus best draw, median minus median; dk extra parameters (see docstring).
        rows.append(dict(rho_inner=inner, population=name, n=len(c['ra']), dlnL_max=ell.max() - sky.max(),
                         dlnL_median=np.median(ell) - np.median(sky), dk=4 if name == 'RR Lyrae' else 3))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / 'results/kin_model_comparison.csv', index=False, float_format='%.1f')
