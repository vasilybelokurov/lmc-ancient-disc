"""Flattening of each kinematic group implied by its rotation and dispersions (paper Section 5; lmcold.virial; KB chapter 'The shape
of the old population', section 'Flattening implied by the motions' and Table shape_q).

For every giant [Fe/H] bin (stage-1 ellipsoid fits at 2-9 deg, data/kin_mcmc_data_ellipsoid_samples.npz) and the RR Lyrae (stage-1
fit at 4-9 deg, data/kin_mcmc_data_ellipsoid_outer_samples.npz, whose dispersions the paper quotes): the kinematic model and the group's
exponential scale length h (results/kin_vc_drift_h.csv) are realised as a round tracer, and the axis ratio q at which the potential
side of the tensor virial theorem equals the kinetic side is found in each potential of lmcold.virial.POTENTIALS.
Uncertainty: NDRAW joint posterior draws (all parameters of one draw together), each with h drawn from a Gaussian of its bootstrap
16-84 half-width. Central extrapolation: the fits constrain the dispersions at R >~ 2 kpc; the exponential gradient carries them
inward, where much of the tracer's kinetic energy lies, so every case is also run with the dispersions held constant inside
R_FLOOR (lmcold.spheroid.velocities).

Absolute balance: the ratios do not test the depth of the potential, so 2K/|W| (lmcold.virial.balance) of the posterior-median model
realised at its q is also given (1 if the potential supports the measured motions).

Reads: the posterior samples data/kin_mcmc_data_ellipsoid_samples.npz and kin_mcmc_data_ellipsoid_outer_samples.npz (Zenodo bundle 1;
scripts/fit_kinematics_mcmc.py) and results/kin_vc_drift_h.csv (scale lengths h with bootstrap 16th and 84th percentiles [kpc],
scripts/kin_vc_drift.py).
Writes results/shape_virial.csv (population, potential, r_floor, h, omega_kin, q and balance of the posterior-median vector, q_p16,
q_p50, q_p84 over the draws with a solution, f_none: draws with no solution, f_prolate: draws with q > 1), read by
scripts/paper_shape_virial.py, paper_table_shape.py, paper_text_shape.py and shape_rrl_cells.py.

Run: .venv/bin/python scripts/shape_virial.py
"""
# %% [markdown]
# ### Settings
# The number of posterior draws per group and the radius inside which the dispersions are held constant in the second set of runs.

# %%
import re

import numpy as np
import pandas as pd

from lmcold import kinfit, spheroid, virial
from lmcold.sources import ROOT

NDRAW = 200  # joint posterior draws per group
R_FLOOR = 2.0  # kpc: dispersions held constant inside this radius (the fits constrain them at R >~ 2 kpc)
P = kinfit.PARAMS  # parameter names of a kinematic model vector

# %% [markdown]
# ### Posterior draws
# `draws` reads the stage-1 posterior samples of one population and returns its posterior-median model and `NDRAW` joint draws, each a
# full parameter vector of `lmcold.kinfit`.


# %%
def draws(npz, pop, rng):
    """Posterior-median vector and NDRAW joint draws (kinfit vectors) of population index pop, stage 1, from a samples file.

    Parameters
    ----------
    npz : str
        File name in data/ of the posterior samples (keys 'stage|population|parameter[_k]', and 'names', the population labels).
    pop : int
        Population index in that file (0-4: the giant [Fe/H] bins; 5: the RR Lyrae in the outer-fit file).
    rng : numpy.random.Generator
        Picks the draws (sample indices, the same index for all parameters of one draw).

    Returns
    -------
    (str, numpy.ndarray, list of numpy.ndarray)
        Population label, posterior-median vector (len(kinfit.PARAMS); parameters absent from the file keep kinfit.start()), and the
        NDRAW draw vectors. Units as kinfit (deg, km/s, kpc).
    """
    z = np.load(ROOT / 'data' / npz, allow_pickle=True)
    # parameter name -> key, for the stage-1 keys '1|pop|name' of this population (a '_k' suffix of vector parameters removed)
    keys = {re.sub(r'_\d+$', '', k.split('|')[2]): k for k in z.files if k.startswith(f'1|{pop}|')}
    # keep only the parameters of kinfit.PARAMS
    keys = {n: k for n, k in keys.items() if n in P}
    flat = {n: z[k].reshape(-1) for n, k in keys.items()}  # parameter name: all samples (chains flattened)
    # start from kinfit.start() so parameters absent from the file keep their default values
    med, out = kinfit.start(), []
    # posterior median of each sampled parameter
    for n, v in flat.items():
        med[P.index(n)] = np.median(v)
    # NDRAW random sample indices; each draw takes all parameters from the same sample
    for i in rng.integers(0, len(next(iter(flat.values()))), NDRAW):
        p = kinfit.start()
        for n, v in flat.items():
            p[P.index(n)] = v[i]
        out.append(p)
    return str(z['names'][pop]), med, out


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The scale lengths $h$ of the groups, and the posterior draws of the five giant bins (2-9 deg fit) and of the RR Lyrae (4-9 deg fit).

    # %%
    # fixed seed: draws and realisations are reproducible
    rng = np.random.default_rng(5)
    H = pd.read_csv(ROOT / 'results/kin_vc_drift_h.csv', index_col=0)  # h, h_p16, h_p84 [kpc] per population
    # the five giant [Fe/H] bins (indices 0-4) of the 2-9 deg fit; the RR Lyrae (index 5) of the 4-9 deg fit
    groups = [draws('kin_mcmc_data_ellipsoid_samples.npz', i, rng) for i in range(5)]
    groups.append(draws('kin_mcmc_data_ellipsoid_outer_samples.npz', 5, rng))

    # %% [markdown]
    # ### The axis ratio in each potential
    # For each group, with and without the central floor on the dispersions, and in each potential P1-P3: $q$ of the posterior-median
    # model (`lmcold.virial.flattening`), $2K/|W|$ of that model realised at its $q$, and $q$ for each posterior draw with $h$ drawn
    # from a Gaussian of half the bootstrap 16th-84th range (floored at 0.3 kpc). Draws whose kinematics no tracer on the $q$ grid
    # matches are counted in `f_none`; those with $q>1$ (elongated along the symmetry axis) in `f_prolate`.

    # %%
    rows = []
    for pop, med, ds in groups:
        # scale length and its bootstrap 16th/84th percentiles [kpc]
        h, hlo, hhi = H.loc[pop, ['h', 'h_p16', 'h_p84']]
        # without (0) and with the constant-dispersion floor inside R_FLOOR [kpc]
        for r_floor in (0.0, R_FLOOR):
            for name, grad in virial.POTENTIALS.items():
                q, om = virial.flattening(med, h, grad, rng, r_floor)  # q and Omega_kin of the posterior-median model
                X = spheroid.spheroid(h, q, virial.NREAL, rng) if np.isfinite(q) else None  # positions (NREAL, 3) [kpc] at that q
                bal = virial.balance(X, spheroid.velocities(med, X, rng, r_floor), grad) if X is not None else np.nan  # 2K/|W|
                # q for each draw, with h drawn from a Gaussian of sigma = half the 16th-84th range, floored at 0.3 kpc
                qd = np.array([virial.flattening(d, max(h + 0.5 * (hhi - hlo) * rng.normal(), 0.3), grad, rng, r_floor)[0] for d in ds])
                ok = np.isfinite(qd)  # draws with a solution
                # 16th, 50th, 84th percentiles of q over the draws with a solution
                pc = np.percentile(qd[ok], [16, 50, 84]) if ok.any() else [np.nan] * 3
                rows.append(dict(population=pop, potential=name, r_floor=r_floor, h=h, omega_kin=om, q=q, balance=bal, q_p16=pc[0], q_p50=pc[1],
                                 # f_prolate: fraction of all draws with q > 1
                                 q_p84=pc[2], f_none=1 - ok.mean(), f_prolate=np.mean(qd[ok] > 1) * ok.mean()))
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / 'results/shape_virial.csv', index=False, float_format='%.4f')
