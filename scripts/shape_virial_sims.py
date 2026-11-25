"""Test of the rotation-implied flattening (lmcold.virial, paper Section 5) on simulated analogues, each in its own potential (KB
chapter 'The shape of the old population', section 'The flattening of simulated analogues').

For every zoom analogue with all particle types (Auriga L3 x 5, FIRE-2 x 3; lmcold.sims.masses; the TNG50 cutouts hold stars only)
and the two metal-poor [Fe/H] bins (populations 0 and 1: [Fe/H] < -1.5 and -1.5 to -1.1 on the LMC scale, lmcold.samples.FEH_BINS):
- q_true, h: the axis ratio (lmcold.spheroid.shape) and exponential scale length (scripts/kin_vc_drift.scale_length: the weighted
  surface density in 1-kpc annuli over 2-8 kpc, as for the LMC) of the bin's particles (scripts/sims_observe.particles, weighted by
  their expected BOSS giants E);
- potential: lmcold.virial.grid_potential of all particles in the disc frame of the stars (lmcold.sims.disc_frame);
- q_model: lmcold.virial.flattening of the kinematic model fitted to the analogue's mock giants (results/sims_observed.csv, p_*),
  in the analogue's potential ('sim') and in P1-P3: what the LMC procedure gives.

Reads: the particles of the Auriga and FIRE-2 analogues, stars (lmcold.sims.load) and all particle types (lmcold.sims.masses), from
the snapshots fetched by scripts/fetch_auriga.py and scripts/fetch_fire2.py (docs/release.md: about 120 GB and 8 GB); the inputs of scripts/sims_observe.setup, and results/sims_observed.csv (the kinematic
fits to the analogues' mock giants, written by scripts/sims_observe.py).
Writes results/shape_virial_sims.csv (suite, name, population, q_true, h, q_model_sim, q_model_P1, q_model_P2, q_model_P3), read by
scripts/paper_text_shape.py.
The particles themselves are not tested against the theorem: their LMC-scale [Fe/H] exists only inside the RHO_MASS cone of
scripts/sims_observe, so the bins are truncated tracers.

Run: .venv/bin/python scripts/shape_virial_sims.py   (after scripts/sims_observe.py)
"""
# %% [markdown]
# ### Settings
# The modules: the particle loading and observing steps of `scripts/sims_observe.py`, the scale-length fit of `scripts/kin_vc_drift.py`,
# and the virial solver of `lmcold.virial`.

# %%
import numpy as np
import pandas as pd

from kin_vc_drift import scale_length
from sims_observe import particles, setup
from lmcold import kinfit, samples, sims, spheroid, virial
from lmcold.sources import ROOT

P = kinfit.PARAMS  # parameter names of a kinematic model vector (columns p_<name> of results/sims_observed.csv)

if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The inputs shared by all analogues (`sims_observe.setup`: the LMC frame, the cumulative-mass knots of the [Fe/H] mapping, the
    # $n_{\rm box}$ and survival grids) and the kinematic fits to each analogue's mock giants. The particles are read per analogue in
    # the next step.

    # %%
    rng = np.random.default_rng(13)
    frame, _, _, knots, nbox, survival = setup()
    obs = pd.read_csv(ROOT / 'results/sims_observed.csv', dtype={'name': str})  # fitted model p_* per suite, name, population

    # %% [markdown]
    # ### True and recovered axis ratios
    # For each Auriga and FIRE-2 analogue: the axisymmetrised potential of all its particles in the disc frame of its stars; then, for
    # the two metal-poor bins, the true axis ratio and scale length of the bin's particles, and the axis ratio that the kinematic model
    # fitted to its mock giants implies in the analogue's own potential and in P1-P3.

    # %%
    rows = []
    for suite, name in [(s, n) for s, n in sims.names() if s != 'TNG50']:
        st = sims.load(suite, name, survival)  # star particles
        _, _, rot = sims.disc_frame(st['x'], st['v'], st['m'])  # rotation matrix to the disc frame of the stars
        xm, m = sims.masses(suite, name)  # all particles: positions (N, 3) [kpc], masses [Msun]
        grad = virial.grid_potential(xm @ rot.T, m, rng)  # gradient function of the analogue's potential
        X, _, _, _, feh, E, _ = particles(suite, name, frame, knots, nbox, survival)  # disc-frame X [kpc], LMC-scale [Fe/H], expected BOSS giants E
        for j in (0, 1):
            b = (feh >= samples.FEH_BINS[j][0]) & (feh < samples.FEH_BINS[j][1])
            q, h = spheroid.shape(X[b], E[b]), scale_length(np.hypot(X[b, 0], X[b, 1]), E[b], np.zeros(b.sum()))[0]  # q_true, h [kpc]
            o = obs[(obs.suite == suite) & (obs.name == name) & (obs.population == j)]
            row = dict(suite=suite, name=name, population=j, q_true=q, h=h)
            if len(o) and np.isfinite(o.vrot4.iloc[0]):  # the bin has a fitted rotation speed vrot4 (otherwise the q_model columns stay empty)
                p = o[[f'p_{n}' for n in P]].to_numpy()[0]
                row['q_model_sim'] = virial.flattening(p, h, grad, rng)[0]
                for pn, g in virial.POTENTIALS.items():
                    row[f'q_model_{pn}'] = virial.flattening(p, h, g, rng)[0]
            rows.append(row)
            print({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / 'results/shape_virial_sims.csv', index=False, float_format='%.4f')
