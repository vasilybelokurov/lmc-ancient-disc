"""Kinematics at birth and today of the simulated analogues' stars, in birth cohorts: results/sims_birth.csv, the table of Fig.
sims_birth (scripts/paper_sims_birth.py; the Letter's Fig. 3) and of the Section 6 numbers (scripts/paper_text_sims_birth.py,
paper_text_sims_suites.py); its cohorts are also the birth points of Fig. sims_heating (a) (scripts/build_sims_heating.py uses the
same method).

Simulation (lmcold.sims.birth; peculiar birth velocity = BirthVel / a, tests/test_sims.py), for the 681 TNG50 and the five Auriga
level-3 analogues: the star particles within 10 kpc of the centre today (Auriga: in situ only, not in any Exsitu list; TNG50: all,
no assembly catalogue), in cohorts of COHORT Gyr of age (cohorts of fewer than 100 such stars are skipped).
- At birth: the galaxy each star formed in is followed by its centre and velocity as functions of formation time
  (lmcold.sims.progenitor_track); each star's birth position and velocity are taken relative to that track at its own formation
  time and rotated to the spin axis of the cohort's stars within R_BIRTH physical kpc (lmcold.sims.birth_phase_space). Measured:
  mass-weighted mean v_phi and sigma_z of the cohort's stars at 0.5 < R < R_BIRTH kpc (the inner cut keeps the centre's ~0.15 kpc
  uncertainty from scrambling v_phi). Cohorts with fewer than MIN_IN such stars get no birth value. Frame errors only add
  dispersion.
- Today: the same stars, wherever they are now, in the z = 0 disc frame (lmcold.sims.disc_frame).
Reads the z = 0 star particles with their birth positions and velocities (TNG50: the cutouts and birth files of
scripts/fetch_tng50.py; Auriga: the level-3 snapshots of scripts/fetch_auriga.py).
Writes results/sims_birth.csv: suite, halo, age [Gyr, cohort centre], n = stars of the cohort within 10 kpc, n_birth = stars
measured, vphi_birth, sz_birth, vphi_same, sz_same [km/s].

Run: .venv/bin/python scripts/manifest.py run build_sims_birth.py   (about 10 min on WORKERS cores; 52 s in the recorded run of
docs/run_times.txt)
"""
# %% [markdown]
# ### Settings
# The width of the cohorts, the birth sphere, the minimum numbers of stars and the radius of the stars used today.

# %%
from multiprocessing import Pool

import numpy as np
import pandas as pd

from lmcold import sims
from lmcold.sources import ROOT

COHORT = 0.1  # Gyr, width of the birth cohorts: shorter than the time in which the plane of star formation turns (< 0.5 Gyr);
# 0.5-Gyr cohorts mixed planes and lowered v_phi at birth by 0.3-0.4 in TNG50
R_BIRTH = 3.0  # physical kpc, birth sphere in which the spin axis and the kinematics are measured
MIN_IN = 50  # stars measured for a cohort to get a birth value
MIN_COHORT = 100  # stars of a cohort within R_TODAY, or it is skipped
R_TODAY = 10.0  # kpc, stars used: within this radius of the centre today
WORKERS = 6  # processes in the Pool (one analogue per process)


# %% [markdown]
# ### Helpers
# `mean_vphi_sigz` measures the rotation and vertical dispersion of a set of stars; `cohorts` does the whole measurement for one
# analogue: it reads its stars, builds the track of the galaxy they formed in, and measures every cohort at birth and today.

# %%
def mean_vphi_sigz(X, V, w):
    """Mass-weighted mean v_phi and sigma_z [km/s] of particles at X, V (N, 3) in a frame with the spin along +z.

    X [kpc], V [km/s], w (N,) weights [Msun]; sigma_z is about the weighted mean v_z."""
    # v_phi = (x v_y - y v_x) / R: azimuthal velocity about +z [km/s]
    vphi = (X[:, 0] * V[:, 1] - X[:, 1] * V[:, 0]) / np.hypot(X[:, 0], X[:, 1])
    # v_z about the weighted mean, so sigma_z is the dispersion
    vz = V[:, 2] - np.average(V[:, 2], weights=w)
    return np.average(vphi, weights=w), np.sqrt(np.average(vz ** 2, weights=w))


def cohorts(job):
    """One row per birth cohort of analogue job = (suite, name) (columns as results/sims_birth.csv)."""
    suite, name = job
    d = sims.birth(suite, name)   # z = 0 phase space, formation time and birth phase space of every star particle
    X, V, _ = sims.disc_frame(d['x'], d['v'], d['m'])   # today, in the z = 0 disc frame
    # stars within R_TODAY of the centre today, in situ only (exsitu is all False for TNG50)
    use = (np.linalg.norm(d['x'], axis=1) < R_TODAY) & ~d['exsitu']
    track = sims.progenitor_track(d['birth_x'][use], d['a_form'][use], d['birth_v'][use])   # centre and velocity against scale factor
    rows = []
    for lo in np.arange(0, 13.5, COHORT):   # cohort ages lo to lo + COHORT [Gyr]
        # indices of the used stars with ages in [lo, lo + COHORT) [Gyr]
        i = np.flatnonzero(use & (d['age'] >= lo) & (d['age'] < lo + COHORT))
        if len(i) < MIN_COHORT:
            continue
        # age: cohort centre [Gyr]; n: stars of the cohort within R_TODAY
        row = dict(suite=suite, halo=name, age=lo + COHORT / 2, n=len(i))
        # P, B: birth positions [kpc] and velocities [km/s] relative to the track, about the cohort's spin; k: stars at 0.5 < R < R_BIRTH
        P, B, k = sims.birth_phase_space(d['birth_x'][i], d['a_form'][i], d['birth_v'][i], d['m'][i], track, R_BIRTH)
        if k.sum() >= MIN_IN:
            row['vphi_birth'], row['sz_birth'] = mean_vphi_sigz(P[k], B[k], d['m'][i][k])
            row['vphi_same'], row['sz_same'] = mean_vphi_sigz(X[i[k]], V[i[k]], d['m'][i[k]])   # the same stars today
            row['n_birth'] = int(k.sum())
        rows.append(row)
    return rows


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data and measure every analogue
    # The TNG50 and Auriga analogues (FIRE-2 birth values are made by `scripts/fire2_birth.py`), one per process; each process reads
    # its analogue's particles through `lmcold.sims.birth` inside `cohorts`.

    # %%
    # (suite, name) of every TNG50 and Auriga analogue
    jobs = [s for s in sims.names() if s[0] in ('TNG50', 'Auriga')]
    with Pool(WORKERS) as pool:
        # one row per cohort of every analogue, flattened into one table
        t = pd.DataFrame([r for rows in pool.map(cohorts, jobs) for r in rows])
    # three decimals: ages [Gyr] and velocities [km/s]
    t.to_csv(ROOT / 'results/sims_birth.csv', index=False, float_format='%.3f')
