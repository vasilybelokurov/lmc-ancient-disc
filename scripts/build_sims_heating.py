"""The tables of Fig. sims_heating (scripts/paper_sims_heating.py) from the TNG50 main progenitors (scripts/fetch_tng50_progenitors.py:
SAMPLE of 60, FULL of 20): results/sims_heating_gas.csv, results/sims_heating_history.csv, results/sims_heating_coherence.csv.
The gas table also gives the star-forming gas band of the Letter's Fig. 4 (scripts/paper_letter_scramble.py), and the coherence table
the birth snapshots of scripts/build_sims_vcirc.py; scripts/sims_halo_settle.py imports coherence_rows.

- gas (panel a): star-forming gas (StarFormationRate > 0) at 0.5 < R < 3 kpc of each progenitor at GAS_SNAPS, about the spin axis of
  the star-forming gas within 3 kpc, velocities relative to its mean there: mass-weighted v_phi, sigma_z and sound speed cs, with the
  progenitor's stellar mass (SubLink tree, lmcold.sims.main_branch);
- history (panels a, b): the COHORT-Gyr cohorts of Fig. sims_birth's method at birth (dt = 0, with the progenitor's stellar mass at
  the tracked snapshot nearest the cohort's birth) and the same stars at every later tracked snapshot (TRACK_SNAPS) in the
  progenitor's disc frame (lmcold.sims.disc_frame), from the first snapshot after the whole cohort formed;
- coherence (panel c): orbital-plane coherence C (lmcold.sims.coherence) of the stars formed in the WINDOW before each snapshot
  within 3 kpc, followed (ParticleIDs, within 10 kpc, disc frame) at every later snapshot up to 0.5 Gyr (FULL sample, FULL_SNAPS,
  z = 4-1.5); late control from the every-third-snapshot cutouts at LATE_SNAPS (z ~ 0.05-0.5).
Reads the SubLink trees and the star, gas and full cutouts of the progenitors (scripts/fetch_tng50_progenitors.py) and the TNG50
birth files of the z = 0 analogues (scripts/fetch_tng50.py).
Columns: gas: halo, snap, z, logMs [log10 Msun], n, vphi, sz, cs [km/s]; history: halo, age [Gyr], dt [Gyr since birth], vphi, sz
[km/s], logMs (birth rows only); coherence: halo, set ('full' or 'track'), s_birth, n0, dt [Gyr since formation], C.

Run: .venv/bin/python scripts/manifest.py run build_sims_heating.py   (after scripts/build_sims_birth.py; ~15 min; 48 s in the
recorded run of docs/run_times.txt)
"""
# %% [markdown]
# ### Settings
# The progenitor samples and snapshot lists come from `scripts/fetch_tng50_progenitors.py`, the cohort width from
# `scripts/build_sims_birth.py`.

# %%
from multiprocessing import Pool

import numpy as np
import pandas as pd

from fetch_tng50_progenitors import FULL, FULL_SNAPS, GAS_SNAPS, SAMPLE, TRACK_SNAPS
from lmcold import sims
from lmcold.sources import ROOT
from build_sims_birth import COHORT

WINDOW = 0.1  # Gyr, formation window of the cohorts of panel c
LATE_SNAPS = tuple(range(75, 100, 3))   # snapshots of the late control cohorts (every third, z ~ 0.5 to 0.05)
WORKERS = 4   # parallel processes, one progenitor each


# %% [markdown]
# ### Helper
# `kin` is the mass-weighted rotation and vertical dispersion of a set of particles in a frame with the spin along $z$.

# %%
def kin(X, V, m):
    """Mass-weighted mean v_phi and sigma_z [km/s] in a frame with the spin along +z (particles at R > 0.1 kpc).

    X [kpc], V [km/s]: (N, 3); m: (N,) [Msun]."""
    R = np.hypot(X[:, 0], X[:, 1])
    # Particles on the axis (R <= 0.1 kpc) are dropped (v_phi below divides by R).
    k = R > 0.1
    # v_phi = (x v_y - y v_x) / R [km/s]; v_z about its mass-weighted mean, so sigma_z is the mass-weighted rms.
    vphi = (X[k, 0] * V[k, 1] - X[k, 1] * V[k, 0]) / R[k]
    vz = V[k, 2] - np.average(V[k, 2], weights=m[k])
    return np.average(vphi, weights=m[k]), np.sqrt(np.average(vz ** 2, weights=m[k]))


# %% [markdown]
# ### Panel (a): the star-forming gas
# For each progenitor and gas snapshot: the star-forming gas within 3 kpc defines the spin axis and the mean velocity; then its
# rotation, vertical dispersion and sound speed at 0.5 < $R$ < 3 kpc.

# %%
def gas_rows(name):
    """Star-forming gas kinematics of one progenitor at GAS_SNAPS (panel a)."""
    # b: the SubLink main branch (SnapNum, Mstar [Msun], ...) of this progenitor.
    b, rows = sims.main_branch(name), []
    for s in GAS_SNAPS:
        p = sims.progenitor(name, s, 'gas', b)
        if p is None or 'gas' not in p:
            continue
        g = p['gas']   # x [kpc], v [km/s], m [Msun], sfr, u (internal energy per unit mass [(km/s)^2])
        # Star-forming gas within 3 kpc (at least 50 cells) defines the velocity origin and the spin axis.
        sf = (g['sfr'] > 0) & (np.linalg.norm(g['x'], axis=1) < 3)
        if sf.sum() < 50:
            continue
        v = g['v'] - np.average(g['v'][sf], axis=0, weights=g['m'][sf])
        # Total angular momentum of that gas; ez its direction; ex, ey complete a right-handed frame (rows of rot).
        L = (g['m'][sf, None] * np.cross(g['x'][sf], v[sf])).sum(0)
        ez = L / np.linalg.norm(L)
        ex = np.cross([0., 0., 1.], ez); ex /= np.linalg.norm(ex)
        rot = np.stack([ex, np.cross(ez, ex), ez])
        # All gas in that frame; the measured cells: star-forming, 0.5 < R < 3 kpc and |z| < 3 kpc (at least 50).
        X, V = g['x'] @ rot.T, v @ rot.T
        R = np.hypot(X[:, 0], X[:, 1])
        k = (g['sfr'] > 0) & (R > 0.5) & (R < 3) & (np.abs(X[:, 2]) < 3)
        if k.sum() < 50:
            continue
        vphi, sz = kin(X[k], V[k], g['m'][k])
        cs = np.sqrt(np.average(5 / 9 * 2 * g['u'][k], weights=g['m'][k]))  # gamma (gamma - 1) u, gamma = 5/3
        rows.append(dict(halo=name, snap=s, z=1 / p['a'] - 1, logMs=np.log10(b['Mstar'][b['SnapNum'] == s][0]), n=int(k.sum()),
                         vphi=vphi, sz=sz, cs=cs))
    return rows


# %% [markdown]
# ### Panels (a) and (b): the cohorts at birth and later
# The birth values of the 0.1-Gyr cohorts as in `scripts/build_sims_birth.py`, with the progenitor's stellar mass at the time; then
# the same stars, found by particle number, at every later tracked snapshot in the progenitor's disc frame.

# %%
def history_rows(name):
    """Panel (a) birth points with the progenitor's stellar mass, and the panel (b) histories, of one analogue."""
    b = sims.main_branch(name)
    d = sims.birth('TNG50', name)
    ids = sims.progenitor(name, 99, 'track', b)['stars']['ids']  # same particles and order as sims.birth (tests/test_sims.py)
    use = np.linalg.norm(d['x'], axis=1) < 10   # kpc, stars within 10 kpc today, as build_sims_birth.R_TODAY
    # The progenitor's centre and velocity against time, from the birth positions and velocities of these stars.
    track = sims.progenitor_track(d['birth_x'][use], d['a_form'][use], d['birth_v'][use])
    snaps = {s: p for s in TRACK_SNAPS if (p := sims.progenitor(name, s, 'track', b))}
    lb = {s: p['lookback'] for s, p in snaps.items()}   # Gyr, lookback time of each tracked snapshot
    frames = {}   # snapshot: (particle ids, disc-frame X [kpc], V [km/s], m [Msun]) of all the progenitor's stars
    # Each tracked snapshot in the progenitor's disc frame.
    for s, p in snaps.items():
        st = p['stars']
        X, V, _ = sims.disc_frame(st['x'], st['v'], st['m'])
        frames[s] = (st['ids'], X, V, st['m'])
    rows = []
    # Cohorts of width COHORT [Gyr] in age, from 0 to 13.5 Gyr, with at least 100 stars of which 50 have birth phase-space
    # coordinates.
    for lo in np.arange(0, 13.5, COHORT):
        i = np.flatnonzero(use & (d['age'] >= lo) & (d['age'] < lo + COHORT))
        if len(i) < 100:
            continue
        # P, B: birth positions [kpc] and velocities [km/s] in the progenitor's frame at birth; k: stars where they are defined.
        P, B, k = sims.birth_phase_space(d['birth_x'][i], d['a_form'][i], d['birth_v'][i], d['m'][i], track)
        if k.sum() < 50:
            continue
        age = lo + COHORT / 2
        vphi, sz = kin(P[k], B[k], d['m'][i][k])
        s0 = min(lb, key=lambda s: abs(lb[s] - age))   # tracked snapshot nearest the cohort's birth
        rows.append(dict(halo=name, age=age, dt=0.0, vphi=vphi, sz=sz, logMs=np.log10(b['Mstar'][b['SnapNum'] == s0][0])))
        # The same stars at every tracked snapshot after the cohort formed (at least 30 found by particle id).
        cid = ids[i[k]]
        for s, (sid, X, V, m) in frames.items():
            if lb[s] > lo:  # the cohort is not fully formed yet
                continue
            w = np.isin(sid, cid)
            if w.sum() >= 30:
                vphi, sz = kin(X[w], V[w], m[w])
                rows.append(dict(halo=name, age=age, dt=age - lb[s], vphi=vphi, sz=sz))
    return rows


# %% [markdown]
# ### Panel (c): the coherence of the orbital planes
# The stars formed in the `WINDOW` before a snapshot within 3 kpc, followed by particle number at every later snapshot (within 10
# kpc), and their coherence $C=|\sum m\,\mathbf{j}|/\sum m\,|\mathbf{j}|$ at each.

# %%
def coherence_rows(job):
    """Panel (c): C of 100-Myr cohorts of one progenitor (job = name, cutout kind, snapshots, dtmax), at every later snapshot up to
    dtmax [Gyr] (late control: 0.9, so that the next every-third snapshot, 0.45-0.55 Gyr later at low z, is included)."""
    name, kind, snaps, dtmax = job
    b = sims.main_branch(name)
    S = []   # (snapshot, lookback [Gyr], stars, disc-frame X, V) of each snapshot with stars
    # Every snapshot with stars in the progenitor's disc frame.
    for s in snaps:
        p = sims.progenitor(name, s, kind, b)
        if p and 'stars' in p:
            st = p['stars']
            X, V, _ = sims.disc_frame(st['x'], st['v'], st['m'])
            S.append((s, p['lookback'], st, X, V))
    rows = []
    for i, (s, lb, st, X, V) in enumerate(S):
        k = (st['age_form'] - lb >= 0) & (st['age_form'] - lb < WINDOW) & (np.linalg.norm(st['x'], axis=1) < 3)   # formed in the WINDOW before s
        # At least 30 stars in the cohort.
        if k.sum() < 30:
            continue
        cid, t_form = st['ids'][k], np.average(st['age_form'][k], weights=st['m'][k])   # ids; mean lookback of formation [Gyr]
        for s1, lb1, st1, X1, V1 in S[i:]:
            if t_form - lb1 > dtmax + 1e-6:
                break
            # The cohort's stars within 10 kpc at a later snapshot (at least 20); dt [Gyr] since the mean formation time.
            w = np.isin(st1['ids'], cid) & (np.linalg.norm(st1['x'], axis=1) < 10)
            if w.sum() >= 20:
                rows.append(dict(halo=name, set=kind, s_birth=s, n0=int(k.sum()), dt=t_form - lb1, C=sims.coherence(X1[w], V1[w], st1['m'][w])))
    return rows


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data and build the three tables
    # Each function reads its progenitor's cutouts itself; the progenitors are shared among `WORKERS` processes. The coherence is
    # measured for the early cohorts (every snapshot, up to 0.5 Gyr) and the late control cohorts (every third snapshot, up to 0.9 Gyr).

    # %%
    with Pool(WORKERS) as pool:
        # Panels (a) and (b) on the SAMPLE of 60; panel (c) on the FULL sample of 20, early (up to 0.5 Gyr) and late control (up to
        # 0.9 Gyr).
        gas = pd.DataFrame([r for rows in pool.map(gas_rows, SAMPLE) for r in rows])
        hist = pd.DataFrame([r for rows in pool.map(history_rows, SAMPLE) for r in rows])
        coh = pd.DataFrame([r for rows in pool.map(coherence_rows, [(n, 'full', FULL_SNAPS, 0.5) for n in FULL] +
                                                    [(n, 'track', LATE_SNAPS, 0.9) for n in FULL]) for r in rows])
    # The three tables of the module docstring.
    gas.to_csv(ROOT / 'results/sims_heating_gas.csv', index=False, float_format='%.4f')
    hist.to_csv(ROOT / 'results/sims_heating_history.csv', index=False, float_format='%.4f')
    coh.to_csv(ROOT / 'results/sims_heating_coherence.csv', index=False, float_format='%.4f')
