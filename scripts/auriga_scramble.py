"""The experiment of Fig. letter_scramble (a) on the five Auriga level-3 LMC analogues (Letter, Section 3):
v_phi/sigma_z of each early cohort's stars, followed from birth, for the real stars and for test particles in the gravity of the
progenitor frozen at the cohort's birth snapshot, with parts of that gravity removed. Built on the machinery of
scripts/sims_scramble.py; an earlier version of the experiment measured the orbital-plane coherence C instead.

Galaxies: lmcold.sims.AURIGA (Au2, Au8, Au9, Au10, Au11; Grand et al. 2024). Birth snapshots: the full snapshots SNAPS (z = 3.6,
2.9, 2.35, 1.65), the only early outputs with gas and dark matter; the real stars are followed through the star-only high-cadence
outputs OUTS (every 20th, ~0.1 Gyr apart). As the full snapshots are 0.5-0.6 Gyr apart, the gravity cannot be interpolated in time
over the DTMAX window, so all test-particle variants are frozen at the birth snapshot (in TNG50 the frozen gravity removes the
rotation as fast as the real stars, Fig. letter_scramble a).
Progenitor: the star particles within R_HOST of the analogue at z = 0 (subhalo 0 of groups_251), followed back by ParticleIDs; centre a
shrinking sphere started at their median position, refined on all stars; velocity origin the mass-weighted mean of the stars within
R_VEL of it (sims_scramble.R_VEL). Units: physical kpc (Coordinates x 1e3 a / h), km/s (Velocities x sqrt(a)), Msun (1e10 / h).
Cohorts (as sims_scramble.py): the stars formed in the WINDOW before a birth snapshot within R_COHORT, of at least MIN_STARS; axes
rotated so that z is the cohort's spin at birth, fixed. The stars followed in v_phi/sigma_z: those at r < R_SUP[1] and cylindrical
R > R_SUP[0] at the birth snapshot (sims_scramble.in_ring), chosen once, by ID, wherever they go.
Gravity (sims_scramble.expansion: AGAMA Multipole, lmax 6, 40 radial nodes 0.05-60 kpc, no symmetry) of the particles within 60 kpc
at the birth snapshot, dark matter (types 1-3) and baryons (gas + stars) expanded separately, each potential with the uniform
frame-acceleration term of sims_scramble.accel. Variants, as the frozen versions of Fig. letter_scramble (a):
  E_split: all of it (TNG50 'E', frozen gravity); G_DM_fr: baryons kept to l = 0, so only the dark matter is aspherical (TNG50
  'G_DM', evolving there); D0_fr: m = 0 of both, axisymmetric about the cohort's spin (TNG50 'D0', evolving there).
Measured, about the cohort's spin at birth (the integration frame's z): v_phi/sigma_z (sims_scramble.support) of the followed stars,
real (vs_meas, at the high-cadence outputs within DTMAX) and test particles (vs_sim, at the same times); and the coherence C of all
cohort members within R_MEMBER (C_meas, C_sim), to compare with the earlier coherence measurement.
Writes results/auriga_scramble.csv (halo, s_birth, z, n0, n_sel, misalign, variant, t [Gyr since the birth snapshot], vs_meas, vs_sim,
C_meas, C_sim).
results/auriga_scramble.csv gives the Auriga numbers of the Letter's Fig. 4 (a) experiment in the text (scripts/paper_text_sims.py).
Reads the Auriga level-3 snapshots, group catalogues and star-only high-cadence outputs (scripts/fetch_auriga.py).
Run: .venv/bin/python scripts/manifest.py run auriga_scramble.py   (AGAMA; WORKERS processes, ~4 GB per full snapshot; 72 s in the
recorded run of docs/run_times.txt)"""
# %% [markdown]
# ### Settings
# The birth snapshots, the high-cadence outputs, the radius that identifies the progenitor's stars, and the three variants of the
# gravity (which terms of the dark-matter and of the baryon expansion are kept). The cohort and expansion settings are imported from
# `scripts/sims_scramble.py`.

# %%
import glob
import tempfile
from multiprocessing import Pool

import agama
import h5py
import numpy as np
import pandas as pd

from lmcold import sims
from lmcold.sources import ROOT
from sims_scramble import (ALL, KPC_KMS_GYR, L0, M0, MIN_STARS, R_COHORT, R_MEMBER, R_SUP, R_VEL, RMAX, WINDOW, DTMAX, accel,
                           expansion, frame, halo_axes, in_ring, keep_terms, spin, support)

SNAPS = (50, 60, 70, 90)   # full snapshots used as birth snapshots
OUTS = range(280, 1201, 20)   # star-only high-cadence outputs (snapdir_highfreqstars_*), ~0.1 Gyr apart
R_HOST = 10.0   # kpc: the analogue's stars at z = 0 that identify its progenitor
VARIANTS = {'E_split': (ALL, ALL), 'G_DM_fr': (ALL, L0), 'D0_fr': (M0, M0)}   # (terms kept of dark matter, of baryons)
WORKERS = 2   # parallel processes (one cohort each; ~4 GB of memory per full snapshot)


# %% [markdown]
# ### Reading Auriga: the progenitor and its centre
# `host_ids` takes the analogue's stars at $z=0$; `centre` and `local` put each snapshot in the frame of the progenitor those stars
# trace; `read_full` and `read_stars` read a full snapshot (gas, dark matter, stars) and a star-only output in that frame.

# %%
def host_ids(d):
    """IDs of the star particles within R_HOST of the analogue at z = 0 (subhalo 0 of groups_251)."""
    # Star IDs and positions [comoving kpc/h x 1e-3] from every file of the z = 0 snapshot 251.
    ids, x = [], []
    for fn in sorted(glob.glob(str(d / 'snapdir_251/snapshot_251.*.hdf5'))):
        with h5py.File(fn) as f:
            h = f['Header'].attrs['HubbleParam']
            if 'PartType4' in f:
                ids.append(f['PartType4/ParticleIDs'][:]); x.append(f['PartType4/Coordinates'][:])
    # Centre of subhalo 0 (the analogue) in the same comoving units.
    with h5py.File(sorted(glob.glob(str(d / 'groups_251/fof_subhalo_tab_251.*.hdf5')))[0]) as f:
        c = f['Subhalo/SubhaloPos'][0]
    ids, x = np.concatenate(ids), np.concatenate(x)
    # Distance converted to physical kpc at z = 0 (a = 1): x 1e3 / h.
    return ids[np.linalg.norm(x - c, axis=1) * 1e3 / h < R_HOST]


def centre(ids, x, m, host):
    """Shrinking-sphere centre [kpc] started at the median position of the progenitor's stars, refined on all stars."""
    def shrink(c, x, r):
        # Iterate: mean position of the stars inside r, then shrink r by 20 per cent, down to 0.5 kpc or fewer than 30 stars.
        while r > 0.5:
            k = np.linalg.norm(x - c, axis=1) < r
            if k.sum() < 30:
                break
            c, r = x[k].mean(0), 0.8 * r
        return c
    # First pass on the progenitor's stars from 50 kpc, second pass on all stars from 20 kpc.
    k = np.isin(ids, host)
    return shrink(shrink(np.median(x[k], axis=0), x[k], 50.), x, 20.)


def local(st, parts, host):
    """Shift st (stars with ids) and the other parts to the progenitor centre and the velocity origin of the stars within R_VEL."""
    # Centre [kpc], then the mass-weighted mean velocity [km/s] of the stars within R_VEL of it, subtracted from every part.
    c = centre(st['ids'], st['x'], st['m'], host)
    w = np.linalg.norm(st['x'] - c, axis=1) < R_VEL
    v0 = np.average(st['v'][w], axis=0, weights=st['m'][w])
    for p in [st] + parts:
        p['x'], p['v'] = p['x'] - c, p['v'] - v0


def read_full(d, s, host):
    """Full snapshot s: gas, dm (types 1-3) and st (stars with ids and age [Gyr]) about the progenitor; a, lookback [Gyr]."""
    # Collect (x, v, m[, ids, formation scale factor]) per part from every file of the snapshot; types 1-3 are dark matter.
    P = {p: [] for p in ('gas', 'dm', 'st')}
    for fn in sorted(glob.glob(str(d / f'snapdir_{s:03d}/snapshot_{s:03d}.*.hdf5'))):
        with h5py.File(fn) as f:
            hdr, mt = dict(f['Header'].attrs), f['Header'].attrs['MassTable']
            for t, p in (('PartType0', 'gas'), ('PartType1', 'dm'), ('PartType2', 'dm'), ('PartType3', 'dm'), ('PartType4', 'st')):
                if t in f:
                    g = f[t]
                    x = g['Coordinates'][:]
                    # particles without a Masses field take the common mass of their type from the header MassTable
                    m = g['Masses'][:] if 'Masses' in g else np.full(len(x), mt[int(t[-1])])
                    extra = (g['ParticleIDs'][:], g['GFM_StellarFormationTime'][:]) if p == 'st' else ()
                    P[p].append((x, g['Velocities'][:], m) + extra)
    h, a, om = hdr['HubbleParam'], hdr['Time'], hdr['Omega0']
    # Lookback time of the snapshot [Gyr]; then each part converted to physical kpc, km/s and Msun (see the module docstring).
    lb = float(sims.lookback(a, h, om))
    out = dict(a=a, lb=lb)
    for p, L in P.items():
        c = [np.concatenate(q) for q in zip(*L)]
        out[p] = dict(x=c[0] * 1e3 * a / h, v=c[1] * np.sqrt(a), m=c[2] * 1e10 / h)
        if p == 'st':
            ok = c[4] > 0   # wind particles have formation time <= 0
            out[p] = {k: v[ok] for k, v in out[p].items()} | dict(ids=c[3][ok], age=sims.lookback(c[4][ok], h, om) - lb)
    # Move everything to the progenitor's centre and velocity origin.
    local(out['st'], [out['gas'], out['dm']], host)
    return out


def read_stars(d, i, host):
    """Star-only output i: stars (ids, x, v, m) about the progenitor and lookback [Gyr]. Heavy low-resolution particles dropped."""
    # Star particles of every file of the star-only output: IDs, formation scale factor, mass, position, velocity (code units).
    L = []
    for fn in sorted(glob.glob(str(d / f'snapdir_highfreqstars_{i:03d}/snapshot_highfreqstars_{i:03d}.*.hdf5'))):
        with h5py.File(fn) as f:
            hdr = dict(f['Header'].attrs)
            if 'PartType4' in f:
                g = f['PartType4']
                L.append((g['ParticleIDs'][:], g['GFM_StellarFormationTime'][:], g['Masses'][:], g['Coordinates'][:], g['Velocities'][:]))
    ids, af, m, x, v = (np.concatenate(q) for q in zip(*L))
    # Real stars (formation time > 0, not wind particles) lighter than 100x the median mass (the heavy low-resolution particles).
    ok = (af > 0) & (m < 100 * np.median(m))
    h, a, om = hdr['HubbleParam'], hdr['Time'], hdr['Omega0']
    # Physical units: Msun, kpc, km/s; then centred on the progenitor.
    st = dict(ids=ids[ok], m=m[ok] * 1e10 / h, x=x[ok] * 1e3 * a / h, v=v[ok] * np.sqrt(a))
    local(st, [], host)
    return st, float(sims.lookback(a, h, om))


# %% [markdown]
# ### Gravity and one cohort
# `split` separates the dark matter and the baryons for their expansions; `cohort` follows one cohort's real stars through the
# high-cadence outputs, builds the three frozen variants of the gravity at the birth snapshot, integrates the stars as test particles
# in each and measures $v_\phi/\sigma_z$ and $C$ at the times of the outputs.

# %%
def split(F, R):
    """Dark matter and baryons within RMAX in the frame R (rows: axes): ((X, M), (X, M))."""
    out = []
    # Dark matter first, then gas + stars; positions rotated into frame R, kept within RMAX (60 kpc) of the centre.
    for parts in (('dm',), ('gas', 'st')):
        X = np.concatenate([F[p]['x'] for p in parts]) @ R.T
        M = np.concatenate([F[p]['m'] for p in parts])
        k = np.linalg.norm(X, axis=1) < RMAX
        out.append((X[k], M[k]))
    return out


def cohort(name, s, window=WINDOW):
    """Rows of results/auriga_scramble.csv for the cohort born before full snapshot s of analogue name (window: the cohort's age
    range [Gyr]; the earlier measurement used 0.1)."""
    d = sims.AURIGA[name]
    # The progenitor's star IDs and the birth snapshot in its frame.
    host = host_ids(d)
    F = read_full(d, s, host)
    st = F['st']
    # The cohort: stars younger than window [Gyr] at the birth snapshot, within R_COHORT (3 kpc); none if fewer than MIN_STARS (30).
    k = (st['age'] >= 0) & (st['age'] < window) & (np.linalg.norm(st['x'], axis=1) < R_COHORT)
    if k.sum() < MIN_STARS:
        return []
    xs, vs, ms, ids = st['x'][k], st['v'][k], st['m'][k], st['ids'][k]   # the cohort at its birth snapshot [kpc, km/s, Msun]
    n = spin(xs, vs, ms)   # its spin at birth: the z axis of the integration frame
    # Rotation matrix (rows: axes) whose z axis is n; sel: the followed stars of the ring R_SUP (boolean, (N,)).
    R = frame(n)
    sel = in_ring(xs, vs, ms)
    # Columns common to every row: redshift, numbers of cohort and followed stars, angle [deg] between the spin and the dark-matter
    # halo's minor axis (third row of halo_axes).
    meta = dict(halo=name, s_birth=s, z=1 / F['a'] - 1, n0=int(k.sum()), n_sel=int(sel.sum()),
                misalign=float(np.degrees(np.arccos(min(1., abs(halo_axes(F['dm']['x'])[2] @ n))))))
    # First row: the real cohort at t = 0 (v_phi/sigma_z of the followed stars, coherence C of all members).
    rows = [dict(meta, variant='real', t=0.0, vs_meas=support(xs[sel], vs[sel], ms[sel], n), C_meas=sims.coherence(xs, vs, ms))]
    # The real stars at each high-cadence output within DTMAX of the birth snapshot.
    t_out = []
    for i in OUTS:
        with h5py.File(sorted(glob.glob(str(d / f'snapdir_highfreqstars_{i:03d}/*.hdf5')))[0]) as f:
            hd = dict(f['Header'].attrs)
        # Time of this output after the birth snapshot [Gyr], from the header only; outputs outside (0, DTMAX] are skipped.
        dt = F['lb'] - sims.lookback(hd['Time'], hd['HubbleParam'], hd['Omega0'])
        if not 0 < dt <= DTMAX + 1e-6:
            continue
        S, _ = read_stars(d, i, host)
        # The same followed stars (by ID) and the cohort members within R_MEMBER (10 kpc) at this output.
        f_sel = np.isin(S['ids'], ids[sel])
        f_mem = np.isin(S['ids'], ids) & (np.linalg.norm(S['x'], axis=1) < R_MEMBER)
        rows.append(dict(meta, variant='real', t=dt, vs_meas=support(S['x'][f_sel], S['v'][f_sel], S['m'][f_sel], n),
                         C_meas=sims.coherence(S['x'][f_mem], S['v'][f_mem], S['m'][f_mem])))
        t_out.append(dt)
    if not t_out:
        return rows
    t = np.r_[0.0, t_out] / KPC_KMS_GYR   # output times since the birth snapshot, AGAMA units
    # 500 random stars within 2 kpc: the points on which the frame acceleration is averaged.
    inner = st['x'][np.linalg.norm(st['x'], axis=1) < 2] @ R.T
    inner = inner[np.random.default_rng(0).choice(len(inner), min(500, len(inner)), replace=False)]
    (Xd, Md), (Xb, Mb) = split(F, R)
    x, v = xs @ R.T, vs @ R.T   # initial conditions in the integration frame
    # The three variants, all frozen at the birth snapshot; orbits, then v_phi/sigma_z and C of the test particles at each time.
    with tempfile.TemporaryDirectory() as tmp:
        fd, fb = expansion(Xd, Md, f'{tmp}/dm.ini'), expansion(Xb, Mb, f'{tmp}/bar.ini')
        for var, (kd, kb) in VARIANTS.items():
            # Dark-matter and baryon potentials with the terms of this variant; the frame acceleration of their sum is added
            # (accel).
            pd_ = agama.Potential(keep_terms(fd, kd, var) if kd is not ALL else fd)
            pb_ = agama.Potential(keep_terms(fb, kb, var) if kb is not ALL else fb)
            both = agama.Potential(pd_, pb_)
            pot = agama.Potential(pd_, pb_, accel([both, both], [inner, inner], t[[0, -1]]))
            orbs = agama.orbit(potential=pot, ic=np.column_stack([x, v]), time=t[-1] * (1 + 1e-6), dtype=object,
                               verbose=False)   # a little past the last time, where the interpolated trajectory is NaN
            # Phase-space positions at the output times, shape (n_times, N, 6); the integration frame's z is the spin, hence ez =
            # (0, 0, 1).
            w6 = np.stack([o(t) for o in orbs], axis=1)
            for tk, q in zip(t, w6):
                mem = np.linalg.norm(q[:, :3], axis=1) < R_MEMBER
                rows.append(dict(meta, variant=var, t=tk * KPC_KMS_GYR, vs_sim=support(q[sel, :3], q[sel, 3:], ms[sel], np.array([0., 0., 1.])),
                                 C_sim=sims.coherence(q[mem, :3], q[mem, 3:], ms[mem])))
    print(name, s, 'done', meta['n0'], meta['n_sel'], flush=True)
    return rows


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data and run every cohort
    # One job per analogue and birth snapshot; each reads its own snapshots inside `cohort`.

    # %%
    with Pool(WORKERS) as pool:
        # Five analogues x four birth snapshots; all rows of all cohorts in one table.
        t = pd.DataFrame([r for rows in pool.starmap(cohort, [(h, s) for h in sims.AURIGA for s in SNAPS]) for r in rows])
    t.to_csv(ROOT / 'results/auriga_scramble.csv', index=False, float_format='%.5f')
    print(f'{t.halo.nunique()} galaxies, {t.groupby("halo").s_birth.nunique().sum()} cohorts')
