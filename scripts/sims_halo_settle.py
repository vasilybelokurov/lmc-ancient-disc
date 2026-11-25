"""How the inner dark-matter halo of the TNG50 progenitors lines up with the stellar disc and becomes rounder (Section 6, Fig. sims_settle),
and how much coherence the 100-Myr cohorts born at each stage lose.

Progenitors: the 60 SAMPLE main progenitors of scripts/fetch_tng50_progenitors.py at snapshots 21-51 (z = 4-1): every snapshot
21-40 for the 20 FULL ones (stars and dark matter from the full cutouts), elsewhere every third (dark matter from sims.TNG_DM_DIR,
stars from the track cutouts). Positions about the main branch's SubhaloPos (lmcold.sims.progenitor); dark matter only below.
- Disc frame: z along the angular momentum of the stars within R_IN = 3 kpc, velocities relative to their mass-weighted mean.
- Shape (eq. settle_inertia): second-moment tensor of the dark-matter particles within R_IN (a sphere, not iterated); b/a, c/a from
  its eigenvalues; misalignment psi = angle of its minor axis to the disc axis [deg, 0-90] (eq. settle_psi).
- Quadrupole (eq. settle_eps): AGAMA Multipole expansion of the dark matter within 30 kpc (lmax 4, no symmetry, 40 radial nodes
  0.2-30 kpc) in the disc frame; at R0 = 2 kpc, eps_tilt = |Phi_2,+-1| / |Phi_00|, eps_elong = |Phi_2,+-2| / |Phi_00|, eps2 their
  quadrature sum.
- Cohorts (Fig. sims_settle dotted lines and Fig. sims_scramble c): the stars formed in the 100 Myr before each track snapshot
  21-51 within 3 kpc (scripts/build_sims_heating.coherence_rows), C at that snapshot (C0) and at the next track snapshot (C1,
  0.40-0.50 Gyr later); loss = C0 - C1; joined to the halo state at the birth snapshot.
Writes results/sims_halo_settle.csv (halo, snap, z, lookback, eps_tilt, eps_elong, eps2, misalign, b_a, c_a, n_in) and
results/sims_halo_cohorts.csv (halo, s_birth, n0, C0, C1, step, loss and the halo state at s_birth).
results/sims_halo_settle.csv is read by scripts/paper_sims_settle.py, paper_sims_sidm.py (the TNG50 band of Fig. sims_sidm a) and
paper_letter_scramble.py (the Letter's Fig. 4 c); results/sims_halo_cohorts.csv by scripts/paper_sims_scramble.py (Fig.
sims_scramble c) and paper_sims_settle.py; both by the text scripts (scripts/paper_text_sims.py).
Reads the SubLink trees and the track, full and dark-matter cutouts of the progenitors (scripts/fetch_tng50_progenitors.py).
Units: positions in physical kpc, eps_* dimensionless (ratios of potential coefficients), misalign in deg, lookback in Gyr.
Run: .venv/bin/python scripts/manifest.py run sims_halo_settle.py   (AGAMA; ~4 processes; 14 s in the recorded run of
docs/run_times.txt)"""
# %% [markdown]
# ### Settings
# The radius of the quadrupole (`R0`), of the disc axis and the shape (`R_IN`) and of the expansion (`R_POT`), the expansion's order
# and radial grid, and the snapshots.

# %%
import tempfile
from multiprocessing import Pool

import agama
import numpy as np
import pandas as pd

from fetch_tng50_progenitors import SAMPLE
from lmcold import sims
from lmcold.sources import ROOT
from build_sims_heating import coherence_rows

R0, R_IN, R_POT = 2.0, 3.0, 30.0   # kpc: eps radius, disc-axis and shape sphere, outer radius of the expansion
LMAX, GRID, RMIN = 4, 40, 0.2   # Multipole: order, radial nodes, inner node [kpc] (outer: R_POT)
SNAPS = range(21, 52)   # TNG50 snapshots 21-51 (z = 4-1), the range of the module docstring
# AGAMA units: mass [Msun], length [kpc], velocity [km/s]; potentials then come out in (km/s)^2
agama.setUnits(mass=1, length=1, velocity=1)


# %% [markdown]
# ### Helpers
# `disc_axes` gives the frame of the stellar disc; `coeffs` reads the coefficients $\Phi_{lm}$ of an expansion at one radius.

# %%
def disc_axes(st):
    """Rows (x, y, z) of the frame with z along the angular momentum of the stars within R_IN, x along z_sim x z."""
    # Stars within R_IN [kpc] of the centre; their velocities relative to the mass-weighted mean [km/s]
    k = np.linalg.norm(st['x'], axis=1) < R_IN
    v = st['v'][k] - np.average(st['v'][k], axis=0, weights=st['m'][k])
    # Total angular momentum (3,) [Msun kpc km/s] and its unit vector, the disc axis
    L = (st['m'][k, None] * np.cross(st['x'][k], v)).sum(0)
    ez = L / np.linalg.norm(L)
    # x in the disc plane along z_sim x z; y completes the right-handed frame; returned as a (3, 3) rotation matrix
    ex = np.cross([0., 0., 1.], ez); ex /= np.linalg.norm(ex)
    return np.stack([ex, np.cross(ez, ex), ez])


def coeffs(pot, r):
    """Phi_lm of an AGAMA Multipole potential at radius r, from its exported table (columns 'l=0,m=0' ...).

    Returns {'l=..,m=..': value [(km/s)^2]}, interpolated linearly in ln r between the table's radial nodes."""
    # AGAMA has no direct accessor for the coefficients: export the potential to a text file and read it back
    with tempfile.NamedTemporaryFile(suffix='.ini') as f:
        pot.export(f.name)
        lines = open(f.name).read().split('\n')
    # The '#Phi' section: a header line of column names (radius, then 'l=..,m=..'), then one row per radial node
    i = lines.index('#Phi') + 1
    head = lines[i].lstrip('#').split('\t')
    # Read the numeric rows until the section ends (blank line, comment or the next '[...]' section)
    rows = []
    for ln in lines[i + 1:]:
        if not ln.strip() or ln.startswith('#') or ln.startswith('['):
            break
        rows.append([float(q) for q in ln.split()])
    # T: (n_nodes, 1 + n_coeffs); column 0 is the radius [kpc]
    T = np.array(rows)
    return {c: np.interp(np.log(r), np.log(T[:, 0]), T[:, j + 1]) for j, c in enumerate(head[1:])}


# %% [markdown]
# ### The halo of one progenitor at every snapshot
# In the frame of the stellar disc: the quadrupole of the dark matter's expansion at `R0` (tilt and elongation parts) and the axis
# ratios and misalignment of the dark matter within `R_IN`.

# %%
def halo_rows(name):
    """Rows of results/sims_halo_settle.csv for progenitor name: one per snapshot of SNAPS with dark matter and stars."""
    b = sims.main_branch(name)   # main branch of the SubLink tree (centres and velocities per snapshot)
    out = []
    for s in SNAPS:
        # Full cutout if it exists (stars + dark matter); otherwise dark matter and stars from the separate cutouts
        full = sims.progenitor(name, s, 'full', b)
        p, st = (full, full) if full else (sims.progenitor(name, s, 'dm', b), sims.progenitor(name, s, 'track', b))
        if not (p and st and 'dm' in p and 'stars' in st):
            continue
        # Dark-matter positions (N, 3) [kpc] rotated into the disc frame; r their radii [kpc]
        X = p['dm']['x'] @ disc_axes(st['stars']).T
        r = np.linalg.norm(X, axis=1)
        k = (r < R_POT) & (r > 1e-4)   # SubhaloPos is the most bound particle, often a DM one: the expansion needs r > 0
        # Multipole expansion of the dark matter within R_POT (settings of the Settings cell)
        pot = agama.Potential(type='Multipole', particles=(X[k], p['dm']['m'][k]), lmax=LMAX, symmetry='none', rmin=RMIN, rmax=R_POT,
                              gridSizeR=GRID)
        c = coeffs(pot, R0)   # Phi_lm at R0 [(km/s)^2]
        p00 = abs(c['l=0,m=0'])   # the monopole, the spherical average of the potential at R0
        # eps_tilt from the m = +-1 and eps_elong from the m = +-2 quadrupole terms (eq. settle_eps)
        tilt, elong = np.hypot(c['l=2,m=-1'], c['l=2,m=1']) / p00, np.hypot(c['l=2,m=-2'], c['l=2,m=2']) / p00
        w = r < R_IN   # dark matter within the sphere of the shape tensor
        val, vec = np.linalg.eigh((X[w, :, None] * X[w, None, :]).sum(0) / w.sum())   # ascending: minor axis first
        cc, bb, aa = np.sqrt(val)   # minor, intermediate, major axis lengths [kpc] (rms extents)
        # One row: redshift from the scale factor a; misalign = angle [deg] of the minor axis (vec[:, 0]) to the disc axis z
        out.append(dict(halo=name, snap=s, z=1 / p['a'] - 1, lookback=p['lookback'], eps_tilt=tilt, eps_elong=elong,
                        eps2=np.hypot(tilt, elong), misalign=np.degrees(np.arccos(min(1., abs(vec[2, 0])))), b_a=bb / aa, c_a=cc / aa,
                        n_in=int(w.sum())))
    return out


# %% [markdown]
# ### The cohorts born at each stage
# The 100-Myr cohorts of `scripts/build_sims_heating.py` born at every third snapshot, with their coherence at birth (`C0`) and one
# tracked snapshot later (`C1`).

# %%
def cohort_rows(name):
    """C0, C1 and the time step [Gyr] between them of each cohort of progenitor name (columns of results/sims_halo_cohorts.csv)."""
    # Cohorts born at every third snapshot 21-54 of the track cutouts, followed for up to 0.6 Gyr after birth
    r = pd.DataFrame(coherence_rows((name, 'track', range(21, 55, 3), 0.6)))
    out = []
    for s, g in (r.groupby('s_birth') if len(r) else []):
        # Sort the measurements of one cohort by time since birth [Gyr]: C0 is the first, C1 the next tracked snapshot
        g = g.sort_values('dt')
        if len(g) >= 2:
            out.append(dict(halo=name, s_birth=s, n0=g.n0.iloc[0], C0=g.C.iloc[0], C1=g.C.iloc[1], step=g.dt.iloc[1] - g.dt.iloc[0]))
    return out


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data and measure every progenitor
    # Each function reads its progenitor's cutouts itself; the cohorts are then joined to the halo state at their birth snapshot.

    # %%
    # One progenitor per process, 4 processes; H: one row per (halo, snapshot), C: one row per cohort
    with Pool(4) as pool:
        H = pd.DataFrame([x for rows in pool.map(halo_rows, SAMPLE) for x in rows])
        C = pd.DataFrame([x for rows in pool.map(cohort_rows, SAMPLE) for x in rows])
    H.to_csv(ROOT / 'results/sims_halo_settle.csv', index=False, float_format='%.5f')
    # Attach the halo state at the birth snapshot to each cohort; loss = C0 - C1 (coherence lost in one step)
    C = C.merge(H.rename(columns={'snap': 's_birth'}), on=['halo', 's_birth'])
    C['loss'] = C.C0 - C.C1
    C.to_csv(ROOT / 'results/sims_halo_cohorts.csv', index=False, float_format='%.5f')
    print(f'{len(H)} halo measurements of {H.halo.nunique()} progenitors; {len(C)} cohorts')
