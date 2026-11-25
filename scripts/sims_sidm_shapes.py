"""Shape and potential quadrupole of the centre of an LMC-mass halo in CDM and self-interacting dark matter (Section 6.6, Fig. sims_sidm):
SIDM Concerto Halo104, CDM and GroupSIDM from the same initial conditions, dark matter only (scripts/fetch_concerto.py), at z = 4, 3,
2, 1.5 (snapshots 109, 126, 149, 163) and 0 (235).

Particles: types 1 and 2 (the two finest; masses from the header mass table) in physical kpc (comoving Mpc/h x a 1e3 / h); types 3-5
only flag contaminated halos. Host: the most massive Rockstar halo with upid = -1 and no type 3-5 particle within Rvir. Centre: its
densest core (densest(): of the five highest peaks of the smoothed particle counts within 0.5 Rvir, each refined by a shrinking
sphere 2 -> 0.2 kpc, the one with the most mass within 2 kpc; a shrinking sphere from Rvir lands between the cores during the z = 3
merger, and the single highest peak can be a subhalo's cusp when the SIDM host has a core).
Shapes: iterative reduced inertia tensor in ellipsoidal shells (shell_shape), NSHELL log-spaced shells from 0.4 kpc to Rvir; q = b/a,
s = c/a; N particles per shell (the figure uses N >= 10^4).
Quadrupole: AGAMA Multipole (lmax 4, no symmetry, 50 radial nodes 0.1 kpc - 2 Rvir) of the particles within 2 Rvir in the frame of the
minor axis of the 1.5-2.5 kpc shell; eps2 = sqrt(sum_m Phi_2m^2) / |Phi_00| and eps2_na without the m = 0 term (the part not
axisymmetric about that axis), at the shells' radii and at 2 kpc (row with N = 0), to compare with the TNG50 progenitors
(results/sims_halo_settle.csv).
Writes results/sims_sidm_shapes.csv (z, run, r [kpc], r_rvir, q, s, N, eps2, eps2_na) and results/sims_sidm_particle_mass.csv
(run, m_part [Msun]: the type-1 particle mass of each run at z = 0, from the snapshot header, for the Letter's text); prints the
hosts and their centres.
Read by scripts/paper_sims_sidm.py (Fig. sims_sidm, the Letter's Fig. 5) and the text scripts (scripts/paper_text_letter.py,
paper_text_sims.py). Reads the Concerto snapshots and Rockstar halo lists of scripts/fetch_concerto.py (the high-redshift snapshots
are available from the simulation authors on request).
Run: .venv/bin/python scripts/manifest.py run sims_sidm_shapes.py   (AGAMA; ~20 GB of memory at the z = 0 snapshot; 927 s in the
recorded run of docs/run_times.txt)"""
# %% [markdown]
# ### Settings
# The snapshots, the number of shells, the settings of the centre finder, the radii of the shells, of the frame and of the
# quadrupole, and the expansion's order and grid.

# %%
import tempfile

import agama
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

from fetch_concerto import HIGHZ, HIGHZ_DIR, RUNS, Z0_DIR
from lmcold.sources import ROOT

agama.setUnits(mass=1, length=1, velocity=1)
H = 0.7                                                       # Concerto's Hubble parameter
SNAPS = {**{s: float(a) for s, a in HIGHZ.items()}, 235: 1.0}   # snapshot -> scale factor a (z = 4, 3, 2, 1.5 and z = 0)
NSHELL = 6                                                      # number of log-spaced shells from R_SHELL_MIN to Rvir
CORE_BOX, N_PEAKS, CORE_SHRINK, CORE_R = 0.5, 5, (2.0, 0.2), 2.0   # densest(): peaks within CORE_BOX Rvir, shrinking sphere [kpc], mass within CORE_R kpc
R_SHELL_MIN, R_KEEP, FRAME_SHELL, R_EPS = 0.4, 2.0, (1.5, 2.5), 2.0   # kpc; particles within R_KEEP Rvir; frame shell [kpc]; eps2 radius [kpc]
LMAX, GRID, RMIN = 4, 50, 0.1   # Multipole: order, radial nodes, inner node [kpc] (outer: R_KEEP Rvir)


# %% [markdown]
# ### Reading the Concerto runs
# `read` reads the particles of one run and snapshot from the 16 Gadget files, `halos` its Rockstar halo list.

# %%
def path(snap):
    """Directory of snapshot snap: the z = 0 or the high-redshift download."""
    return Z0_DIR if snap == 235 else HIGHZ_DIR


def read(run, snap, a):
    """Positions [physical kpc] and masses [Msun] of types 1 + 2, and positions of types 3-5, over the 16 Gadget format-1 files."""
    hi, m, lo = [], [], []   # per file: type 1-2 positions, their masses, type 3-5 positions
    for k in range(16):
        with open(path(snap) / run / f'snapshot_{snap}.{k}', 'rb') as f:
            # Header record (256 bytes between 4-byte markers): six particle counts (int32) and six masses (float64, 1e10 Msun/h).
            f.read(4); h = f.read(256); f.read(4)
            n, mass = np.frombuffer(h[:24], 'i4'), np.frombuffer(h[24:72], 'f8')
            # Position record: 3 float32 per particle, all types in order [comoving Mpc/h], shape (sum n, 3).
            f.read(4); x = np.frombuffer(f.read(12 * n.sum()), 'f4').reshape(-1, 3)
        # Offsets of each type in the position block; types 1-2 with masses [Msun] from the mass table, types 3-5 positions only.
        c = np.r_[0, np.cumsum(n)]
        hi.append(x[c[1]:c[3]]); m.append(np.repeat(mass[1:3] * 1e10 / H, n[1:3])); lo.append(x[c[3]:])
    s = a * 1e3 / H   # comoving Mpc/h -> physical kpc
    return np.concatenate(hi).astype(float) * s, np.concatenate(m), np.concatenate(lo).astype(float) * s


def halos(run, snap, a):
    """Rockstar halos: id, upid, Mvir [Msun], Rvir [physical kpc], centre [physical kpc]."""
    cols = {1: 'id', 6: 'upid', 10: 'Mvir', 11: 'Rvir', 17: 'x', 18: 'y', 19: 'z'}   # column index in the hlist -> name
    t = pd.read_csv(path(snap) / run / 'rockstar/hlists' / f'hlist_{a:.5f}.list', sep=r'\s+', comment='#', header=None,
                    usecols=list(cols)).rename(columns=cols)
    # Msun/h -> Msun; comoving kpc/h -> physical kpc (Rvir); comoving Mpc/h -> physical kpc (centre).
    t['Mvir'] /= H; t['Rvir'] *= a / H
    for c in 'xyz':
        t[c] *= a * 1e3 / H
    return t


# %% [markdown]
# ### Centre, shape and quadrupole
# `densest` finds the host's densest core, `shell_shape` the axis ratios of an ellipsoidal shell, `eps2_at` the quadrupole of the
# expansion relative to its monopole at given radii.

# %%
def shrink(x, c, r0, r1, nmin=200):
    """Shrinking-sphere centre of the points x from c, radius r0 down to r1 (20 steps) or until fewer than nmin points."""
    for r in np.geomspace(r0, r1, 20):
        # Points within r of the current centre; stop below nmin, else move to their mean position.
        w = np.linalg.norm(x - c, axis=1) < r
        if w.sum() < nmin:
            break
        c = x[w].mean(0)
    return c


def densest(x, m, c0, rvir):
    """Centre [physical kpc] of the densest core near c0: the N_PEAKS highest peaks of the smoothed counts of the points x (N, 3)
    within CORE_BOX x rvir [kpc] of c0, each refined by a shrinking sphere (CORE_SHRINK), and the one with the most mass m within
    CORE_R kept."""
    h = CORE_BOX * rvir   # half-size of the box [kpc]
    # Counts on a 100^3 grid over the box about c0 (bin edges e per axis), smoothed with a Gaussian of 1 bin.
    D, e = np.histogramdd(x[np.linalg.norm(x - c0, axis=1) < h] - c0, bins=100, range=[(-h, h)] * 3)
    D = gaussian_filter(D, 1)
    # Local maxima (5^3-bin neighbourhood), sorted by height; the N_PEAKS highest.
    pk = np.argwhere(D == maximum_filter(D, 5))
    pk = pk[np.argsort(-D[tuple(pk.T)], kind='stable')][:N_PEAKS]
    # Each peak's bin centre refined by a shrinking sphere CORE_SHRINK (2 -> 0.2 kpc); keep the centre with most mass within CORE_R.
    cs = [shrink(x, c0 + np.array([(e[k][p[k]] + e[k][p[k] + 1]) / 2 for k in range(3)]), *CORE_SHRINK) for p in pk]
    return max(cs, key=lambda c: m[np.linalg.norm(x - c, axis=1) < CORE_R].sum())


def shell_shape(x, m, r1, r2, tol=1e-3, itmax=100):
    """q, s, axes (rows: major, intermediate, minor) and number of points of the ellipsoidal shell r1 <= r_ell < r2.

    x (N, 3) [kpc, centred], m (N,) [Msun], r1, r2 [kpc, major-axis radius]. Iterative reduced inertia tensor (weights m / r_ell^2;
    Zemp et al. 2011, ApJS 197, 30) until q = b/a and s = c/a change by less than tol; NaN for fewer than 50 points."""
    q = s = 1.        # start from a sphere
    E = np.eye(3)     # principal axes (rows) in the input frame
    for _ in range(itmax):
        # Coordinates along the current axes and squared ellipsoidal radius [kpc^2]; points in the shell.
        y = x @ E.T
        re2 = y[:, 0] ** 2 + (y[:, 1] / q) ** 2 + (y[:, 2] / s) ** 2
        w = (re2 >= r1 ** 2) & (re2 < r2 ** 2)
        if w.sum() < 50:
            return np.nan, np.nan, E, int(w.sum())
        # Eigen-decomposition of the reduced inertia tensor sum m x x^T / r_ell^2; axes ordered major, intermediate, minor.
        val, vec = np.linalg.eigh((m[w, None, None] * x[w, :, None] * x[w, None, :] / re2[w, None, None]).sum(0))
        E = vec[:, ::-1].T
        a, b, c = np.sqrt(val[::-1])
        # Converged when q and s change by less than tol.
        if abs(b / a - q) < tol and abs(c / a - s) < tol:
            return b / a, c / a, E, int(w.sum())
        q, s = b / a, c / a
    return q, s, E, int(w.sum())


def eps2_at(x, m, frame, rmax, radii):
    """eps2 and its part without m = 0, at radii, of the Multipole expansion of x (centred) in the frame (rows).

    eps2 = sqrt(sum over m of Phi_2m^2) / |Phi_00| (all five l = 2 terms) and eps2_na the same without Phi_20, each interpolated
    linearly in ln r at radii [kpc]; x (N, 3) [kpc], m (N,) [Msun], particles within rmax [kpc]."""
    k = np.linalg.norm(x, axis=1) < rmax   # particles within rmax
    # Multipole expansion of the particles rotated into the frame, exported to a temporary INI file and read back as text.
    with tempfile.NamedTemporaryFile(suffix='.ini') as f:
        agama.Potential(type='Multipole', particles=(x[k] @ frame.T, m[k]), lmax=LMAX, symmetry='none', gridSizeR=GRID, rmin=RMIN,
                        rmax=rmax).export(f.name)
        lines = open(f.name).read().split('#dPhi/dr')[0].splitlines()
    # Table of Phi_lm(r): header line '#radius', columns 'l=..,m=..'; r [kpc] and Phi_lm per column.
    i = next(j for j, s in enumerate(lines) if s.startswith('#radius'))
    cols = lines[i].split('\t')[1:]
    tab = np.array([[float(v) for v in s.split('\t')] for s in lines[i + 1:] if s.strip()])
    r, P = tab[:, 0], dict(zip(cols, tab[:, 1:].T))
    l2 = [c for c in cols if c.startswith('l=2,')]   # the five l = 2 terms
    tot = np.sqrt(sum(P[c] ** 2 for c in l2)) / np.abs(P['l=0,m=0'])
    non = np.sqrt(sum(P[c] ** 2 for c in l2 if c != 'l=2,m=0')) / np.abs(P['l=0,m=0'])
    return np.interp(np.log(radii), np.log(r), tot), np.interp(np.log(radii), np.log(r), non)


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data and measure both runs at every redshift
    # For each snapshot and run: read the particles, pick the host (the most massive distinct halo with no coarse particle within
    # $R_{\rm vir}$), centre it on its densest core, and measure the shells and the quadrupole. The particles are released before the
    # next run is read.

    # %%
    rows = []
    for snap, a in SNAPS.items():
        z = 1 / a - 1   # redshift
        for name, run in zip(('CDM', 'SIDM'), RUNS):
            # Fine particles (positions [kpc], masses [Msun]) and coarse positions; tree of the coarse ones.
            hi, mm, lo = read(run, snap, a)
            tl = cKDTree(lo)
            # Host: of the 20 most massive distinct halos (upid = -1), the first with no coarse particle within Rvir.
            cand = halos(run, snap, a).query('upid == -1').sort_values('Mvir', ascending=False, kind='stable').head(20)
            h0 = next(h for h in cand.itertuples() if not tl.query_ball_point([h.x, h.y, h.z], h.Rvir))
            c = densest(hi, mm, np.array([h0.x, h0.y, h0.z]), h0.Rvir)   # centre [kpc]: densest core
            # Positions about the centre [kpc], kept within R_KEEP Rvir.
            x = hi - c
            k = np.linalg.norm(x, axis=1) < R_KEEP * h0.Rvir
            x, m = x[k], mm[k]
            # Shell edges (log-spaced 0.4 kpc to Rvir) and their geometric-mean radii [kpc].
            edges = np.geomspace(R_SHELL_MIN, h0.Rvir, NSHELL + 1)
            rm = np.sqrt(edges[1:] * edges[:-1])
            # eps2 and eps2_na at the shell radii and at R_EPS, in the frame of the FRAME_SHELL shell's axes.
            e_t, e_n = eps2_at(x, m, shell_shape(x, m, *FRAME_SHELL)[2], R_KEEP * h0.Rvir, np.r_[rm, R_EPS])
            # One row per shell, then a row at R_EPS (2 kpc) with N = 0 and no shape.
            for j, (r1, r2) in enumerate(zip(edges[:-1], edges[1:])):
                q, s, _, n = shell_shape(x, m, r1, r2)
                rows.append(dict(z=z, run=name, r=rm[j], r_rvir=rm[j] / h0.Rvir, q=q, s=s, N=n, eps2=e_t[j], eps2_na=e_n[j]))
            rows.append(dict(z=z, run=name, r=R_EPS, r_rvir=R_EPS / h0.Rvir, q=np.nan, s=np.nan, N=0, eps2=e_t[-1], eps2_na=e_n[-1]))
            print(f'z {z:.2f} {name}: Mvir {h0.Mvir:.2e} Msun, Rvir {h0.Rvir:.1f} kpc, centre {np.round(c, 2)}, eps2(2 kpc) {e_t[-1]:.4f}',
                  flush=True)
            del hi, mm, lo, tl   # release the particles before the next run is read
    pd.DataFrame(rows).to_csv(ROOT / 'results/sims_sidm_shapes.csv', index=False, float_format='%.6g')

    # %% [markdown]
    # ### Particle mass of each run
    # The type-1 particle mass from the header of the $z=0$ snapshot, for the text.

    # %%
    mpart = []   # type-1 particle mass [Msun] of each run (mass table entry 1, 1e10 Msun/h)
    for name, run in zip(('CDM', 'SIDM'), RUNS):
        with open(Z0_DIR / run / 'snapshot_235.0', 'rb') as f:   # Gadget format 1: header after 4 bytes
            f.read(4)
            mpart.append(dict(run=name, m_part=np.frombuffer(f.read(256)[24:72], 'f8')[1] * 1e10 / H))
    pd.DataFrame(mpart).to_csv(ROOT / 'results/sims_sidm_particle_mass.csv', index=False)
