"""What turns the orbital planes of the early TNG50 cohorts (Section 6.4, Fig. sims_scramble): the stars of each cohort moved as test
particles through the gravity of their own progenitor, with parts of that gravity removed, and their orbital-plane coherence C
compared with that of the real stars.
results/sims_scramble.csv is read by scripts/paper_sims_scramble.py (Fig. sims_scramble a, b and the numbers of Section 6.4) and by
scripts/paper_letter_scramble.py (the Letter's Fig. 4 a: v_phi/sigma_z of the real stars and of the variants E, G_DM and D0);
scripts/auriga_scramble.py imports its functions for the same experiment on Auriga.

Method in plain words. A test particle is a massless point moved in a given gravitational field. Each cohort's stars are started
from their positions and velocities at the birth snapshot and integrated (AGAMA, Vasiliev 2019, MNRAS 482, 1525) for up to 0.5 Gyr
in the gravity of all the particles of their progenitor, written as a multipole expansion: a sum over spherical harmonics Y_lm of
degree l and order m, each with its own radial coefficient. Setting some coefficients to zero removes a known part of the gravity:
l = 0 keeps only the spherical part, m = 0 only the part axisymmetric about the cohort's spin, even l removes the lopsided (odd)
terms. Expanding the dark matter and the baryons separately lets one of them be made spherical while the other keeps its shape. If
the test particles in a reduced gravity keep their orbital planes while the real stars lose them, the part removed is what turns
the orbits. The test particles do not attract each other, and the expansion is not softened.

Galaxies: the 20 FULL progenitors of scripts/fetch_tng50_progenitors.py, full cutouts at every snapshot 21-40 (z = 4-1.5;
lmcold.sims.progenitor 'full': stars, gas, dark matter).
Frame: origin at the stars' shrinking-sphere centre (10 -> 0.5 kpc), velocities relative to the mass-weighted mean of the stars
within 2 kpc of it; axes rotated so that z is the cohort's spin at its birth snapshot (fixed: inertial).
Cohorts: the stars formed in the WINDOW (25 Myr) before a birth snapshot within 3 kpc (>= 30; short, so that the experiment starts
within ~0.1 orbit of formation), every second snapshot (odd numbers), followed
by ParticleIDs (members within 10 kpc) at every snapshot up to 0.5 Gyr after their mean formation time (>= 3 snapshots).
Gravity: AGAMA Multipole expansions (lmax 6, 40 radial nodes 0.05-60 kpc, no symmetry) of the particles within 60 kpc, made at every
snapshot; removing terms (real harmonics, the coefficients set to zero) removes parts of the gravity. The origin follows the stellar
centre, not an inertial frame: each potential gets a UniformAcceleration term -a0(t), a0 = mean force on 500 stars within 2 kpc.
Variants (C_sim of the test particles within 10 kpc at each snapshot time):
  evolving (linear interpolation between snapshots), all particles in one expansion: A all, B l = 0, C even l, C1 no |m| = 1,
    D |m| < 2, D0 m = 0 (axisymmetric about the cohort's spin);
  frozen at the birth snapshot: E (A frozen);
  evolving, dark matter and baryons (gas + stars) in separate expansions: A_split, G_DM (baryons l = 0: only the DM non-round),
    G_bar (DM l = 0);
  frozen at birth, separate expansions: E_split, C_fr (even l), D0_fr (m = 0), H1 (DM axisymmetric about its own minor axis: even l,
    m = 0 in the halo frame), H1p (about its major axis), H2 (DM rotated rigidly so that its minor axis lies on the cohort's spin and
    its major axis on the projection of the original one on the cohort plane).
The three versions of the Letter's Fig. 4 (a) are E (all of the gravity, frozen at birth: no change of the potential in time is
needed), G_DM (only the dark matter keeps its shape, the gas and stars are made spherical) and D0 (all of the gravity made
axisymmetric about the cohort's spin); G_DM and D0 evolve with the snapshots. The test particles keep their rotation only in D0.
Halo axes: iterative reduced inertia tensor of the DM within ellipsoidal radius 3 kpc (halo_axes); misalign = angle of the minor axis
to the cohort's spin at birth [deg].
Orbits: agama.orbit (dtype=object: the float64 interpolated trajectory).
Rotational support (real stars: _meas; test particles: _sim): mass-weighted v_phi / sigma_z of the cohort members that at the birth
snapshot lie at r < 3 kpc and cylindrical R > 0.5 kpc (R_SUP, as the stars at formation in Fig. sims_birth; in_ring), chosen once and
followed wherever they go, about
three axes: the set's own spin at that time (vs_*_own), the galaxy's disc spin at that snapshot (lmcold.sims.disc_frame of all the real
stars; vs_*_disc, as Fig. sims_heating b; the test particles of a variant are measured about the real disc too) and the cohort's spin
at its birth snapshot (vs_*_birth, the integration frame's z).
Units: kpc, km/s, Msun (agama.setUnits); AGAMA's time unit is 1 kpc / (km/s) = KPC_KMS_GYR Gyr.
Writes results/sims_scramble.csv (halo, s_birth, z, n0, misalign, variant, t [Gyr since the birth snapshot], C_meas, C_sim,
vs_meas_own, vs_meas_disc, vs_meas_birth, vs_sim_own, vs_sim_disc, vs_sim_birth).
Run: .venv/bin/python scripts/manifest.py run sims_scramble.py   (AGAMA; 4 processes; 146 s in the recorded run of docs/run_times.txt)"""
# %% [markdown]
# ### Settings
# The expansion (order, radial grid, outer radius), the cohort definition, the radii of the frame and of the members, and the
# variants of the gravity that keep all particles in one expansion: each is a rule saying which $(l, m)$ terms are kept.

# %%
import tempfile
from multiprocessing import Pool

import agama
import numpy as np
import pandas as pd

from fetch_tng50_progenitors import FULL, FULL_SNAPS
from lmcold import sims
from lmcold.sources import ROOT

agama.setUnits(mass=1, length=1, velocity=1)  # AGAMA units: Msun, kpc, km/s
KPC_KMS_GYR = 0.9778        # 1 kpc / (km/s) in Gyr: AGAMA's time unit
LMAX, GRID, RMIN, RMAX = 6, 40, 0.05, 60.   # Multipole: highest degree l, radial nodes, inner and outer node [kpc]
R_SUP = (0.5, 3.0)   # kpc: the stars followed in v_phi/sigma_z, chosen at the birth snapshot (cylindrical R > 0.5, r < 3), as at formation
WINDOW, DTMAX = 0.025, 0.5   # Gyr: cohort window before the birth snapshot (was 0.1: started ~50 Myr after formation)
R_COHORT, MIN_STARS, BIRTH_EVERY = 3.0, 30, 2   # kpc, stars: cohorts within R_COHORT, of >= MIN_STARS, at odd snapshots
R_VEL, R_MEMBER = 2.0, 10.0   # kpc: stars defining the frame velocity; cohort members counted within R_MEMBER
# Rules (l, m) -> keep the term: all terms, the monopole only, even l only, m = 0 only, even l and m = 0.
ALL, L0, EVEN = (lambda l, m: True), (lambda l, m: l == 0), (lambda l, m: l % 2 == 0)
M0, AXI = (lambda l, m: m == 0), (lambda l, m: l % 2 == 0 and m == 0)
# Variants with all particles in one evolving expansion: name -> rule (module docstring).
EVOLVING = {'A': ALL, 'B': L0, 'C': EVEN, 'C1': lambda l, m: abs(m) != 1, 'D': lambda l, m: abs(m) < 2, 'D0': M0}


# %% [markdown]
# ### Helpers: the frame of each snapshot and the halo's axes
# `load` reads one snapshot of a progenitor and moves it to the stars' centre and mean velocity; `frame` builds axes with $z$ along a
# given direction; `halo_axes` finds the principal axes of the dark matter.

# %%
def shrink(x, m, r0=10., r1=0.5, nmin=20):
    """Shrinking-sphere centre [kpc] of points x (N, 3) with weights m (N,): the weighted mean within radii from r0 down to r1 (15
    steps, each about the previous centre), stopping when fewer than nmin points remain."""
    c = np.zeros(3)  # start at the origin of x
    # 15 radii, logarithmically spaced from r0 to r1 [kpc]; each step re-centres on the points inside the current sphere.
    for rad in np.geomspace(r0, r1, 15):
        w = np.linalg.norm(x - c, axis=1) < rad
        if w.sum() < nmin:
            break
        c = np.average(x[w], axis=0, weights=m[w])
    return c


def load(name, s, b):
    """Particles of snapshot s in the local frame (stellar centre, mean velocity of the stars within 2 kpc), or None.

    Returns dict s, a (scale factor), lb (lookback [Gyr]), st (stars: x [kpc], v [km/s], m [Msun], ids, age_form), dm and gas
    (x, m); positions relative to the stars' centre, star velocities relative to their mean within R_VEL."""
    p = sims.progenitor(name, s, 'full', b)  # full cutout of snapshot s along main branch b (dict per particle type)
    # Skip snapshots without a cutout, stars or dark matter.
    if not p or 'stars' not in p or 'dm' not in p:
        return None
    st = p['stars']
    c0 = shrink(st['x'], st['m'])  # stellar centre [kpc]
    w2 = np.linalg.norm(st['x'] - c0, axis=1) < R_VEL  # stars within R_VEL of it
    v0 = np.average(st['v'][w2], axis=0, weights=st['m'][w2])
    # Positions relative to c0 for every type; velocities relative to v0 [km/s] for the stars.
    out = dict(s=s, a=p['a'], lb=p['lookback'], st=dict(st, x=st['x'] - c0, v=st['v'] - v0), dm=dict(x=p['dm']['x'] - c0, m=p['dm']['m']))
    # Gas, or an empty set if the cutout has none.
    gas = p.get('gas', dict(x=np.zeros((0, 3)), m=np.zeros(0)))
    out['gas'] = dict(x=gas['x'] - c0, m=gas['m'])
    return out


def frame(n):
    """Rows (x, y, z) with z along the unit vector n and x in the simulation x-y plane."""
    ez = n / np.linalg.norm(n)
    ex = np.cross([0, 0, 1.], ez)  # perpendicular to ez and to the simulation z axis
    # Normalise; if n is along the simulation z axis, take the simulation x axis.
    ex = ex / np.linalg.norm(ex) if np.linalg.norm(ex) > 1e-8 else np.array([1., 0, 0])
    return np.stack([ex, np.cross(ez, ex), ez])


def halo_axes(x, rmax=3., tol=1e-3, itmax=50):
    """Rows (major, intermediate, minor) of the iterative reduced inertia tensor of the points x within ellipsoidal radius rmax;
    right-handed, with the largest component of the major and minor axes positive (a fixed sign, whatever the eigensolver returns).

    x: (N, 3) [kpc], centred; rmax [kpc]. Each iteration weights the points by 1 / r_ell^2 inside the ellipsoid of the previous axis
    ratios q = b/a, s = c/a (Zemp et al. 2011, ApJS 197, 30), until q and s change by less than tol."""
    q = s = 1.  # start from a sphere
    E = np.eye(3)  # current axes (rows)
    for _ in range(itmax):
        y = x @ E.T  # coordinates along the current axes
        re2 = y[:, 0] ** 2 + (y[:, 1] / q) ** 2 + (y[:, 2] / s) ** 2   # ellipsoidal radius squared [kpc^2]
        w = (re2 < rmax ** 2) & (re2 > 0)  # points inside the ellipsoid (the centre excluded: weight 1 / re2)
        # Reduced inertia tensor sum x_i x_j / r_ell^2; eigenvalues ascending, so the axes are reversed to major first.
        val, vec = np.linalg.eigh((x[w, :, None] * x[w, None, :] / re2[w, None, None]).sum(0))
        E = vec[:, ::-1].T
        qn, sn = np.sqrt(val[1] / val[2]), np.sqrt(val[0] / val[2])  # new axis ratios b/a, c/a
        if abs(qn - q) < tol and abs(sn - s) < tol:
            break
        q, s = qn, sn
    E[[0, 2]] *= np.sign(E[[0, 2], np.abs(E[[0, 2]]).argmax(1)])[:, None]   # fixed signs: largest component of major, minor positive
    E[1] = np.cross(E[2], E[0])                                              # right-handed (eigh may return either sign)
    return E


# %% [markdown]
# ### Gravity: multipole expansions and their parts
# `expansion` writes the AGAMA multipole expansion of a set of particles to a file; `keep_terms` copies that file with some
# coefficients set to zero; `accel` adds the inertial acceleration of the frame that follows the stellar centre; `evolving` joins
# the expansions of successive snapshots into one potential interpolated linearly in time.

# %%
def expansion(x, m, path):
    """Write the AGAMA Multipole expansion (LMAX, GRID nodes from RMIN to RMAX, no symmetry) of particles x (N, 3) [kpc], m (N,)
    [Msun] to the file path; returns path."""
    agama.Potential(type='Multipole', particles=(x, m), lmax=LMAX, symmetry='none', gridSizeR=GRID, rmin=RMIN, rmax=RMAX).export(path)
    return path


def keep_terms(src, keep, tag):
    """Copy of the Multipole INI src with the coefficients of the (l, m) terms for which keep(l, m) is False set to zero."""
    out, cols = [], None  # output lines; cols: keep flag per column of the current coefficient table
    for line in open(src):
        s = line.rstrip('\n')
        if s.startswith('#radius'):   # header of a coefficient table: columns 'l=..,m=..'
            cols = [True] + [keep(*map(int, c.replace('l=', '').replace('m=', '').split(','))) for c in s.split('\t')[1:]]
        # Data row of a coefficient table: zero the columns of the terms not kept (the radius column is kept).
        elif cols and s and not s.startswith('#'):
            s = '\t'.join(v if k else '0' for v, k in zip(s.split('\t'), cols))
        out.append(s)
    dst = src.replace('.ini', f'_{tag}.ini')  # new file name with the variant tag
    open(dst, 'w').write('\n'.join(out) + '\n')
    return dst


def accel(pots, centres, t):
    """UniformAcceleration potential -a0(t) for the potentials (one per time t) and the frame points of each.

    a0 at each time is the mean force of that time's potential on its frame points (centres: one (K, 3) array [kpc] per time);
    t in AGAMA time units. Subtracting it keeps the frame's origin, the stellar centre, at rest."""
    acc = np.array([p.force(c).mean(axis=0) for p, c in zip(pots, centres)])  # (times, 3) mean force per unit mass
    return agama.Potential(type='UniformAcceleration', file=np.column_stack([t, -acc]))


def evolving(files, t, path):
    """AGAMA Evolving potential from the expansion files (one per time t, AGAMA units), linearly interpolated in time; its
    description is written to path."""
    # AGAMA INI: one line 'time file' per snapshot.
    open(path, 'w').write('[Potential]\ntype=Evolving\ninterpLinear=true\nTimestamps\n' + ''.join(f'{tk:.8f} {f}\n' for tk, f in zip(t, files)))
    return agama.Potential(path)


def particles(c, parts, R):
    """Positions (rotated by R, rows: axes) and masses of the particle types parts ('dm', 'gas', 'st') of snapshot c, within RMAX."""
    X = np.concatenate([c[p]['x'] for p in parts]) @ R.T  # (N, 3) [kpc] in the frame with axes R
    M = np.concatenate([c[p]['m'] for p in parts])  # (N,) [Msun]
    k = np.linalg.norm(X, axis=1) < RMAX  # inside the outer node of the expansion
    return X[k], M[k]


# %% [markdown]
# ### Rotational support of a set of stars
# $v_\phi/\sigma_z$ about a given axis (`support`), the spin of a set (`spin`), the stars at 0.5 < $R$ < 3 kpc that are followed
# (`in_ring`), and $v_\phi/\sigma_z$ about the three axes of the output (`supports`).

# %%
def support(x, v, m, ez):
    """Mass-weighted v_phi / sigma_z of particles x, v (N, 3) about the unit axis ez (v_phi positive along ez's sense); NaN with fewer
    than 5 particles."""
    if len(m) < 5:
        return np.nan
    F = frame(ez)
    X, V = x @ F.T, v @ F.T  # positions and velocities in a frame with z along ez
    # Azimuthal velocity (x v_y - y v_x) / R [km/s]; vertical velocity relative to the mass-weighted mean.
    vphi = (X[:, 0] * V[:, 1] - X[:, 1] * V[:, 0]) / np.hypot(X[:, 0], X[:, 1])
    vz = V[:, 2] - np.average(V[:, 2], weights=m)
    # Mass-weighted mean v_phi over the mass-weighted rms of v_z.
    return np.average(vphi, weights=m) / np.sqrt(np.average(vz ** 2, weights=m))


def spin(x, v, m):
    """Unit vector of the total angular momentum of particles x, v (N, 3), m (N,)."""
    L = (m[:, None] * np.cross(x, v)).sum(0)  # total angular momentum [Msun kpc km/s]
    return L / np.linalg.norm(L)


def in_ring(x, v, m):
    """The particles at r < R_SUP[1] kpc and cylindrical R > R_SUP[0] kpc about the spin of those within R_SUP[1] (as the stars at
    formation, lmcold.sims.birth_phase_space)."""
    r = np.linalg.norm(x, axis=1)  # spherical radius [kpc]
    # Spin of the particles within R_SUP[1]; cylindrical R about it is |x cross ez|.
    ez = spin(x[r < R_SUP[1]], v[r < R_SUP[1]], m[r < R_SUP[1]])
    return (r < R_SUP[1]) & (np.linalg.norm(np.cross(x, ez), axis=1) > R_SUP[0])


def supports(x, v, m, disc, birth):
    """v_phi / sigma_z about the set's own spin, the galaxy's disc spin and the cohort's spin at birth (keys own, disc, birth); NaN
    when the set is empty (all particles beyond 10 kpc), as C."""
    # Empty set: NaN for all three axes.
    if m.sum() == 0:
        return dict(own=np.nan, disc=np.nan, birth=np.nan)
    return dict(own=support(x, v, m, spin(x, v, m)), disc=support(x, v, m, disc), birth=support(x, v, m, birth))


# %% [markdown]
# ### One cohort: the real stars, the variants of the gravity and the orbits
# `run_cohort` measures the real stars at each later snapshot, builds every variant of the gravity from the expansions of the dark
# matter, the baryons and both together, integrates the stars as test particles in each, and measures $C$ and $v_\phi/\sigma_z$ of
# the test particles at the snapshot times.

# %%
def run_cohort(snaps, i0, ids, t_form, tmp):
    """Rows of results/sims_scramble.csv (without halo) for one cohort.

    snaps: the loaded snapshots of a progenitor (load()); i0: index of the birth snapshot in snaps; ids: particle ids of the cohort;
    t_form: its mass-weighted mean lookback time of formation [Gyr]; tmp: directory for the expansion files. Empty if fewer than
    three snapshots fall within DTMAX of formation."""
    c0 = snaps[i0]  # the birth snapshot
    # Cohort members present at birth within R_MEMBER of the centre.
    w0 = np.isin(c0['st']['ids'], ids) & (np.linalg.norm(c0['st']['x'], axis=1) < R_MEMBER)
    xs, vs, ms = c0['st']['x'][w0], c0['st']['v'][w0], c0['st']['m'][w0]   # the cohort at its birth snapshot
    sel = in_ring(xs, vs, ms)        # the stars whose v_phi/sigma_z is followed: chosen once, at the birth snapshot
    ids_sel = c0['st']['ids'][w0][sel]
    n = (ms[:, None] * np.cross(xs, vs)).sum(0); n /= np.linalg.norm(n)   # the cohort's spin at birth
    R = frame(n)   # integration frame: z along n, fixed
    # Snapshots from birth up to DTMAX after the mean formation time.
    later = [i for i in range(i0, len(snaps)) if t_form - snaps[i]['lb'] <= DTMAX + 1e-6]
    if len(later) < 3:
        return []
    t = np.array([(c0['lb'] - snaps[i]['lb']) / KPC_KMS_GYR for i in later])   # snapshot times since birth, AGAMA units
    # 500 random stars within 2 kpc of each snapshot (rotated by P): the points on which the frame acceleration is averaged.
    pick = lambda c, P: (lambda xc: xc[np.random.default_rng(0).choice(len(xc), min(500, len(xc)), replace=False)])(
        c['st']['x'][np.linalg.norm(c['st']['x'], axis=1) < 2] @ P.T)
    cen = [pick(snaps[i], R) for i in later]
    # The real stars at each later snapshot: C of the members, the galaxy's disc spin, v_phi/sigma_z of the followed stars.
    meas, vs_meas, disc = [], [], []
    for i in later:
        st = snaps[i]['st']
        w = np.isin(st['ids'], c0['st']['ids'][w0]) & (np.linalg.norm(st['x'], axis=1) < R_MEMBER)
        meas.append(sims.coherence(st['x'][w], st['v'][w], st['m'][w]))
        disc.append(sims.disc_frame(st['x'], st['v'], st['m'])[2][2])   # the galaxy's disc spin (simulation frame)
        f = np.isin(st['ids'], ids_sel)
        vs_meas.append(supports(st['x'][f], st['v'][f], st['m'][f], disc[-1], n))
    # Expansions at every later snapshot, in the integration frame: all particles, dark matter only, baryons (gas + stars) only.
    f_all = [expansion(*particles(snaps[i], ('dm', 'gas', 'st'), R), f'{tmp}/s{i}_all.ini') for i in later]
    f_dm = [expansion(*particles(snaps[i], ('dm',), R), f'{tmp}/s{i}_dm.ini') for i in later]
    f_bar = [expansion(*particles(snaps[i], ('gas', 'st'), R), f'{tmp}/s{i}_bar.ini') for i in later]
    pots = {}                                                     # variant: (potential, rotation cohort frame -> integration frame)
    # Evolving, one expansion of all particles: A, B, C, C1, D, D0.
    for v, keep in EVOLVING.items():
        fs = [keep_terms(f, keep, v) for f in f_all]
        pots[v] = (agama.Potential(evolving(fs, t, f'{tmp}/evo_{v}.ini'), accel([agama.Potential(f) for f in fs], cen, t)), np.eye(3))
    # E: all of the gravity, frozen at the birth snapshot.
    p0 = agama.Potential(f_all[0])
    pots['E'] = (agama.Potential(p0, accel([p0, p0], [cen[0], cen[0]], t[[0, -1]])), np.eye(3))
    # Evolving, dark matter and baryons expanded separately: (terms kept of the dark matter, of the baryons).
    for v, (kd, kb) in {'A_split': (ALL, ALL), 'G_DM': (ALL, L0), 'G_bar': (L0, ALL)}.items():
        fd, fb = [keep_terms(f, kd, v) for f in f_dm], [keep_terms(f, kb, v) for f in f_bar]
        both = [agama.Potential(agama.Potential(d), agama.Potential(b)) for d, b in zip(fd, fb)]
        pots[v] = (agama.Potential(evolving(fd, t, f'{tmp}/evo_{v}_dm.ini'), evolving(fb, t, f'{tmp}/evo_{v}_bar.ini'),
                                   accel(both, cen, t)), np.eye(3))

    def frozen(pd_, pb_, c):
        """Dark matter pd_ plus baryons pb_, frozen, with the frame acceleration on the points c."""
        return agama.Potential(pd_, pb_, accel([agama.Potential(pd_, pb_)] * 2, [c, c], t[[0, -1]]))

    # Frozen at birth, separate expansions, the same terms kept in both.
    for v, keep in {'E_split': ALL, 'C_fr': EVEN, 'D0_fr': M0}.items():
        pots[v] = (frozen(agama.Potential(keep_terms(f_dm[0], keep, v)), agama.Potential(keep_terms(f_bar[0], keep, v)), cen[0]), np.eye(3))
    # The halo's shape: H1, H1p (dark matter axisymmetric about its own minor or major axis) and H2 (halo rotated onto the cohort).
    E = halo_axes(c0['dm']['x'])                                  # simulation frame (centred)
    misalign = np.degrees(np.arccos(min(1., abs(E[2] @ n))))      # deg, halo minor axis to the cohort's spin
    for v, P in (('H1', E), ('H1p', E[[1, 2, 0]])):               # integration frame: z' = minor (H1) or major (H1p) axis
        fd = expansion(*particles(c0, ('dm',), P), f'{tmp}/{v}_dm.ini')
        fb = expansion(*particles(c0, ('gas', 'st'), P), f'{tmp}/{v}_bar.ini')
        pots[v] = (frozen(agama.Potential(keep_terms(fd, AXI, 'axi')), agama.Potential(fb), pick(c0, P)), P @ R.T)
    t1 = E[0] - (E[0] @ n) * n; t1 /= np.linalg.norm(t1)   # projection of the halo's major axis on the cohort plane
    T = np.stack([t1, np.cross(n, t1), n])
    c2 = dict(c0, dm=dict(c0['dm'], x=c0['dm']['x'] @ E.T @ T))  # x -> T^T E x: halo axes onto (t1, t2, n)
    fd = expansion(*particles(c2, ('dm',), R), f'{tmp}/H2_dm.ini')
    pots['H2'] = (frozen(agama.Potential(fd), agama.Potential(f_bar[0]), cen[0]), np.eye(3))
    # Orbits of the cohort's stars in each variant, and C and v_phi/sigma_z of the test particles at the snapshot times.
    rows = []
    x, v = xs @ R.T, vs @ R.T
    for var, (pot, Q) in pots.items():
        orbs = agama.orbit(potential=pot, ic=np.column_stack([x @ Q.T, v @ Q.T]), time=t[-1], dtype=object, verbose=False)
        w6 = np.stack([o(t) for o in orbs], axis=1)   # (times, stars, 6): positions and velocities
        Cs = [sims.coherence(s[w, :3], s[w, 3:], ms[w]) for s in w6 for w in [np.linalg.norm(s[:, :3], axis=1) < R_MEMBER]]
        vs_sim = [supports(s[sel, :3] @ Q, s[sel, 3:] @ Q, ms[sel], R @ d, np.array([0., 0., 1.]))   # in the cohort frame R
                  for s, d in zip(w6, disc)]
        rows += [dict(s_birth=c0['s'], z=1 / c0['a'] - 1, n0=len(ms), misalign=misalign, variant=var, t=tk * KPC_KMS_GYR, C_meas=cm,
                      C_sim=cs, **{f'vs_meas_{k}': q for k, q in vm.items()}, **{f'vs_sim_{k}': q for k, q in vsim.items()})
                 for tk, cm, cs, vm, vsim in zip(t, meas, Cs, vs_meas, vs_sim)]
    return rows


# %% [markdown]
# ### One progenitor
# `one` loads every snapshot of a progenitor, finds the cohorts born before each odd snapshot, and runs each.

# %%
def one(name):
    """Rows of results/sims_scramble.csv for all cohorts of the FULL progenitor name."""
    b = sims.main_branch(name)  # main-progenitor branch of the galaxy
    # Every snapshot of FULL_SNAPS in the local frame (load), missing ones dropped.
    snaps = [d for d in (load(name, s, b) for s in FULL_SNAPS) if d is not None]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, c in enumerate(snaps):
            # Birth snapshots: odd snapshot numbers only.
            if c['s'] % BIRTH_EVERY == 0:
                continue
            st = c['st']
            ar = st['age_form'] - c['lb']   # Gyr before this snapshot at which each star formed
            # Cohort: stars formed within WINDOW before this snapshot, within R_COHORT of the centre.
            k = (ar >= 0) & (ar < WINDOW) & (np.linalg.norm(st['x'], axis=1) < R_COHORT)
            if k.sum() >= MIN_STARS:
                # Mass-weighted mean formation lookback time [Gyr]; run the experiment and tag the rows with the galaxy.
                t_form = float(np.average(st['age_form'][k], weights=st['m'][k]))
                rows += [dict(halo=name, **r) for r in run_cohort(snaps, i, st['ids'][k], t_form, tmp)]
    print(name, 'done', flush=True)
    return rows


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data and run every progenitor
    # The 20 FULL progenitors, four at a time; each process reads its progenitor's full cutouts through `load`.

    # %%
    # One process per progenitor, four at a time; all rows in one table.
    with Pool(4) as pool:
        t = pd.DataFrame([r for rows in pool.map(one, FULL) for r in rows])
    t.to_csv(ROOT / 'results/sims_scramble.csv', index=False, float_format='%.5f')
    # Number of galaxies and of cohorts in the table.
    print(f'{t.halo.nunique()} galaxies, {t.groupby("halo").s_birth.nunique().sum()} cohorts')
