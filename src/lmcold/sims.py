"""Simulated LMC analogues "observed" like the LMC giants (paper Section 6).

Simulations (star particles at z = 0; all hydrodynamical runs available to us except COLIBRE, which is proprietary):
- TNG50-1 (Nelson et al. 2019; Pillepich et al. 2019): all 681 centrals with bound stellar mass 1.5-5e9 Msun and subhalo
  mass 5e10-5e11 Msun (scripts/fetch_tng50.py; cutouts with Coordinates, Velocities, Masses, GFM_Metals,
  GFM_StellarFormationTime; sources.yaml: sims_tng50). Masses are present-day; formed mass = present / present_fraction(age, [M/H])
  (lmcold.surviving_mass, the Kroupa IMF of the LMC mass measurement).
- Auriga level 3, Halos_1e11msol (Grand et al. 2024): Au2, 8, 9, 10, 11 (sources.yaml: sims_auriga_l3). Formed mass =
  GFM_InitialMass.
- FIRE-2 public release (Wetzel et al. 2023): m11d, m11e, m11h at 7100 Msun resolution (sources.yaml: sims_fire2). Formed mass as for
  TNG50.
[Fe/H] = log10[(X_Fe / X_H) / (X_Fe / X_H)_sun] with the solar A(Fe) = 7.46 of Asplund, Amarsi & Grevesse (2021).

Observing (load -> disc_frame -> place):
1. disc_frame: positions and velocities relative to the mass-weighted centre of velocity of the stars within ALIGN_R, rotated so that z
   is along their total angular momentum (prograde rotation counter-clockwise about +z).
2. place: the disc-frame particles put at the LMC position with the orientation and systemic motion of a kinfit parameter vector:
   x along the line of nodes, y along n x line, z along the normal n (so the simulation's prograde v_phi is kinfit's positive v_phi),
   every particle at its own 3D position; returns sky positions, proper motions (at the particle's own distance) and v_los.
Formulae and conventions as lmcold.kinfit._model (verified by tests/test_sims.py: a particle disc moving with the kinfit mean field
gives back kinfit.model's proper motions and v_los).
Units: positions in physical kpc, velocities in km/s (peculiar, about the galaxy's bulk motion), masses in Msun, ages as lookback
times in Gyr, [Fe/H] in dex. Particle files are read from SIMS (~/data/sims), fetched by the scripts named above.
"""
import glob
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter

from . import geometry as geo
from . import kinfit
from .constants import KAPPA, LMC

SIMS = Path("~/data/sims").expanduser()
TNG_DIR = SIMS / "tng/TNG50-1/lmc_analogue_cutouts"
TNG_BIRTH_DIR = SIMS / "tng/TNG50-1/lmc_analogue_birth"  # BirthPos, BirthVel of the same cutouts (scripts/fetch_tng50.py)
TNG_H, TNG_BOX = 0.6774, 35000.0  # Hubble parameter, comoving box size [ckpc/h] of TNG50-1
TNG_OM = 0.3089  # Omega_m of the TNG simulations (Planck 2015 cosmology, like TNG_H)
# Main progenitors of the analogues of scripts/fetch_tng50_progenitors.py (SubLink trees; star, gas and full-cadence cutouts).
TNG_TREE_DIR = SIMS / "tng/TNG50-1/debate_heating"
TNG_TRACK_DIR = SIMS / "tng/TNG50-1/old_star_tracks"
TNG_GAS_DIR = SIMS / "tng/TNG50-1/gas_progenitors"
TNG_FULL_DIR = SIMS / "tng/scramble/TNG50-1"
TNG_DM_DIR = SIMS / "tng/TNG50-1/dm_extend"  # dark matter where TNG_FULL_DIR has no cutout (scripts/fetch_tng50_progenitors.py)
AURIGA = {f"Au{n}": SIMS / f"auriga/level3/Halos_1e11msol/halo_{n}" for n in (2, 8, 9, 10, 11)}
FIRE = {n: SIMS / f"fire/{n}_res7100" for n in ("m11d", "m11e", "m11h")}
A_FE_SUN = 7.46  # Asplund, Amarsi & Grevesse 2021, A&A 653, A141, Table 2
FE_H_SUN = 10 ** (A_FE_SUN - 12) * 55.845 / 1.008  # solar Fe/H mass ratio (number ratio x atomic masses of Fe and H)
ALIGN_R = 10.0  # kpc, stars defining the centre of velocity and the spin axis
R_MAX = 50.0  # kpc, particles kept: a 12-deg cone behind a zoom galaxy otherwise reaches other haloes of the volume


def names():
    """All analogues: (suite, name) for TNG50 (subhalo ids), Auriga L3 and FIRE-2.

    Returns
    -------
    list of (str, str)
        suite ('TNG50', 'Auriga' or 'FIRE-2') and name (TNG50 subhalo id as a string, 'Au2', ..., 'm11d', ...); the TNG50 ids are
        read from TNG_DIR/sample.json.
    """
    tng = [s["id"] for s in json.load(open(TNG_DIR / "sample.json"))["subhalos"]]
    return [("TNG50", str(i)) for i in tng] + [("Auriga", n) for n in AURIGA] + [("FIRE-2", n) for n in FIRE]


def _feh(x_fe, x_h):
    """[Fe/H] from Fe and H mass fractions (floor at -9 for primordial particles)."""
    return np.log10(np.clip(x_fe, 1e-15, None) / np.clip(x_h, 1e-15, None) / FE_H_SUN).clip(-9)


def lookback(a, h, om):
    """Lookback time [Gyr] to scale factor a in flat LCDM with Hubble parameter h and matter density om.

    Parameters
    ----------
    a : float or ndarray
        Scale factor(s), 1e-4 <= a <= 1.
    h : float
        H0 / (100 km/s/Mpc).
    om : float
        Omega_m today (Omega_Lambda = 1 - om; radiation neglected).

    Returns
    -------
    float or ndarray
        Lookback time from a = 1 to a [Gyr]: t(1) - t(a), with t(a) = int_0^a da' / (a' H(a')) and
        H(a) = H0 sqrt(om a^-3 + 1 - om), integrated by the trapezoidal rule on 20000 points.
    """
    grid = np.linspace(1e-4, 1.0, 20000)
    H0 = 100 * h / 977.792  # 1/Gyr (977.792 = 1 / (km/s/Mpc) in Gyr)
    dt = 1 / (grid * H0 * np.sqrt(om / grid**3 + 1 - om))
    t = np.concatenate([[0], np.cumsum(0.5 * (dt[1:] + dt[:-1]) * np.diff(grid))])  # cosmic time at a
    return t[-1] - np.interp(a, grid, t)


def load(suite, name, survival):
    """Star particles of one analogue within R_MAX of its centre: dict x, v (N, 3) [kpc, km/s, about the galaxy's centre and bulk
    motion, simulation axes], m (present) and m_formed [Msun], feh [dex], age [Gyr].

    survival : pandas.DataFrame
        lmcold.surviving_mass.survival_grid (mh, logage, present_fraction): formed mass = present / present_fraction for TNG50 and
        FIRE-2, whose snapshots carry no initial masses (Auriga: GFM_InitialMass).
    """
    st = _read(suite, name)
    keep = np.linalg.norm(st["x"], axis=1) < R_MAX
    st = {k: v[keep] for k, v in st.items()}
    # No initial masses in the snapshot (TNG50, FIRE-2): formed = present / surviving fraction at the particle's [M/H] and age.
    if np.isnan(st["m_formed"]).all():
        st["m_formed"] = st["m"] / on_grid(survival, "present_fraction", st["feh"], st["age"])
    return st


def _read(suite, name):
    """All star particles of one analogue (load(), without the radius cut; m_formed NaN where the snapshot has no initial mass)."""
    if suite == "TNG50":
        sub = {str(s["id"]): s for s in json.load(open(TNG_DIR / "sample.json"))["subhalos"]}[name]
        with h5py.File(TNG_DIR / f"cutout_{name}.hdf5") as f:
            g = f["PartType4"]
            af = g["GFM_StellarFormationTime"][:]
            ok = af > 0  # wind particles have af <= 0
            x, v, m, met = (g[k][:][ok] for k in ("Coordinates", "Velocities", "Masses", "GFM_Metals"))
        c = np.array([sub["pos_x"], sub["pos_y"], sub["pos_z"]])
        x = ((x - c + TNG_BOX / 2) % TNG_BOX - TNG_BOX / 2) / TNG_H  # periodic box, physical kpc at z = 0
        v = v - np.array([sub["vel_x"], sub["vel_y"], sub["vel_z"]])
        # Masses in 1e10 Msun/h; GFM_Metals columns H, He, C, N, O, Ne, Mg, Si, Fe, ... (index 8 is Fe, 0 is H).
        return dict(x=x, v=v, m=m * 1e10 / TNG_H, m_formed=np.full(len(m), np.nan), feh=_feh(met[:, 8], met[:, 0]),
                    age=lookback(af[ok], TNG_H, TNG_OM))
    if suite == "Auriga":
        d = AURIGA[name]
        parts = []
        for fn in sorted(glob.glob(str(d / "snapdir_251/snapshot_251.*.hdf5"))):
            with h5py.File(fn) as f:
                hdr = dict(f["Header"].attrs)
                if "PartType4" in f:
                    g = f["PartType4"]
                    parts.append([g[k][:] for k in ("Coordinates", "Velocities", "Masses", "GFM_InitialMass", "GFM_Metals",
                                                    "GFM_StellarFormationTime")])
        x, v, m, mi, met, af = (np.concatenate(p) for p in zip(*parts))
        h, a = hdr["HubbleParam"], hdr["Time"]
        with h5py.File(sorted(glob.glob(str(d / "groups_251/fof_subhalo_tab_251.*.hdf5")))[0]) as f:
            c, vc = f["Subhalo/SubhaloPos"][0], f["Subhalo/SubhaloVel"][0]  # main subhalo of the first FoF group
        ok = af > 0  # wind particles have af <= 0
        # Ages: lookback to formation minus the lookback to the snapshot (Auriga's last snapshot is at a = Time).
        # Lengths in Mpc/h comoving (UnitLength 3.08568e24 cm), velocities km/s sqrt(a), masses 1e10 Msun/h.
        return dict(x=(x[ok] - c) * 1e3 * a / h, v=v[ok] * np.sqrt(a) - vc, m=m[ok] * 1e10 / h, m_formed=mi[ok] * 1e10 / h,
                    feh=_feh(met[ok, 8], met[ok, 0]), age=lookback(af[ok], h, hdr["Omega0"]) - lookback(a, h, hdr["Omega0"]))
    d = FIRE[name]
    with h5py.File(d / "output/snapshot_600.hdf5") as f:
        hdr = dict(f["Header"].attrs)
        g = f["PartType4"]
        x, v, m, Z, af = (g[k][:] for k in ("Coordinates", "Velocities", "Masses", "Metallicity", "StellarFormationTime"))
    h, a = hdr["HubbleParam"], hdr["Time"]
    with h5py.File(d / "track/host_coordinates.hdf5") as f:
        c, vc = f["host.position"][600, 0], f["host.velocity"][600, 0]  # comoving kpc, km/s (Wetzel et al. 2023, Sect. 4.7)
    # Coordinates comoving kpc/h, velocities km/s sqrt(a), masses 1e10 Msun/h; Metallicity columns (Z, He, C, N, O, Ne, Mg, Si, S, Ca, Fe).
    # Hydrogen mass fraction X = 1 - Z - Y (columns 0 and 1).
    return dict(x=(x / h - c) * a, v=v * np.sqrt(a) - vc, m=m * 1e10 / h, m_formed=np.full(len(m), np.nan),
                feh=_feh(Z[:, 10], 1 - Z[:, 0] - Z[:, 1]), age=lookback(af, h, hdr["Omega0"]) - lookback(a, h, hdr["Omega0"]))


def masses(suite, name):
    """All particles (gas, dark matter, stars, black holes) of a zoom analogue within R_MAX of the centre used by _read: x (N, 3) [kpc,
    simulation axes] and m [Msun], for the analogue's potential (paper Section 5). Auriga and FIRE-2 only: the TNG50 cutouts hold the
    stars alone. Auriga's high-resolution dark matter (PartType1) takes its mass from the header MassTable.

    Raises ValueError for suite 'TNG50'. Unit conversions as in _read (comoving Mpc/h or kpc/h to physical kpc, 1e10 Msun/h to
    Msun)."""
    xs, ms = [], []
    if suite == "Auriga":
        d = AURIGA[name]
        for fn in sorted(glob.glob(str(d / "snapdir_251/snapshot_251.*.hdf5"))):
            with h5py.File(fn) as f:
                hdr = dict(f["Header"].attrs)
                for t in range(6):
                    if f"PartType{t}" in f:
                        g = f[f"PartType{t}"]
                        xs.append(g["Coordinates"][:])
                        ms.append(g["Masses"][:] if "Masses" in g else np.full(len(xs[-1]), hdr["MassTable"][t]))
        h, a = hdr["HubbleParam"], hdr["Time"]
        with h5py.File(sorted(glob.glob(str(d / "groups_251/fof_subhalo_tab_251.*.hdf5")))[0]) as f:
            c = f["Subhalo/SubhaloPos"][0]
        x, m = (np.concatenate(xs) - c) * 1e3 * a / h, np.concatenate(ms) * 1e10 / h
    elif suite == "FIRE-2":
        d = FIRE[name]
        with h5py.File(d / "output/snapshot_600.hdf5") as f:
            hdr = dict(f["Header"].attrs)
            for t in range(6):
                if f"PartType{t}" in f and "Coordinates" in f[f"PartType{t}"]:
                    xs.append(f[f"PartType{t}/Coordinates"][:])
                    ms.append(f[f"PartType{t}/Masses"][:])
        h, a = hdr["HubbleParam"], hdr["Time"]
        with h5py.File(d / "track/host_coordinates.hdf5") as f:
            c = f["host.position"][600, 0]
        x, m = (np.concatenate(xs) / h - c) * a, np.concatenate(ms) * 1e10 / h
    else:
        raise ValueError(f"{suite}: no particles other than stars")
    keep = np.linalg.norm(x, axis=1) < R_MAX
    return x[keep], m[keep]


def birth(suite, name):
    """Star particles of a TNG50 or Auriga analogue with their z = 0 phase space and their birth position and velocity (paper
    Section 6, Fig. sims_birth).

    Returns a dict of arrays over the star particles (wind particles dropped):
    x, v : (N, 3) z = 0 position [physical kpc] and velocity [km/s] relative to the main subhalo (as _read);
    m : weight [Msun] (Auriga: GFM_InitialMass; TNG50: present mass, the cutouts hold no initial mass); age : lookback time to
    formation [Gyr]; a_form : scale factor at formation; birth_x : (N, 3) BirthPos [comoving kpc; TNG50: unwrapped about the
    subhalo's z = 0 position in the periodic box]; birth_v : (N, 3) peculiar birth velocity [km/s];
    exsitu : Auriga: True for particles in any Exsitu list of lists/accretedstardata (formed in another subhalo); TNG50: all False
    (no assembly catalogue on disk).
    Units: BirthPos is comoving like Coordinates. BirthVel is Arepo's internal velocity a^2 dx/dt in both suites, so the peculiar
    velocity is BirthVel / a, not the snapshot convention Velocities x sqrt(a) (tests/test_sims.py: the progenitor's centre moves
    through the box with the median BirthVel / a of the stars it forms).
    TNG50 birth fields come from TNG_BIRTH_DIR/birth_<id>.hdf5 (scripts/fetch_tng50.py), whose particles are in the order of the
    z = 0 cutout.
    """
    if suite == "TNG50":
        d = _read(suite, name)
        with h5py.File(TNG_BIRTH_DIR / f"birth_{name}.hdf5") as f:
            g = f["PartType4"]
            af = g["GFM_StellarFormationTime"][:]
            ok = af > 0
            bx, bv = g["BirthPos"][:][ok], g["BirthVel"][:][ok]
        sub = {str(s["id"]): s for s in json.load(open(TNG_DIR / "sample.json"))["subhalos"]}[name]
        c = np.array([sub["pos_x"], sub["pos_y"], sub["pos_z"]])
        bx = ((bx - c + TNG_BOX / 2) % TNG_BOX - TNG_BOX / 2) / TNG_H  # unwrapped about c, ckpc/h -> comoving kpc
        return dict(x=d["x"], v=d["v"], m=d["m"], age=d["age"], a_form=af[ok], birth_x=bx, birth_v=bv / af[ok, None],
                    exsitu=np.zeros(ok.sum(), bool))
    d = AURIGA[name]
    parts = []
    for fn in sorted(glob.glob(str(d / "snapdir_251/snapshot_251.*.hdf5"))):
        with h5py.File(fn) as f:
            hdr = dict(f["Header"].attrs)
            if "PartType4" in f:
                g = f["PartType4"]
                parts.append([g[k][:] for k in ("ParticleIDs", "GFM_StellarFormationTime", "GFM_InitialMass", "Coordinates",
                                                "Velocities", "BirthPos", "BirthVel")])
    ids, af, mi, x, v, bx, bv = (np.concatenate(p) for p in zip(*parts))
    h, a, om = hdr["HubbleParam"], hdr["Time"], hdr["Omega0"]
    with h5py.File(sorted(glob.glob(str(d / "groups_251/fof_subhalo_tab_251.*.hdf5")))[0]) as f:
        c, vc = f["Subhalo/SubhaloPos"][0], f["Subhalo/SubhaloVel"][0]
    ex = [np.empty(0, np.uint64)]  # ParticleIDs of the ex-situ stars, from every file of the accreted-star lists
    for fn in glob.glob(str(d.parent / "lists/accretedstardata" / d.name / "*.hdf5")):
        with h5py.File(fn) as f:
            if "Exsitu" in f:
                ex.append(f["Exsitu/ParticleIDs"][:])
    ok = af > 0
    return dict(x=(x[ok] - c) * 1e3 * a / h, v=v[ok] * np.sqrt(a) - vc, m=mi[ok] * 1e10 / h,
                age=lookback(af[ok], h, om) - lookback(a, h, om), a_form=af[ok], birth_x=bx[ok] * 1e3 / h,
                birth_v=bv[ok] / af[ok, None], exsitu=np.isin(ids[ok], np.concatenate(ex)))


def progenitor_track(birth_x, a_form, birth_v, group=60, smooth=5):
    """Position [comoving kpc] and peculiar velocity [km/s] of the galaxy the stars formed in, as functions of the scale factor.

    The stars are sorted by formation time and cut into groups of `group`; each group's centre is found by shrinking spheres (to
    the innermost 20 per cent) and its velocity is the median birth velocity of its inner half. Both are median-filtered over
    `smooth` groups (a group's few stars born in a companion do not move the track). Returns f(a) -> (centre (N, 3), velocity (N, 3)),
    linear interpolation between groups. A track, rather than one frame per group, because TNG50's particles are heavy: a group of
    60 spans tens of Myr, during which the progenitor moves several comoving kpc.
    """
    s = np.argsort(a_form, kind='stable')  # tied formation times keep the order of the particle files, on every machine
    A, C, V = [], [], []
    for j in range(0, len(s) - group + 1, group):
        g = s[j:j + group]
        p = birth_x[g]
        c = np.median(p, axis=0)
        for frac in np.geomspace(1, 0.2, 12):  # shrinking spheres: keep the inner frac of the group, 100 to 20 per cent
            r = np.linalg.norm(p - c, axis=1)
            c = p[r <= np.quantile(r, frac)].mean(0)
        r = np.linalg.norm(p - c, axis=1)
        A.append(np.median(a_form[g])); C.append(c); V.append(np.median(birth_v[g][r < np.median(r)], axis=0))
    A, C, V = np.array(A), median_filter(np.array(C), (smooth, 1), mode="nearest"), median_filter(np.array(V), (smooth, 1), mode="nearest")
    return lambda a: tuple(np.stack([np.interp(a, A, Y[:, k]) for k in range(3)], axis=1) for Y in (C, V))


def birth_phase_space(birth_x, a_form, birth_v, m, track, r_spin=3.0, r_min=0.5):
    """Birth positions [physical kpc] and velocities [km/s] relative to the progenitor (track from progenitor_track), rotated so that
    z is along the angular momentum of the stars within r_spin. Returns X, V (N, 3) and the mask of the stars to measure: within
    r_spin and at cylindrical R > r_min. The inner cut: the track's centre is uncertain by ~0.15 kpc (60-star groups), which
    scrambles the azimuth of the innermost stars; with it a synthetic disc's v_phi is recovered (tests/test_sims.py: 35.0 of 40 km/s
    without, 39.8 with; sigma_z is not affected)."""
    c, vc = track(a_form)  # progenitor centre [comoving kpc] and velocity [km/s] at each star's formation
    P, B = (birth_x - c) * a_form[:, None], birth_v - vc  # comoving -> physical kpc with the formation scale factor
    k = np.linalg.norm(P, axis=1) < r_spin
    L = (m[k, None] * np.cross(P[k], B[k])).sum(0)
    ez = L / np.linalg.norm(L)
    ex = np.cross([0.0, 0.0, 1.0], ez) if abs(ez[2]) < 0.99 else np.array([1.0, 0.0, 0.0]) - ez[0] * ez  # perpendicular to ez
    ex /= np.linalg.norm(ex)
    rot = np.stack([ex, np.cross(ez, ex), ez])  # rows: new x, y, z axes
    P, B = P @ rot.T, B @ rot.T
    return P, B, k & (np.hypot(P[:, 0], P[:, 1]) > r_min)


def main_branch(name):
    """Main progenitor branch of TNG50 analogue `name` from its SubLink tree (TNG_TREE_DIR, scripts/fetch_tng50_progenitors.py):
    the rows from the z = 0 subhalo down its first progenitors to the main leaf. Returns a dict of arrays: SnapNum, SubfindID,
    SubhaloPos [ckpc/h], SubhaloVel [km/s, peculiar], Mstar (SubhaloMassType of stars [Msun])."""
    with h5py.File(TNG_TREE_DIR / f"sublink_{name}.hdf5") as t:
        # SubLink depth-first order: the main branch is the rows from the root (row 0) to its MainLeafProgenitorID.
        end = np.flatnonzero(t["SubhaloID"][:] == t["MainLeafProgenitorID"][0])[0] + 1
        b = {k: t[k][:end] for k in ("SnapNum", "SubfindID", "SubhaloPos", "SubhaloVel")}
        b["Mstar"] = t["SubhaloMassType"][:end, 4] * 1e10 / TNG_H
    return b


def progenitor(name, snap, kind, branch=None):
    """Particles of the main progenitor of TNG50 analogue `name` at snapshot `snap`, from the cutout of `kind`: 'track' (stars at
    every third snapshot, TNG_TRACK_DIR), 'gas' (TNG_GAS_DIR), 'full' (stars, gas and dark matter at every snapshot, TNG_FULL_DIR) or
    'dm' (dark matter elsewhere, TNG_DM_DIR). Positions relative to the branch's SubhaloPos (periodic box) in physical kpc, velocities
    peculiar (snapshot convention, x sqrt(a)) relative to its SubhaloVel. Returns None if the file is missing; else dict(a, lookback
    [Gyr], stars=..., gas=..., dm=...) with, per part type present: stars x, v (N, 3), m [Msun], ids, age_form (lookback to formation
    [Gyr], wind particles dropped); gas x, v, m, sfr, u (InternalEnergy, (km/s)^2); dm x and m (the header's particle mass).
    branch: main_branch(name), read if not given."""
    path = {"track": TNG_TRACK_DIR / name / f"snap_{snap:03d}.hdf5", "gas": TNG_GAS_DIR / name / f"gas_{snap:03d}.hdf5",
            "full": TNG_FULL_DIR / name / f"snap_{snap:03d}.hdf5", "dm": TNG_DM_DIR / name / f"snap_{snap:03d}.hdf5"}[kind]
    if not path.exists():
        return None
    b = branch or main_branch(name)
    j = np.flatnonzero(b["SnapNum"] == snap)[0]
    out = {}
    with h5py.File(path) as f:
        a = f["Header"].attrs["Time"]
        out.update(a=a, lookback=float(lookback(a, TNG_H, TNG_OM)))
        if "PartType1" in f:
            x = ((f["PartType1/Coordinates"][:] - b["SubhaloPos"][j] + TNG_BOX / 2) % TNG_BOX - TNG_BOX / 2) * a / TNG_H
            out["dm"] = dict(x=x, m=np.full(len(x), f["Header"].attrs["MassTable"][1] * 1e10 / TNG_H))
        for key, part in (("stars", "PartType4"), ("gas", "PartType0")):
            if part not in f or kind == "dm":   # the 'dm' cutouts of the FULL sample also hold gas positions only (unused)
                continue
            g = f[part]
            x = ((g["Coordinates"][:] - b["SubhaloPos"][j] + TNG_BOX / 2) % TNG_BOX - TNG_BOX / 2) * a / TNG_H
            d = dict(x=x, v=g["Velocities"][:] * np.sqrt(a) - b["SubhaloVel"][j], m=g["Masses"][:] * 1e10 / TNG_H)
            if key == "stars":
                af = g["GFM_StellarFormationTime"][:]
                ok = af > 0
                d = {k: q[ok] for k, q in d.items()} | dict(ids=g["ParticleIDs"][:][ok], age_form=lookback(af[ok], TNG_H, TNG_OM))
            else:
                d |= dict(sfr=g["StarFormationRate"][:], u=g["InternalEnergy"][:])
            out[key] = d
    return out


def coherence(x, v, m):
    """Orbital-plane coherence C = |sum m j| / sum m |j| (j = x cross v) of particles: 1 when all orbit in one plane with one sense,
    ~ N^-1/2 for N randomly oriented orbits (tests/test_sims.py).

    x, v : (N, 3) positions [kpc] and velocities [km/s] about the centre; m : (N,) masses (any unit). Returns a float in [0, 1].
    """
    j = np.cross(x, v)
    return np.linalg.norm((m[:, None] * j).sum(0)) / (m * np.linalg.norm(j, axis=1)).sum()


def disc_frame(x, v, m):
    """Positions and velocities (N, 3) relative to the centre of velocity of the stars within ALIGN_R, rotated so that z is along
    their angular momentum (x: the simulation x axis projected on the plane, or its own x when the spin is along z). Returns X, V and
    the rotation matrix rot (X = x @ rot.T).

    x : (N, 3) [kpc] about the galaxy's centre; v : (N, 3) [km/s]; m : (N,) [Msun], the weights of the centre of velocity and of
    the angular momentum. X [kpc], V [km/s] (N, 3); rot (3, 3), rows the new x, y, z axes in simulation coordinates.
    """
    k = np.linalg.norm(x, axis=1) < ALIGN_R  # stars that define the centre of velocity and the spin axis
    v = v - np.average(v[k], axis=0, weights=m[k])
    L = (m[k, None] * np.cross(x[k], v[k])).sum(0)
    ez = L / np.linalg.norm(L)
    ex = np.cross([0.0, 0.0, 1.0], ez) if abs(ez[2]) < 0.99 else np.array([1.0, 0.0, 0.0]) - ez[0] * ez  # perpendicular to ez
    ex /= np.linalg.norm(ex)
    rot = np.stack([ex, np.cross(ez, ex), ez])
    return x @ rot.T, v @ rot.T, rot


def place(X, V, p):
    """Sky positions and observables of disc-frame particles placed with the centre, orientation, distance and systemic motion of the
    kinfit parameter vector p (dx, dy, pmra0, pmdec0, inc, theta, vlos0; distance LMC['D0']).

    Parameters
    ----------
    X, V : ndarray, shape (N, 3)
        Disc-frame positions [kpc] and velocities [km/s] (disc_frame()).
    p : array_like, shape (33,)
        kinfit parameter vector (lmcold.kinfit.PARAMS order); only dx, dy, pmra0, pmdec0, inc, theta and vlos0 are used.

    Returns dict ra, dec [deg], dist [kpc], pmra, pmdec [mas/yr], vlos [km/s], arrays of shape (N,)."""
    ra0, dec0 = (float(c) for c in kinfit.centre(p[0], p[1]))
    r0, eE0, eN0, line, n = (np.asarray(u) for u in geo.symmetry_axis(ra0, dec0, np.deg2rad(p[4]), np.deg2rad(p[5])))
    perp = np.cross(n, line)
    # Heliocentric position [kpc]: disc x along the line of nodes, y along n x line, z along the normal n.
    pos = LMC["D0"] * r0 + X[:, :1] * line + X[:, 1:2] * perp + X[:, 2:3] * n
    dist = np.linalg.norm(pos, axis=1)
    rhat = pos / dist[:, None]
    ra, dec = np.arctan2(rhat[:, 1], rhat[:, 0]) % (2 * np.pi), np.arcsin(np.clip(rhat[:, 2], -1, 1))
    vsys = p[11] * r0 + KAPPA * LMC["D0"] * (p[2] * eE0 + p[3] * eN0)  # as kinfit._model
    Vt = vsys + V[:, :1] * line + V[:, 1:2] * perp + V[:, 2:3] * n  # total heliocentric velocity [km/s]
    _, ea, ed = (np.asarray(u) for u in geo.sky_basis(ra, dec))
    return dict(ra=np.rad2deg(ra), dec=np.rad2deg(dec), dist=dist, pmra=(Vt * ea).sum(1) / (KAPPA * dist),
                pmdec=(Vt * ed).sum(1) / (KAPPA * dist), vlos=(Vt * rhat).sum(1))


def on_grid(table, column, feh, age):
    """Bilinear interpolation of table[column] on its (mh, logage) grid at [M/H] = feh and log10(age [yr]), clamped at the edges.

    Used for the giants per formed mass n_box (results/giant_conversion.csv, BOSS) and the present mass per formed mass
    (lmcold.surviving_mass.survival_grid).

    Parameters
    ----------
    table : pandas.DataFrame
        Long table with columns mh [dex], logage [log10 yr] and `column`, forming a complete regular grid.
    column : str
        Column to interpolate.
    feh : array_like
        [M/H] [dex] of the points.
    age : array_like
        Ages [Gyr], same shape as feh (floored at 1 Myr before the logarithm).

    Returns
    -------
    ndarray
        Interpolated values, same shape as feh.
    """
    from scipy.interpolate import RegularGridInterpolator
    G = table.pivot_table(index="mh", columns="logage", values=column)
    f = RegularGridInterpolator((G.index.to_numpy(), G.columns.to_numpy()), G.to_numpy())
    x = np.clip(feh, G.index.min(), G.index.max())
    y = np.clip(np.log10(np.clip(age, 1e-3, None)) + 9, G.columns.min(), G.columns.max())  # Gyr -> log10(age [yr])
    return f(np.stack([x, y], -1))


def lmc_scale(feh, w, lmc_feh, lmc_cum):
    """[Fe/H] on the LMC scale by matching cumulative formed mass: a particle below which a fraction F of the analogue's formed mass
    lies gets the LMC [Fe/H] below which the same fraction of the LMC's formed mass lies.

    Parameters
    ----------
    feh, w : ndarray
        Particle [Fe/H] and formed mass (the analogue's population whose mass fractions are matched).
    lmc_feh, lmc_cum : ndarray
        LMC [Fe/H] knots and the cumulative formed-mass fraction below them (increasing, from 0 to 1); linear between knots.

    Returns
    -------
    ndarray
        [Fe/H] [dex] of each particle on the LMC scale, same shape as feh.
    """
    o = np.argsort(feh, kind="stable")
    F = np.empty(len(feh))
    F[o] = (np.cumsum(w[o]) - 0.5 * w[o]) / w.sum()  # mid-point cumulative fraction of each particle
    return np.interp(F, lmc_cum, lmc_feh)
