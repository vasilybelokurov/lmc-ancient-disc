"""Rotation and vertical dispersion at birth of the stars of the three FIRE-2 LMC-mass analogues (m11d, m11e, m11h at 7100 Msun;
Wetzel et al. 2023, ApJS 265, 44), for the Letter's FIRE-2 number (scripts/paper_text_letter.py, FireRbMed) and Section 6.4 of the
extended draft (scripts/paper_text_sims_suites.py): results/research/fire_early_birth.csv.

Input (scripts/fetch_fire2.py): output/snapshot_600.hdf5 (z = 0 star particles: formation scale factor, mass) and
track/host_coordinates.hdf5, whose star.form.host.distance and star.form.host.velocity give every z = 0 star's position [physical
kpc] and velocity [km/s] at formation relative to the host galaxy, in the host's principal-axis frame at that time (Wetzel et al.
2023, Sect. 4.7). Cosmology (h, Omega_m) from the snapshot header.

The stars are grouped by lookback time at formation (lmcold.sims.lookback) into cohorts of WIDTHS Gyr from LB_MIN to LB_MAX Gyr. For
each cohort with at least MIN_SPIN stars within R_SPIN kpc at birth, the birth vectors are rotated to the spin axis of those stars;
then the mass-weighted mean v_phi and sigma_z (about the mass-weighted mean v_z) of the cohort's stars at R_IN < R < R_OUT kpc
(cylindrical, in the rotated frame), as for the TNG50 and Auriga birth values of Fig. sims_birth (scripts/build_sims_birth.py).
tilt [deg]: angle between that spin axis and the z axis of the host's principal-axis frame.
Writes results/research/fire_early_birth.csv: halo, width [Gyr], lb [Gyr, cohort centre], n (stars at R_IN < R < R_OUT), vphi, sz
[km/s], tilt [deg].

Run: .venv/bin/python scripts/manifest.py run fire2_birth.py   (seconds per analogue; 2 s in the recorded run of docs/run_times.txt)
"""
# %% [markdown]
# ### Settings
# The cohort widths, the range of look-back times, and the radii of the spin axis and of the measurement.

# %%
import h5py
import numpy as np
import pandas as pd

from lmcold import sims
from lmcold.sources import ROOT

WIDTHS = (0.5, 0.1)          # Gyr, cohort widths (0.1 is the one quoted; 0.5 as the early TNG50 cohorts)
LB_MIN, LB_MAX = 7.5, 13.5   # Gyr, range of lookback times at formation
R_SPIN, MIN_SPIN = 3.0, 30   # kpc and stars: the spin axis is that of the cohort's stars within R_SPIN at birth
R_IN, R_OUT = 0.5, 3.0       # kpc, cylindrical annulus of the measurement


# %% [markdown]
# ### Helpers
# `spin_frame` rotates to the spin of a set of stars, `vphi_sigma_z` measures rotation and vertical dispersion in the annulus, and
# `cohorts` reads one analogue and measures each of its cohorts at birth.

# %%
def spin_frame(x, v, m):
    """Rotation matrix (rows: new x, y, z axes) with z along the total angular momentum of particles x, v [kpc, km/s], m.

    x, v: (N, 3); m: (N,) [Msun]. Returns a (3, 3) matrix R; positions in the new frame are x @ R.T."""
    L = (m[:, None] * np.cross(x, v)).sum(0)   # total angular momentum [Msun kpc km/s]
    ez = L / np.linalg.norm(L)   # new z: along L
    # New x: perpendicular to both the old z and the new z (along the line of nodes); new y completes a right-handed frame.
    ex = np.cross([0., 0., 1.], ez)
    ex /= np.linalg.norm(ex)
    return np.stack([ex, np.cross(ez, ex), ez])


def vphi_sigma_z(X, V, m):
    """Mass-weighted mean v_phi and sigma_z [km/s] and the number of particles at R_IN < R < R_OUT.

    X [kpc], V [km/s]: (N, 3) in a frame with the spin along +z; m: (N,) [Msun]."""
    R = np.hypot(X[:, 0], X[:, 1])   # kpc, cylindrical radius
    k = (R > R_IN) & (R < R_OUT)   # particles in the annulus
    # Azimuthal velocity (x v_y - y v_x) / R, and v_z about its mass-weighted mean [km/s].
    vphi = (X[k, 0] * V[k, 1] - X[k, 1] * V[k, 0]) / R[k]
    vz = V[k, 2] - np.average(V[k, 2], weights=m[k])
    return np.average(vphi, weights=m[k]), np.sqrt(np.average(vz ** 2, weights=m[k])), int(k.sum())


def cohorts(name):
    """Rows of the table for FIRE-2 analogue name (a key of lmcold.sims.FIRE)."""
    d = sims.FIRE[name]   # folder of the analogue
    # z = 0 snapshot: cosmology from the header, then the star particles (PartType4).
    with h5py.File(d / 'output/snapshot_600.hdf5') as f:
        h, om = f['Header'].attrs['HubbleParam'], f['Header'].attrs['Omega0']
        af, m = f['PartType4/StellarFormationTime'][:], f['PartType4/Masses'][:] * 1e10 / h   # scale factor at formation; mass [Msun]
    with h5py.File(d / 'track/host_coordinates.hdf5') as f:
        x, v = f['star.form.host.distance'][:], f['star.form.host.velocity'][:]   # (N, 3) at formation [kpc, km/s], host frame
    lb = sims.lookback(af, h, om)   # Gyr, lookback time of formation
    near = np.linalg.norm(x, axis=1) < R_SPIN   # stars born within R_SPIN [kpc, 3D distance] of the host centre
    rows = []   # one dict per (width, cohort)
    for width in WIDTHS:
        # Cohorts [lo, lo + width) in lookback time [Gyr].
        for lo in np.arange(LB_MIN, LB_MAX, width):
            c = (lb >= lo) & (lb < lo + width)
            # Skip a cohort with fewer than MIN_SPIN stars to define the spin axis.
            if (c & near).sum() < MIN_SPIN:
                continue
            # Spin axis of the cohort's central stars; all cohort stars rotated to it and measured in the annulus.
            rot = spin_frame(x[c & near], v[c & near], m[c & near])
            vphi, sz, n = vphi_sigma_z(x[c] @ rot.T, v[c] @ rot.T, m[c])
            # tilt [deg]: angle between the new z axis (row 2 of rot) and the host frame's z axis.
            rows.append(dict(halo=d.name, width=width, lb=lo + width / 2, n=n, vphi=vphi, sz=sz,
                             tilt=np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))))
    return rows


if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data and measure the three analogues
    # Each analogue's snapshot and host-coordinate file is read inside `cohorts`; the rows of all three go into one table.

    # %%
    # One row per (analogue, width, cohort), all three analogues.
    t = pd.DataFrame([r for name in sims.FIRE for r in cohorts(name)])
    t.to_csv(ROOT / 'results/research/fire_early_birth.csv', index=False, float_format='%.4f')
