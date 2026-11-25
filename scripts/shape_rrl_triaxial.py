"""Does an in-plane elongation explain the oblate sky-cell fit of the RR Lyrae? (paper Section 5, Appendix; KB chapter 'The shape of
the old population', the arrow of Figure paper_shape_rrl and the mocks of section 'The RR Lyrae cells')

Mock RR Lyrae (lmcold.shape.mock, N stars as in the data footprint, noise floor 0.07 mag, no outliers) of a triaxial K0 body: an
oblate K0 spheroid (scale length H, vertical axis ratio c of C_ALL) squashed along one in-plane axis to BA, that axis turned by PHI from the line
of nodes, placed at the old population's kinematic geometry (inclination and line of nodes of the metal-poor giants, stage-1
ellipsoid fits, results/kin_mcmc_data_ellipsoid.csv). Axis ratios as the RR Lyrae ellipsoid of Cusano et al. (2021, MNRAS 504, 1,
https://doi.org/10.1093/mnras/stab901: 6.5 : 4.6 : 3.7 kpc, b/a 0.71, c/a 0.57) and a milder b/a, with their c/a and the thinner c/a
that the RR Lyrae depth gradient gives at the kinematic inclination (0.41); BA = 1 is the oblate control.
Each mock is fitted with the oblate model of lmcold.shape.fit_cells (all five parameters free), as the data.

Reads: data/rrab.parquet (Zenodo bundle 1; only the number of RR Lyrae in the footprint), results/kin_mcmc_data_ellipsoid.csv
(scripts/fit_kinematics_mcmc.py) and results/shape_rrl_cells.csv (scripts/shape_rrl_cells.py, the data's free fit).
Writes results/shape_rrl_triaxial.csv (c, ba, phi, fitted inc, theta, q, h, delta, q_sky, g; and the data's free fit, last row with
c, ba, phi empty), read by scripts/paper_shape_rrl.py and scripts/paper_text_shape.py.

Run: .venv/bin/python scripts/shape_rrl_triaxial.py   (after scripts/shape_rrl_cells.py)
"""
# %% [markdown]
# ### Settings
# The scale length and vertical axis ratios of the mock bodies, and the cases: the oblate control and two in-plane axis ratios $b/a$,
# each with its major axis at six angles from the line of nodes.

# %%
import numpy as np
import pandas as pd

from lmcold import shape
from lmcold.spheroid import spheroid
from lmcold.sources import ROOT
from shape_rrl_cells import OLD, kin_inc

H = 1.7  # kpc, scale length of the mock K0 bodies
C_ALL = (0.57, 0.41)  # vertical / major axis ratios c/a: Cusano et al. (2021), and q of the RR Lyrae depth gradient at the giants' tilt
CASES = [(1.0, 0.0)] + [(ba, phi) for ba in (0.71, 0.85) for phi in (0, 30, 60, 90, 120, 150)]  # (b/a, PHI [deg]); b/a = 1: oblate control

if __name__ == '__main__':
    # %% [markdown]
    # ### Load the data
    # The geometry of the metal-poor giants from their motions (inclination and line of nodes, deg), and the number of RR Lyrae in the
    # footprint, which each mock reproduces. The data's free fit is read at the end from `results/shape_rrl_cells.csv`.

    # %%
    rng = np.random.default_rng(17)   # fixed seed of all mocks
    inc, theta = (kin_inc('kin_mcmc_data_ellipsoid', OLD, k)[0] for k in ('inc', 'theta'))  # deg
    # RRab members with a distance modulus from the OGLE I-band Wesenheit (dmu_src 'I').
    d = pd.read_parquet(ROOT / 'data/rrab.parquet')
    d = d[d.member & (d.dmu_src == 'I')]
    n = int(shape.in_footprint(d.ra.to_numpy(), d.dec.to_numpy(), d.dmu.to_numpy()).sum())  # RR Lyrae in the footprint

    # %% [markdown]
    # ### Mocks fitted as oblate
    # For each $c/a$ one oblate K0 spheroid is drawn; for each case it is squashed along its in-plane $y$ axis to $b/a$, turned by the
    # angle $\phi$ of the major axis from the line of nodes, observed at the giants' geometry (`lmcold.shape.mock`), counted in the sky
    # cells and fitted with the oblate model. The fitted $(i,q)$ is also expressed as the sky outline $q_{\rm sky}$ and depth gradient $g$.

    # %%
    rows = []
    for cz in C_ALL:
        # shape.PARAMS vector: inc, theta [deg], q, h [kpc], delta = 0 mag, outlier fraction f = 0, noise floor 0.07 mag in each annulus
        p = np.array([inc, theta, cz, H, 0.0, 0.0] + [0.07] * 6)
        base = spheroid(H, cz, 40 * n, rng)  # disc-frame positions (40 n, 3) [kpc], the pool from which shape.mock places the stars
        for ba, phi in CASES:
            c, s = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
            X = base * [1, ba, 1] @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # squashed along y, then turned by phi
            # Mock stars from the pool X at the giants' geometry, binned in sky cells, fitted as oblate: x = CELL_PARAMS values.
            x, ll = shape.fit_cells(shape.cell_stats(*shape.mock(p, n, rng, X=X)))
            rows.append(dict(c=cz, ba=ba, phi=phi, **dict(zip(shape.CELL_PARAMS, x)),
                             **dict(zip(('q_sky', 'g'), shape.projected(x[0], x[2])))))
            print({k: round(float(v), 3) for k, v in rows[-1].items()}, flush=True)

    # %% [markdown]
    # ### The data's fit, for comparison
    # The free oblate fit of the RR Lyrae cells is appended as the last row, with its $q_{\rm sky}$ and $g$.

    # %%
    # Free oblate fit of the data (scripts/shape_rrl_cells.py), appended with c, b/a, phi NaN.
    R = pd.read_csv(ROOT / 'results/shape_rrl_cells.csv')
    f = R[R.fit == 'free'].iloc[0]
    rows.append(dict(c=np.nan, ba=np.nan, phi=np.nan, **{k: f[k] for k in shape.CELL_PARAMS}, **dict(zip(('q_sky', 'g'), shape.projected(f.inc, f.q)))))
    print(f'truth: inc {inc:.1f} theta {theta:.1f}; n {n}')
    # One row per mock (c, b/a, phi, fitted CELL_PARAMS, q_sky, g) and the data row last.
    pd.DataFrame(rows).to_csv(ROOT / 'results/shape_rrl_triaxial.csv', index=False, float_format='%.4f')
