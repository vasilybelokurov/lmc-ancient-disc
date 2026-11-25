"""True reddening for the stellar-population conversion (not the targeting reddening of the BOSS carton).

Inside the OGLE-IV red-clump map of Skowron et al. (2021, ApJS 252, 23; VizieR J/ApJS/252/23, scripts/fetch_skowron21.py):
E(V-I) of the box containing the star, on the SFD E(B-V) scale with E(V-I) / E(B-V) = 2.742 - 1.505 = 1.237 (Schlafly &
Finkbeiner 2011, Table 6, Landolt V and I, R_V = 3.1), the scale of the A_V = 0.86 x 3.1 E(B-V) used throughout. Outside the
map: SFD, which the carton also used unscaled there (lmcold.magcloud).

Input: data/sources.yaml key skowron21_table1 (81,684 boxes of 3.4-27.5 arcmin). Used by lmcold.mass_samples and the mass
scripts (scripts/build_giant_mass.py, paper_mass_method.py, paper_parent_extinction.py, ...) to give the true reddening of a star
or region for lmcold.isochrones.count_grid.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .sources import path

EVI_PER_EBV = 2.742 - 1.505  # E(V-I) / E(B-V) on the SFD scale: A_V/E(B-V) - A_I/E(B-V), Schlafly & Finkbeiner 2011 Table 6


def rc_map():
    """Skowron et al. (2021) boxes: ra, dec [deg], box [arcmin, side], evi [mag].

    Returns
    -------
    pandas.DataFrame
        One row per box: ra, dec [deg] of the box centre, box [arcmin] side of the square box, evi = E(V-I) [mag].
    """
    t = pd.read_fwf(path("skowron21_table1"), colspecs=[(0, 4), (5, 14), (15, 25), (26, 31)], names=["box", "rah", "dec", "evi"],
                    header=None, compression="gzip")  # bytes 1-4, 6-14, 16-25, 27-31 of the CDS ReadMe
    return t.assign(ra=15 * t.rah)[["ra", "dec", "box", "evi"]]  # RA given in hours


def _xyz(ra, dec):
    """Unit vectors, shape (N, 3), of positions ra, dec [deg]."""
    r, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def true_ebv(ra, dec, ebv_sfd, m):
    """SFD-scale E(B-V): the RC map m (rc_map()) where one of its square boxes contains the position, ebv_sfd elsewhere.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    ebv_sfd : array_like
        SFD (Schlegel et al. 1998) E(B-V) [mag] at the positions, used outside the map.
    m : pandas.DataFrame
        The map, rc_map().

    Returns
    -------
    ndarray
        E(B-V) [mag]: E(V-I) / EVI_PER_EBV of the containing box, else ebv_sfd.
    """
    # Box whose centre is nearest on the sphere; the star is then tested against that box's square only.
    _, j = cKDTree(_xyz(m.ra.to_numpy(), m.dec.to_numpy())).query(_xyz(ra, dec))
    c = m.iloc[j]
    dx = (np.asarray(ra) - c.ra.to_numpy() + 180) % 360 - 180  # RA offset [deg], wrapped to [-180, 180)
    # Inside: |offset| <= half the box side [arcmin] along RA (times cos dec) and along Dec.
    inside = (np.abs(dx) * np.cos(np.radians(dec)) * 60 <= c.box.to_numpy() / 2) & (np.abs(dec - c.dec.to_numpy()) * 60 <= c.box.to_numpy() / 2)
    return np.where(inside, c.evi.to_numpy() / EVI_PER_EBV, ebv_sfd)
