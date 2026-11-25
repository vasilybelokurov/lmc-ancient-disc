"""OGLE-IV survey footprint around the Magellanic Clouds: the union of the field rectangles (Udalski et al. 2015).

Each field is modelled as a rectangle aligned with RA and Dec, centred on the published field centre, HALF_WIDTH x HALF_HEIGHT
degrees in the field's tangent plane (published maximal width 1 deg 23 arcmin along RA and height 1 deg 14 arcmin 10 arcsec
along Dec; the real mosaic is narrower at its corners, which the empirical check in scripts/paper_rrab_selection.py measures).

Inputs: data/sources.yaml keys ogle4_fields_lmc and ogle4_fields_mbr (scripts/fetch_ogle4_fields.py). Used for the RR Lyrae
completeness (scripts/paper_selection.py, paper_selection_checks.py, check_rrab_independence.py): OGLE's completeness is
measured only inside its footprint.
"""
import numpy as np
import pandas as pd

from . import geometry as geo
from .sources import path

# Half-sizes [deg]: half of 1 deg 23 arcmin (RA) and of 1 deg 14 arcmin 10 arcsec (Dec).
HALF_WIDTH, HALF_HEIGHT = 0.5 * (1 + 23 / 60), 0.5 * (1 + 14 / 60 + 10 / 3600)  # deg


def field_centres():
    """(ra, dec) [deg] of the OGLE-IV LMC and Magellanic Bridge fields.

    Returns
    -------
    pandas.DataFrame
        Columns ra, dec [deg], one row per field.
    """
    # Field lines of the OGLE text files are indented by two spaces; columns 7 and 8 are RA [h] and Dec [deg].
    rows = []
    for name in ("ogle4_fields_lmc", "ogle4_fields_mbr"):
        for line in path(name).read_text().splitlines():
            if line.startswith("  ") and not line.lstrip().startswith("#"):
                p = line.split()
                rows.append((15 * float(p[6]), float(p[7])))
    return pd.DataFrame(rows, columns=["ra", "dec"])


def in_footprint(ra, dec, fields=None):
    """True for positions (deg) inside any OGLE-IV field rectangle.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    fields : pandas.DataFrame, optional
        Field centres (columns ra, dec [deg]); field_centres() if None.

    Returns
    -------
    ndarray of bool, shape (N,)
    """
    f = field_centres() if fields is None else fields
    ra, dec = np.atleast_1d(ra), np.atleast_1d(dec)
    rhat = np.asarray(geo.sky_basis(np.deg2rad(ra), np.deg2rad(dec))[0])
    inside = np.zeros(len(ra), bool)
    for fra, fdec in zip(f.ra, f.dec):
        r0, eE, eN = (np.asarray(v) for v in geo.sky_basis(np.deg2rad(fra), np.deg2rad(fdec)))
        z = rhat @ r0
        xi, eta = np.rad2deg(rhat @ eE / z), np.rad2deg(rhat @ eN / z)  # gnomonic projection about the field centre
        inside |= (z > 0) & (np.abs(xi) < HALF_WIDTH) & (np.abs(eta) < HALF_HEIGHT)
    return inside
