"""APOGEE-2 targeting of the LMC red giants (fields LMC1-35).

Nidever et al. (2020, arXiv:1901.03448, Sect. 2), Santana et al. (2021, arXiv:2108.11908, Sect. 4.5) and the IDL code
svn.sdss.org/public/repo/apogee/apogeetarget/trunk/pro/mcs/select_mc_targets.pro. In observed 2MASS (J - Ks, H):
  BrtRGB  (0.5632, 12.35), (0.55455, 12.8), (1.2488, 12.8), (1.2834, 12.35); drawn at random;
  FntRGB  (0.55455, 12.8), (0.5219, maglim), (1.1182, maglim), (1.2488, 12.8); filled brightest first.
In LMC1-17 both also needed the Washington + DDO51 giant flag, which DR17 does not publish (WASH_DDO51_GIANT_FLAG = -1).
Plate design dropped stars with a >= 2 mag brighter (H) neighbour within 6 arcsec.

Applied to the apogee2Object_LMC<n> target files and the plate/field lists (data/sources.yaml keys apogee2_object_lmc1..35,
apogee2_field). Used by scripts/build_apogee_parent.py (the APOGEE parent catalogue: candidates in the polygons, Gaia-matched)
and scripts/apogee_selection.py (weights; figures/paper_selection.png (c)); sides() is also used for the APOGEE giants per formed
mass (lmcold.isochrones.count_grid).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import giants, selection

FIELDS = [f"LMC{i}" for i in range(1, 36)]
RADIUS = 0.95  # deg, largest RGB targeting radius of the LMC fields (0.8 deg on some earlier plates)
# FntRGB faint limit: plan files (svn .../mcs/plan/LMC<n>.plan) give 14.5, and 14.9 for LMC2 and LMC12; LMC18-35 have no public
# plan file, and their targets reach H = 15.29, consistent with the 15.3 of Povick et al. (2024, arXiv:2306.06348).
MAGLIM = {**{f"LMC{i}": 14.5 for i in range(1, 18)}, "LMC2": 14.9, "LMC12": 14.9, **{f"LMC{i}": 15.3 for i in range(18, 36)}}
TOL = 0.005  # mag: 2MASS magnitudes are quoted to 1e-3 and a few targets sit up to 0.003 mag outside the blue edge


def sides(h, maglim):
    """Blue and red J - Ks edges of the BrtRGB + FntRGB polygon at H.

    Parameters
    ----------
    h : array_like
        Observed 2MASS H [mag].
    maglim : float or array_like
        FntRGB faint limit [mag] (MAGLIM of the field).

    Returns
    -------
    blue, red : ndarray
        J - Ks [mag] edges, linear between the vertices of the module docstring (BrtRGB above H = 12.8, FntRGB below).
        Not limited in H: rgb_class applies the H range.
    """
    h = np.asarray(h, float)
    f = np.clip((h - 12.8) / (np.asarray(maglim) - 12.8), 0, None)
    blue = np.where(h < 12.8, 0.5632 + (0.55455 - 0.5632) * (h - 12.35) / 0.45, 0.55455 + (0.5219 - 0.55455) * f)
    red = np.where(h < 12.8, 1.2834 + (1.2488 - 1.2834) * (h - 12.35) / 0.45, 1.2488 + (1.1182 - 1.2488) * f)
    return blue, red


def rgb_class(jk, h, maglim, tol=TOL):
    """0 outside, 1 BrtRGB (12.35 <= H < 12.8), 2 FntRGB (12.8 <= H <= maglim).

    Parameters
    ----------
    jk, h : array_like
        Observed 2MASS J - Ks and H [mag].
    maglim : float or array_like
        FntRGB faint limit [mag].
    tol : float
        Tolerance [mag] added outside every edge.

    Returns
    -------
    ndarray of int
    """
    blue, red = sides(h, maglim)
    inside = (jk >= blue - tol) & (jk <= red + tol) & (h >= 12.35 - tol) & (h <= np.asarray(maglim) + tol)
    return np.where(~inside, 0, np.where(h < 12.8, 1, 2))


def bright_neighbour(ra, dec, h, radius=6.0, dmag=2.0):
    """True for stars with a neighbour >= dmag brighter in H within radius [arcsec].

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    h : ndarray
        H [mag].
    radius : float
        Search radius [arcsec].
    dmag : float
        Magnitude difference [mag].

    Returns
    -------
    ndarray of bool
    """
    # Pairs within the chord length 2 sin(radius / 2) of unit vectors.
    r, d = np.radians(ra), np.radians(dec)
    xyz = np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    pairs = cKDTree(xyz).query_pairs(2 * np.sin(np.radians(radius / 3600) / 2), output_type="ndarray")
    out = np.zeros(len(h), bool)
    if len(pairs):
        i, j = pairs.T
        out[i[h[j] <= h[i] - dmag]] = True
        out[j[h[i] <= h[j] - dmag]] = True
    return out


def join_gaia(objects, gaia):
    """Attach the nearest Gaia mate to each 2MASS object, keeping unmatched objects.

    Parameters
    ----------
    objects : pandas.DataFrame
        APOGEE objects with apogee_id = '2M' + 2MASS designation.
    gaia : pandas.DataFrame
        Gaia DR3 cross-match rows with tmass (2MASS designation), gsep (separation) and source_id; several rows per tmass allowed.

    Returns
    -------
    pandas.DataFrame
        objects with the Gaia columns of the closest match (NaN / <NA> where none).
    """
    gaia = gaia.sort_values(['gsep', 'source_id']).drop_duplicates('tmass').copy()  # equal separations: lowest source_id
    gaia['source_id'] = gaia.source_id.astype('Int64')  # nullable integer survives a left join
    return objects.assign(tmass=objects.apogee_id.str[2:]).merge(gaia, on='tmass', how='left').drop(columns='tmass')


def weights(c, snr_min, pg, with_pass=True):
    """(quality, covered, w) per candidate. quality = observed and apogee_spectrum_ok(snr_min): a good spectrum, giant or not, as
    for BOSS; non-giants are removed after weighting (a dwarf is a non-member, not a failed spectrum). Covered groups:
    BrtRGB (cls 1) with >= 3 quality stars; FntRGB with >= 2 and observed fraction >= 0.2; any group with >= 1 quality star and a
    quality fraction >= 0.5. w = pass factor (lmcold.selection, pass groups pg) / quality fraction, for quality stars in covered
    groups. Columns of c: group, cls, observed, snr, extratarg, aspcapflag, feh.

    Parameters
    ----------
    c : pandas.DataFrame
        APOGEE parent candidates (one row each); snr NaN for unobserved stars.
    snr_min : float
        S/N threshold of apogee_spectrum_ok.
    pg : ndarray of int
        Pass-group index per candidate (lmcold.selection.pass_groups on H and XP [M/H]).
    with_pass : bool
        Apply the pass factor (False gives the targeting-only weight, for comparison).

    Returns
    -------
    quality : pandas.Series of bool
    covered : ndarray of bool
    w : ndarray
        Weight per candidate, 0 for non-quality or uncovered stars.
    """
    s = c[c.snr.notna()]
    q = c.observed & pd.Series(giants.apogee_spectrum_ok(s, snr_min), s.index).reindex(c.index, fill_value=False)
    g = q.groupby(c.group)
    n_q = g.transform("sum")
    f_obs, f_q = c.groupby("group")["observed"].transform("mean"), n_q / g.transform("size")  # observed and quality fractions
    covered = np.where(c.cls == 1, n_q >= 3, (n_q >= 2) & (f_obs >= 0.2)) | ((f_q >= 0.5) & (n_q >= 1))
    f = selection.pass_factor(pd.factorize(c.group)[0], c.observed.to_numpy(), q.to_numpy(), pg, np.ones(len(c))) if with_pass else 1.0
    return q, covered, np.where(covered & q, f / f_q, 0.0)
