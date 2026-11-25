"""The SDSS-V Magellanic Genesis carton mwm_magcloud_rgb_boss (Nidever et al. 2026, arXiv:2602.02693), re-implemented on Gaia DR3.

Source: github.com/sdss/target_selection, src/target_selection/cartons/mwm_magcloud_rgb.py (class MWM_MagCloud_RGB_BOSS, priority
2819) with the plan 1.0.50 parameters in src/target_selection/config/target_selection.yml (read 2026-10-25). Eligible stars:
  - G < 18.5, BP - RP >= 1, -3 <= pmra, pmdec <= 3 mas/yr, finite astrometry and photometry;
  - not (parallax > 0 and (parallax + 0.025) / parallax_error > 5);
  - proper motion in Magellanic Stream coordinates (Nidever et al. 2008) within 1.2 mas/yr of (pm_L cos B, pm_B) = (1.8, 0.40);
  - dereddened (G, BP - RP) inside the carton's polygon; E(B-V) from SFD98, multiplied by 0.07 where E(B-V) >= 0.1 within 4.5 deg of
    the LMC or 2.5 deg of the SMC (in Magellanic Stream coordinates); A_V = 0.86 x 3.1 E(B-V); A_G, A_BP, A_RP = (0.8509, 1.0901,
    0.5895) A_V;
  - G <= 17.5.
The polygon test uses matplotlib.path.Path.contains_points, as the carton code does.

Used to build the BOSS parent catalogue (scripts/build_boss_parent.py: stars eligible for this carton), the fibre-priority class
of the selection groups (scripts/boss_selection.py, build_boss_cartons.py) and, through the polygon and the carton reddening, the
BOSS giants per formed mass (lmcold.isochrones.count_grid, scripts/build_giant_mass.py).
"""
import numpy as np
from matplotlib.path import Path
from .constants import LMC_TARGET_MS, SMC_TARGET_MS

# Vertices of the carton polygon in dereddened (BP - RP)_0 [mag] and G_0 [mag], in order (carton code, plan 1.0.50).
BPRP_CUT = [0.95, 1.2649, 1.62255, 1.9889, 2.4164, 3.1492, 4.283, 4.30, 3.3411]  # (BP - RP)_0 [mag]
G_CUT = [17.5, 15.8064, 15.2065, 14.6065, 14.0212, 14.6358, 15.6162, 16.24, 17.5]  # G_0 [mag], paired with BPRP_CUT
A_OVER_AV = dict(G=0.8509139481578785, BP=1.0901361069987485, RP=0.5894794093231802)  # A_X / A_V of the carton code
PRIORITY = 2819  # targetdb priority of the carton


def ms_coordinates(ra, dec, pmra, pmdec):
    """Magellanic Stream longitude, latitude [deg] and proper motions pm_L cos B, pm_B [mas/yr] (Nidever et al. 2008 frame).

    Parameters
    ----------
    ra, dec : ndarray
        ICRS positions [deg].
    pmra, pmdec : ndarray
        Proper motions [mas/yr], pmra including cos(dec); must be finite.

    Returns
    -------
    L, B, pmL, pmB : ndarray
        L wrapped to (-180, 180] deg, B [deg], pm_L cos B and pm_B [mas/yr] (gala MagellanicStreamNidever08).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from gala.coordinates import MagellanicStreamNidever08

    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr)
    m = c.transform_to(MagellanicStreamNidever08())
    return m.L.wrap_at(180 * u.deg).deg, m.B.deg, m.pm_L_cosB.to_value(u.mas / u.yr), m.pm_B.to_value(u.mas / u.yr)


def rescaled_ebv(ebv, lmcrad, smcrad):
    """The carton's SFD E(B-V): times 0.07 where E(B-V) >= 0.1 within 4.5 deg of the LMC or 2.5 deg of the SMC.

    Parameters
    ----------
    ebv : array_like
        SFD98 E(B-V) [mag].
    lmcrad, smcrad : array_like
        Distance [deg] in the (L, B) plane of Magellanic Stream coordinates from LMC_TARGET_MS and SMC_TARGET_MS
        (lmcold.constants).

    Returns
    -------
    ndarray
        E(B-V) [mag] the carton used.
    """
    return np.where(((lmcrad <= 4.5) | (smcrad <= 2.5)) & (ebv >= 0.1), 0.07 * ebv, ebv)


def carton_ebv(ra, dec, ebv):
    """The E(B-V) the carton removed at (ra, dec) [deg], from the SFD98 ebv.

    Parameters
    ----------
    ra, dec : array_like
        Positions [deg].
    ebv : array_like
        SFD98 E(B-V) [mag] at the positions.

    Returns
    -------
    ndarray
        rescaled_ebv with the Magellanic Stream distances of the positions [mag].
    """
    # Proper motions are irrelevant here: zeros, so only the position is transformed.
    L, B = ms_coordinates(np.asarray(ra, float), np.asarray(dec, float), np.zeros(np.size(ra)), np.zeros(np.size(ra)))[:2]
    return rescaled_ebv(np.asarray(ebv, float), np.hypot(L - LMC_TARGET_MS[0], B - LMC_TARGET_MS[1]), np.hypot(L - SMC_TARGET_MS[0], B - SMC_TARGET_MS[1]))


def dereddened(g, bp, rp, ebv, lmcrad, smcrad):
    """(G0, (BP - RP)0) with the carton's rescaled SFD reddening.

    Parameters
    ----------
    g, bp, rp : array_like
        Observed Gaia G, BP, RP [mag].
    ebv, lmcrad, smcrad : array_like
        As in rescaled_ebv.

    Returns
    -------
    g0, bprp0 : ndarray
        Dereddened G and BP - RP [mag], with A_V = 0.86 x 3.1 E(B-V) and A_X = A_OVER_AV[X] A_V.
    """
    av = 0.86 * 3.1 * rescaled_ebv(ebv, lmcrad, smcrad)
    return g - A_OVER_AV["G"] * av, (bp - A_OVER_AV["BP"] * av) - (rp - A_OVER_AV["RP"] * av)


def astrometry_ok(ra, dec, pmra, pmdec, parallax, parallax_error):
    """The carton's astrometric cuts: finite, |pmra|, |pmdec| <= 3, parallax and Magellanic Stream PM window.

    Parameters
    ----------
    ra, dec : ndarray
        Positions [deg].
    pmra, pmdec : ndarray
        Proper motions [mas/yr].
    parallax, parallax_error : ndarray
        Gaia parallax and its error [mas].

    Returns
    -------
    ndarray of bool
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(parallax) & (np.abs(pmra) <= 3) & (np.abs(pmdec) <= 3)
        # Reject a significant parallax (> 5 sigma after adding 0.025 mas, the carton's offset): foreground stars.
        ok &= ~((parallax > 0) & ((parallax + 0.025) / parallax_error > 5))
    L, B, pmL, pmB = ms_coordinates(ra, dec, np.nan_to_num(pmra), np.nan_to_num(pmdec))
    return ok & (np.hypot(pmL - 1.8, pmB - 0.40) < 1.2)  # within 1.2 mas/yr of the Magellanic motion in MS coordinates


def eligible(ra, dec, pmra, pmdec, parallax, parallax_error, g, bp, rp, ebv):
    """True for stars the carton selects (arrays; degrees, mas/yr, mas, mag; ebv from SFD98).

    Parameters
    ----------
    ra, dec : ndarray
        Positions [deg].
    pmra, pmdec : ndarray
        Proper motions [mas/yr].
    parallax, parallax_error : ndarray
        [mas].
    g, bp, rp : ndarray
        Observed Gaia DR3 magnitudes [mag].
    ebv : ndarray
        SFD98 E(B-V) [mag].

    Returns
    -------
    ndarray of bool
        All the cuts of the module docstring.
    """
    ok = astrometry_ok(ra, dec, pmra, pmdec, parallax, parallax_error)
    with np.errstate(invalid="ignore"):
        ok &= np.isfinite(g) & np.isfinite(bp) & np.isfinite(rp) & (g < 18.5) & (bp - rp >= 1)
    L, B = ms_coordinates(ra, dec, np.zeros_like(ra), np.zeros_like(ra))[:2]
    g0, bprp0 = dereddened(g, bp, rp, ebv, np.hypot(L - LMC_TARGET_MS[0], B - LMC_TARGET_MS[1]), np.hypot(L - SMC_TARGET_MS[0], B - SMC_TARGET_MS[1]))
    ok &= Path(list(zip(BPRP_CUT, G_CUT))).contains_points(np.column_stack([bprp0, g0]))
    return ok & (g <= 17.5)


def carton_class(cartons):
    """Fibre-priority class from the ';'-joined targetdb cartons of a star: kgiant (halo distant K-giant cartons), xp (XP and WISE
    metal-poor halo cartons), other (any other non-APOGEE carton), magcloud (only this carton, or with APOGEE cartons).

    The classes are tested in this order, so a star in a K-giant carton is 'kgiant' whatever its other cartons.

    Parameters
    ----------
    cartons : pandas.Series of str
        Carton names joined by ';' ('' for none).

    Returns
    -------
    ndarray of str
    """
    other = cartons.str.split(";").apply(lambda l: any(x != "mwm_magcloud_rgb_boss" and "apogee" not in x for x in l if x))
    return np.select([cartons.str.contains("kgiant"), cartons.str.contains("_xp_boss|_wise_boss"), other.to_numpy()],
                     ["kgiant", "xp", "other"], "magcloud")
