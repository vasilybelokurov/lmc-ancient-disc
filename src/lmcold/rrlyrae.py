"""RRab stars: OGLE-IV catalogue reader and photometric [Fe/H] from the light-curve phase difference phi31.

phi31 is defined modulo 2 pi, so each relation first wraps it around the value typical of LMC RRab.

Also: the relative distance modulus from a period-Wesenheit-[Fe/H] relation fitted on the LMC itself, and the Zinn & West ->
Crestani et al. (2021) scale transformation for spectroscopic [Fe/H]. Inputs: the OGLE-IV LMC RR Lyrae collection
(data/sources.yaml keys ogle4_ident and ogle4_rrab; scripts/fetch_ogle4_rrab.py). Used by scripts/build_rrab.py (the RRab catalogue) and the
[Fe/H] figures (scripts/paper_rrab_feh.py, fig_rrab_feh_spec.py). Periods in days, phases in radians, magnitudes in mag.
"""
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord


# OGLE-LMC-RRLYR-39960 onwards were added to the OGLE collection from Gaia: 1,246 stars in the OGLE-IV outer fields "discovered by
# the Gaia team" (Soszynski et al. 2019, AcA 69, 87) and later additions from Gaia DR3 (OGLE README, 2022-08-20 and 2023-09-25).
# Only OGLE's own discoveries are independent of Gaia.
FIRST_GAIA_ADDED = 39960  # running number of the first Gaia-added star


def ogle_own_discovery(ogle_id):
    """True for OGLE-LMC-RRLYR identifiers found by OGLE's own search (not added from Gaia).

    Parameters
    ----------
    ogle_id : iterable of str
        Identifiers OGLE-LMC-RRLYR-NNNNN (the last five characters are the running number).

    Returns
    -------
    ndarray of bool
    """
    return np.array([int(i[-5:]) < FIRST_GAIA_ADDED for i in ogle_id])


def wrap(phi, centre):
    """Angles wrapped into [centre - pi, centre + pi).

    Parameters
    ----------
    phi : array_like
        Angles [rad].
    centre : float
        Centre of the target interval [rad].

    Returns
    -------
    ndarray
    """
    return np.mod(phi - centre + np.pi, 2 * np.pi) + centre - np.pi


def feh_ogle_I(P, phi31_I):
    """[Fe/H] on the Crestani et al. (2021) scale from OGLE I-band period [d] and cosine phi31.

    Martinez-Vazquez et al. (2026, arXiv:2609.27119, eq. A2): I -> V transform of phi31 (Dorfi & Feuchtinger 1999), then the
    Mullen et al. (2021, ApJ 912, 144) V-band RRab relation. The transform has slope 0.788, so the input is wrapped first.

    Parameters
    ----------
    P : array_like
        Period [d].
    phi31_I : array_like
        OGLE I-band phi31 [rad] (cosine Fourier series).

    Returns
    -------
    ndarray
        [Fe/H] [dex], Crestani et al. (2021) scale.
    """
    # phi31_V: I-band phase through the linear I -> V transform, plus pi (eq. A2); then the Mullen et al. relation.
    phi31_V = -0.039 + 0.788 * wrap(phi31_I, 2.5) + np.pi
    return -1.22 - 7.60 * (P - 0.58) + 1.42 * (phi31_V - 5.25)


def feh_gaia_G_m25(P, phi31_g):
    """[Fe/H] from Gaia G-band period [d] and cosine phi31_g (Muraveva et al. 2025, MNRAS 536, 2749, as quoted by Prakash et
    al. 2026, eq. 13).

    Parameters
    ----------
    P : array_like
        Period [d].
    phi31_g : array_like
        Gaia G-band phi31 [rad] (cosine series).

    Returns
    -------
    ndarray
        [Fe/H] [dex].
    """
    return -5.55 * P + 0.94 * wrap(phi31_g, 2.1) - 0.37


def feh_gaia_G_p26(P, phi31_g):
    """[Fe/H] from Gaia G-band period [d] and cosine phi31_g (Prakash et al. 2026, arXiv:2605.13811, eq. 7, sine convention).

    The relation is written for sine-series phi31; the cosine value from Gaia is converted by adding pi.

    Parameters
    ----------
    P : array_like
        Period [d].
    phi31_g : array_like
        Gaia G-band phi31 [rad] (cosine series).

    Returns
    -------
    ndarray
        [Fe/H] [dex].
    """
    return -6.93 - 6.04 * P + 1.65 * wrap(phi31_g + np.pi, 5.25)


def zw_to_c21(feh_zw):
    """Zinn & West (1984) / Harris (1996) scale -> Crestani et al. (2021) scale: Carretta et al. (2009) transformation to their
    UVES scale, [Fe/H]_C09 = 1.105 [Fe/H]_ZW + 0.160, then [Fe/H]_C21 = [Fe/H]_C09 + 0.08 (Martinez-Vazquez et al. 2026).

    Parameters
    ----------
    feh_zw : array_like
        [Fe/H] [dex] on the Zinn & West / Harris scale.

    Returns
    -------
    ndarray
        [Fe/H] [dex] on the Crestani et al. (2021) scale.
    """
    return 1.105 * feh_zw + 0.160 + 0.08


def relative_modulus(W, P, feh, core, nclip=5):
    """Relative distance modulus W - (a + b log10 P + c [Fe/H]), with (a, b, c) fitted on the `core` stars with 3-sigma clipping.

    Returns (dmu, (a, b, c), fitted mask). W, P, feh are arrays; feh must be finite wherever W is.

    Parameters
    ----------
    W : ndarray, shape (N,)
        Wesenheit magnitude [mag] (NaN where unavailable).
    P : ndarray, shape (N,)
        Period [d].
    feh : ndarray, shape (N,)
        [Fe/H] [dex].
    core : ndarray of bool, shape (N,)
        Stars eligible for the fit (scripts/build_rrab.py: rho < 3 deg with measured [Fe/H]).
    nclip : int
        Fit-and-clip iterations; each keeps stars within 3 x 1.4826 MAD of the median residual.

    Returns
    -------
    dmu : ndarray, shape (N,)
        W minus the fitted relation [mag]: distance modulus relative to the mean of the core.
    coef : ndarray, shape (3,)
        a [mag], b [mag per dex in P], c [mag/dex].
    fit : ndarray of bool, shape (N,)
        Stars kept by the last clipping.
    """
    X = np.column_stack([np.ones(len(W)), np.log10(P), feh])
    fit = core & np.isfinite(W) & np.isfinite(X).all(1)
    for _ in range(nclip):
        coef = np.linalg.lstsq(X[fit], W[fit], rcond=None)[0]
        r = W - X @ coef
        med = np.median(r[fit])
        fit &= np.abs(r - med) < 3 * 1.4826 * np.median(np.abs(r[fit] - med))
    return W - X @ coef, coef, fit


def read_ogle_rrab(directory):
    """OGLE-IV ident.dat joined with RRab.dat: one row per RRab, positions in degrees.

    Parameters
    ----------
    directory : pathlib.Path
        Directory holding the OGLE-IV files ident.dat and RRab.dat.

    Returns
    -------
    pandas.DataFrame
        ogle_id, ra, dec [deg], I, V (intensity-mean magnitudes) [mag], P, P_err [d], T0 (maximum light) [HJD - 2450000],
        amp_I [mag],
        R21, phi21, R31, phi31 (I-band Fourier parameters; phases [rad]).
    """
    ident = pd.read_fwf(directory / "ident.dat", header=None, usecols=[0, 1, 2, 3],
                        names=["ogle_id", "type", "ra", "dec"])
    cols = ["ogle_id", "I", "V", "P", "P_err", "T0", "amp_I", "R21", "phi21", "R31", "phi31"]
    pars = pd.read_csv(directory / "RRab.dat", sep=r"\s+", header=None, names=cols, na_values="-")
    rrab = ident.merge(pars, on="ogle_id", validate="one_to_one")
    c = SkyCoord(rrab.ra, rrab.dec, unit=("hourangle", "deg"))
    return rrab.assign(ra=c.ra.deg, dec=c.dec.deg).drop(columns="type")
