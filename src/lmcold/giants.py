"""Spectroscopic giants: quality masks and the BOSS -> APOGEE calibration.

APOGEE DR17 (ASPCAP, R ~ 22,500) is the reference scale. BOSS DR20 (BOSSNet) is placed on it with relations fitted on stars
observed by both surveys (same Gaia DR3 source):
  v:      v_BOSS - v_APOGEE = d0 + d1 t,                                t = (Teff_BOSS - 4200 K) / 1000 K;
  [Fe/H]: [Fe/H]_BOSS = c0 + c1 [Fe/H]_APOGEE + c2 t + c3 s,           s = 20 / (S/N)_BOSS - 1.
The [Fe/H] relation is fitted BOSS-on-APOGEE (APOGEE errors are the smaller) and inverted for BOSS-only stars; this undoes the
BOSSNet scale compression without the attenuation a fit of APOGEE on BOSS would introduce. Fits are robust (Huber weights).

The quality functions take pandas DataFrames with the common column names of data/spec_giants.parquet (built by
scripts/build_spec_giants.py) and return boolean NumPy arrays. The calibration is fitted and applied by
scripts/build_giants.py and shown in figures/paper_boss_calibration.png (appendix B).
"""
import numpy as np

T_PIVOT = 4200.0  # K
ASPCAP_STAR_BAD = 23  # ASPCAPFLAG bit (APOGEE DR17 bitmask STAR_BAD)
EXTRATARG_DUPLICATE = 4  # EXTRATARG bit: duplicate observation of a star
# Below S/N 16 BOSSNet [Fe/H] barely tracks APOGEE on stars with both spectra (slope 0.16 +- 0.23 at S/N 10-13, 0.51 +- 0.10 at
# 13-16, 0.79 +- 0.02 above 40) and the calibrated values run 0.06-0.16 dex high (results/snr_cut_check.json, scripts/fig_snr_cut_check.py).
BOSS_SNR_MIN = 16


def bit(x, b):
    """True where bit b (0 = least significant) of the integer flag x is set; x is cast to int64."""
    return ((np.asarray(x).astype(np.int64) >> b) & 1).astype(bool)


def gaia_ok(g, ruwe_max=1.4):
    """Finite Gaia DR3 proper motions and covariance, RUWE <= ruwe_max (columns pmra, pmdec, pmra_error, pmdec_error,
    pmra_pmdec_corr, ruwe).

    Parameters
    ----------
    g : pandas.DataFrame
        Gaia DR3 columns pmra, pmdec, pmra_error, pmdec_error [mas/yr], pmra_pmdec_corr, ruwe.
    ruwe_max : float
        Largest accepted renormalised unit weight error.

    Returns
    -------
    ndarray of bool
    """
    ok = np.isfinite(g.pmra) & np.isfinite(g.pmdec) & np.isfinite(g.pmra_error) & np.isfinite(g.pmdec_error)
    return (ok & np.isfinite(g.pmra_pmdec_corr) & (g.ruwe <= ruwe_max)).to_numpy()


def apogee_spectrum_ok(a, snr_min=50):
    """APOGEE spectroscopic quality: not a duplicate, ASPCAP STAR_BAD unset, finite [Fe/H], S/N >= snr_min.
    Columns (common names, data/spec_giants.parquet): extratarg, aspcapflag, feh, snr.

    Parameters
    ----------
    a : pandas.DataFrame
        APOGEE stars.
    snr_min : float
        Smallest accepted spectrum S/N (APOGEE SNR, per pixel).

    Returns
    -------
    ndarray of bool
    """
    ok = ~bit(a.extratarg, EXTRATARG_DUPLICATE) & ~bit(a.aspcapflag, ASPCAP_STAR_BAD) & np.isfinite(a.feh)
    return (ok & (a.snr >= snr_min)).to_numpy()


def rv_ok(a, ev_max=1.0, vscatter_max=1.5):
    """Good APOGEE radial velocity: velocity error and visit-to-visit scatter cuts (columns ev, vscatter).

    Parameters
    ----------
    a : pandas.DataFrame
        APOGEE stars: ev (VERR) and vscatter (VSCATTER) [km/s]; a missing vscatter (single visit) counts as 0.
    ev_max, vscatter_max : float
        Upper limits [km/s], exclusive.

    Returns
    -------
    ndarray of bool
    """
    return ((a.ev < ev_max) & (a.vscatter.fillna(0) < vscatter_max)).to_numpy()


def apogee_ok(a):
    """Good APOGEE spectrum and velocity of a giant: apogee_spectrum_ok (S/N >= 50), is_giant and rv_ok with their defaults.

    Parameters
    ----------
    a : pandas.DataFrame
        APOGEE stars (columns of the three functions).

    Returns
    -------
    ndarray of bool
    """
    return apogee_spectrum_ok(a) & is_giant(a) & rv_ok(a)


def is_giant(t, teff_max=5500, logg_max=3.5):
    """Astrophysical giant cuts (they remove contaminants, not failed spectra).

    Parameters
    ----------
    t : pandas.DataFrame
        Columns teff [K] and logg [dex, cgs] from the survey pipeline.
    teff_max : float
        Teff upper limit [K], exclusive.
    logg_max : float
        log g upper limit, exclusive.

    Returns
    -------
    ndarray of bool
        False also where teff or logg is NaN.
    """
    return ((t.teff < teff_max) & (t.logg < logg_max)).to_numpy()


def boss_spectrum_ok(b, snr_min=BOSS_SNR_MIN, ev_max=10.0):
    """BOSS spectroscopic quality: BOSSNet flag_bad unset, finite [Fe/H] and velocity, S/N and velocity-error cuts.
    Columns (common names, data/spec_giants.parquet): flag_bad, feh, v, ev, snr.

    Parameters
    ----------
    b : pandas.DataFrame
        BOSS stars: flag_bad, feh [dex], v, ev [km/s], snr.
    snr_min : float
        Smallest accepted S/N (BOSS_SNR_MIN by default).
    ev_max : float
        Velocity-error upper limit [km/s], exclusive.

    Returns
    -------
    ndarray of bool
    """
    ok = (b.flag_bad == 0) & np.isfinite(b.feh) & np.isfinite(b.v) & np.isfinite(b.ev)
    return (ok & (b.snr >= snr_min) & (b.ev < ev_max)).to_numpy()


def boss_ok(b):
    """Good BOSS spectrum of a giant: boss_spectrum_ok and is_giant with their defaults.

    Parameters
    ----------
    b : pandas.DataFrame
        BOSS stars (columns of the two functions).

    Returns
    -------
    ndarray of bool
    """
    return boss_spectrum_ok(b) & is_giant(b)


def huber_fit(X, y, c=1.5, n_iter=100):
    """Linear least squares with Huber weights. Returns (coefficients, robust sd of the residuals).

    Iteratively reweighted least squares from the ordinary solution: residuals beyond c x the robust sd get weight c s / |r|,
    the others weight 1; stops when the coefficients change by < 1e-10 or after n_iter iterations.

    Parameters
    ----------
    X : ndarray, shape (N, k)
        Design matrix.
    y : ndarray, shape (N,)
        Data.
    c : float
        Huber threshold in units of the robust sd.
    n_iter : int
        Maximum number of reweighting iterations.

    Returns
    -------
    beta : ndarray, shape (k,)
        Coefficients.
    sd : float
        1.4826 x median absolute deviation of the final residuals (Gaussian-equivalent sd), in the units of y.
    """
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-12))
        new = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        if np.allclose(new, beta, atol=1e-10):
            break
        beta = new
    r = y - X @ beta
    return beta, 1.4826 * np.median(np.abs(r - np.median(r)))


def regressors(teff, snr):
    """Calibration regressors t = (Teff - T_PIVOT) / 1000 K and s = 20 / (S/N) - 1 (zero at S/N = 20); see module docstring.

    Parameters
    ----------
    teff : array_like
        BOSS Teff [K].
    snr : array_like
        BOSS S/N.

    Returns
    -------
    t, s : ndarray
    """
    return (np.asarray(teff, float) - T_PIVOT) / 1000.0, 20.0 / np.asarray(snr, float) - 1.0


def fit_calibration(teff_b, snr_b, v_b, v_a, feh_b, feh_a):
    """Fit the BOSS -> APOGEE relations on stars observed by both. Returns dict(d0, d1, c0, c1, c2, c3, sd_v, sd_feh, N).

    Parameters
    ----------
    teff_b, snr_b : array_like
        BOSS Teff [K] and S/N of the common stars.
    v_b, v_a : array_like
        BOSS and APOGEE line-of-sight velocities [km/s].
    feh_b, feh_a : array_like
        BOSS and APOGEE [Fe/H] [dex].

    Returns
    -------
    dict
        d0 [km/s], d1 [km/s per 1000 K]: velocity offset; c0..c3: [Fe/H] relation (module docstring); sd_v [km/s] and
        sd_feh [dex]: robust residual sd of each fit; N: number of stars.
    """
    t, s = regressors(teff_b, snr_b)
    (d0, d1), sd_v = huber_fit(np.column_stack([np.ones_like(t), t]), np.asarray(v_b) - np.asarray(v_a))
    (c0, c1, c2, c3), sd_feh = huber_fit(np.column_stack([np.ones_like(t), feh_a, t, s]), np.asarray(feh_b, float))
    return dict(d0=d0, d1=d1, c0=c0, c1=c1, c2=c2, c3=c3, sd_v=sd_v, sd_feh=sd_feh, N=len(t))


def calibrate(cal, teff_b, snr_b, v_b, feh_b):
    """BOSS velocity and [Fe/H] on the APOGEE scale.

    Parameters
    ----------
    cal : dict
        Output of fit_calibration.
    teff_b, snr_b, v_b, feh_b : array_like
        BOSS Teff [K], S/N, velocity [km/s] and [Fe/H] [dex].

    Returns
    -------
    v, feh : ndarray
        Velocity minus the fitted offset [km/s]; [Fe/H] from inverting [Fe/H]_BOSS = c0 + c1 [Fe/H]_APOGEE + c2 t + c3 s [dex].
    """
    t, s = regressors(teff_b, snr_b)
    v = np.asarray(v_b) - (cal["d0"] + cal["d1"] * t)
    feh = (np.asarray(feh_b) - cal["c0"] - cal["c2"] * t - cal["c3"] * s) / cal["c1"]
    return v, feh
