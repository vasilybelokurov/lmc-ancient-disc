"""The kinematic samples as dictionaries of arrays: ra, dec [deg], pmra, pmdec, pmra_error, pmdec_error [mas/yr], pmra_pmdec_corr;
the giants also vlos, evlos [km/s] (APOGEE, or BOSS calibrated to APOGEE).

Inputs:
- RR Lyrae: data/rrab.parquet (sources.yaml: release_rrab; built by scripts/build_rrab.py), members only (rho < 10 deg,
  outside the SMC mask, |dmu| < 0.6 mag; lmcold.footprint) with Gaia DR3 astrometry. No line-of-sight velocities. Relative
  distance modulus dmu [mag] from the OGLE I-band Wesenheit only (core scatter 0.10 mag; the Gaia Wesenheit, 0.19 mag, is NaN here
  so that one scatter parameter describes all moduli in lmcold.kinfit).
- Giants: the release catalogues data/giants_apogee.parquet and data/giants_boss.parquet (sources.yaml: release_giants_apogee,
  release_giants_boss; built by scripts/build_giants.py): APOGEE DR17 and SDSS-V DR20 BOSS giants with Gaia DR3 within 12 deg,
  100 < v_los < 450 km/s, parallax consistent with the LMC, RUWE <= 1.4; APOGEE S/N >= 50 with velocity error < 1 km/s and visit
  scatter < 1.5 km/s; BOSS S/N >= 16 (lmcold.giants.BOSS_SNR_MIN) with [Fe/H] and velocity calibrated to APOGEE. A star in both
  surveys keeps its APOGEE values; every star is in its measured bin. Near the peak of the MDF (bins from [Fe/H] = -1.1 up) the
  BOSS [Fe/H] errors blur the narrow bins (scripts/paper_kin_feh_mixing.py), so there the APOGEE stars alone (survey ==
  'apogee') give the reference values with true [Fe/H]. Milky Way foreground stars inside the velocity window remain; the kinematic model
  (lmcold.kinfit) absorbs them with its outlier component.
Used by scripts/fit_centre.py, scripts/fit_kinematics_mcmc.py, scripts/fig_kin_degeneracy.py and the centre figures.
"""
import pandas as pd

from .sources import DATA, path

# Columns returned for every star (Gaia DR3 names).
COLUMNS = ("ra", "dec", "pmra", "pmdec", "pmra_error", "pmdec_error", "pmra_pmdec_corr")
# Giant [Fe/H] bins [dex], lower edge inclusive; -9 and 9 stand for open ends. Spectroscopic [Fe/H] on the APOGEE scale.
FEH_BINS = [(-9, -1.5), (-1.5, -1.1), (-1.1, -0.8), (-0.8, -0.5), (-0.5, 9)]


def rrab():
    """RRab members with Gaia astrometry.

    Returns
    -------
    dict of ndarray
        COLUMNS for RRab with member True and a Gaia proper motion and correlation (stars found only by OGLE have none), plus dmu
        [mag]: the relative distance modulus where it comes from the OGLE Wesenheit (dmu_src 'I'), else NaN.
    """
    t = pd.read_parquet(DATA / "rrab.parquet")
    t = t[t.member & t.pmra.notna() & t.pmra_pmdec_corr.notna()]
    return {c: t[c].to_numpy() for c in COLUMNS} | dict(dmu=t.dmu.where(t.dmu_src == "I").to_numpy())


def merge_giants(a, b):
    """One table of the APOGEE (a) and BOSS (b) release giants: a star in both keeps its APOGEE row (better velocity and [Fe/H]).

    Parameters
    ----------
    a, b : pandas.DataFrame
        Release catalogues (columns of scripts/build_giants.py, including source_id).

    Returns
    -------
    pandas.DataFrame
        Rows of a, then the rows of b whose source_id is not in a, with a column survey ('apogee' or 'boss').
    """
    return pd.concat([a.assign(survey="apogee"), b[~b.source_id.isin(a.source_id)].assign(survey="boss")], ignore_index=True)


def _bins(g):
    """Table g of giants in the [Fe/H] bins of FEH_BINS: list of (label, dict of COLUMNS plus vlos, evlos, feh, teff, G, snr,
    survey)."""
    out = []
    for lo, hi in FEH_BINS:
        s = g[(g.feh >= lo) & (g.feh < hi)]
        label = f"[Fe/H] < {hi}" if lo < -5 else (f"[Fe/H] > {lo}" if hi > 5 else f"{lo} < [Fe/H] < {hi}")
        out.append((label, {c: s[c].to_numpy() for c in COLUMNS + ("vlos", "evlos", "feh", "teff", "G", "snr", "survey")}))
    return out


def giants():
    """Giants of the release catalogues (merge_giants) in the [Fe/H] bins of FEH_BINS, each in its measured bin (APOGEE scale;
    BOSS calibrated). Select survey == 'apogee' for the APOGEE-only reference values.

    Returns
    -------
    list of (str, dict of ndarray)
        One entry per bin of FEH_BINS: a label such as '-0.8 < [Fe/H] < -0.5' and the COLUMNS plus vlos, evlos [km/s], feh [dex,
        APOGEE scale], teff [K], G [mag], snr and survey ('apogee' or 'boss', the source of the spectrum).
    """
    return _bins(merge_giants(pd.read_parquet(path("release_giants_apogee")), pd.read_parquet(path("release_giants_boss"))))


def overlap():
    """Stars in both release catalogues: feh_a (APOGEE), feh_b (BOSS, calibrated), teff_b, G_b, snr_b (BOSS)."""
    a, b = pd.read_parquet(path("release_giants_apogee")), pd.read_parquet(path("release_giants_boss"))
    return a.merge(b, on="source_id", suffixes=("_a", "_b"))[["feh_a", "feh_b", "teff_b", "G_b", "snr_b"]]
