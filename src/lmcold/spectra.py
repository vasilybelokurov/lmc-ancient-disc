"""Spectroscopic [Fe/H] of LMC RR Lyrae from low-resolution spectra, one row per star, on the Harris (1996) ~ ZW84 scale.

Gratton et al. (2004): positions from Di Fabrizio et al. (2005) (same fields and star numbers; the mean V must agree, because star
numbers repeat between the two CCD chips of a field). Borissova et al. (2004, 2006): positions from SIMBAD, or from the J2000
position encoded in OGLE-II names.

Inputs (data/sources.yaml, all fetched by scripts/fetch_spec_rrl.py): spec_gratton04 (+ _readme), spec_difabrizio05_A and _B,
spec_borissova04 (+ _readme), spec_borissova06 (+ _readme), spec_simbad_macho. The values are compared with the photometric
[Fe/H] scales of lmcold.rrlyrae (after lmcold.rrlyrae.zw_to_c21) in scripts/paper_rrab_feh.py (figures/paper_rrab_feh.png,
appendix) and scripts/fig_rrab_feh_spec.py.
"""
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import ascii

from .sources import path

# Fixed-width byte ranges (0-based, end exclusive) of the Di Fabrizio et al. (2005) tables 5 and 6: star number, RA (h, m, s),
# Dec (sign, d, m, s) and the intensity-mean V magnitude.
DF05_COLS = [(0, 5), (6, 7), (8, 10), (11, 16), (17, 18), (18, 20), (21, 23), (24, 29), (70, 76)]
DF05_NAMES = ["num", "rah", "ram", "ras", "sign", "ded", "dem", "des", "Vint"]  # column names, one per DF05_COLS range


def cds(name):
    """The CDS table of input `name`, read with its ReadMe (input `name`_readme), as a DataFrame."""
    return ascii.read(path(name), readme=path(name + "_readme"), format="cds").to_pandas()


def gratton_per_star(table):
    """One row per star from Gratton et al. table 7 (one row per spectrum): the star average [Fe/H]a, on each star's first row.

    Parameters
    ----------
    table : pandas.DataFrame
        cds("spec_gratton04").

    Returns
    -------
    pandas.DataFrame
        The rows with [Fe/H]a, one per Star, with the column feh = [Fe/H]a [dex].
    """
    return table[table["[Fe/H]a"].notna()].assign(feh=lambda t: t["[Fe/H]a"]).drop_duplicates("Star")


def position(name, simbad):
    """SIMBAD position of a star name, or the J2000 position encoded in an OGLE-II name (OGLE hhmmss.ss-ddmmss.s).

    Parameters
    ----------
    name : str
        Star name as in the Borissova et al. tables.
    simbad : pandas.DataFrame
        spec_simbad_macho indexed by the queried name, columns ra, dec [deg].

    Returns
    -------
    ra, dec : float
        Degrees; NaN if neither source gives a position.
    """
    if name in simbad.index and np.isfinite(simbad.ra[name]):
        return simbad.ra[name], simbad.dec[name]
    if name.startswith("OGLE ") and len(name) == 23:
        c = SkyCoord(name[5:7] + "h" + name[7:9] + "m" + name[9:14] + "s", name[14:17] + "d" + name[17:19] + "m" + name[19:] + "s")
        return c.ra.deg, c.dec.deg
    return np.nan, np.nan


def read_lmc_rrl_spectra():
    """DataFrame (src, ra, dec, feh) with src G04, B04 or B06; feh on the Harris (1996) scale.

    Returns
    -------
    pandas.DataFrame
        src, ra, dec [deg], feh [dex]; rows without a position or [Fe/H] are dropped.
    """
    df = pd.concat([pd.read_fwf(path(f"spec_difabrizio05_{f}"), colspecs=DF05_COLS, header=None, names=DF05_NAMES,
                                na_values={"Vint": "-"}, keep_default_na=False).assign(field=f) for f in "AB"])
    g = gratton_per_star(cds("spec_gratton04"))
    g = g.assign(field=g.Star.str[0], num=g.Star.str.split("-").str[1].astype(int)).merge(df, on=["field", "num"])
    # Star numbers repeat between the two CCD chips of a field: keep the match whose mean V agrees within 0.05 mag.
    g = g[(g.Vmag - pd.to_numeric(g.Vint)).abs() < 0.05]
    out = [pd.DataFrame(dict(src="G04", ra=15 * (g.rah + g.ram / 60 + g.ras / 3600),
                             dec=np.where(g.sign == "-", -1, 1) * (g.ded + g.dem / 60 + g.des / 3600), feh=g.feh))]
    simbad = pd.read_csv(path("spec_simbad_macho"))
    simbad = simbad.assign(user_specified_id=simbad.user_specified_id.str.strip()).set_index("user_specified_id")
    for name, col, src in (("spec_borissova04", "MACHO", "B04"), ("spec_borissova06", "Name", "B06")):
        b = cds(name)
        # B04 lists bare MACHO numbers; B06 gives full names.
        ids = [("MACHO " + n if src == "B04" else n).strip() for n in b[col].astype(str)]
        pos = np.array([position(n, simbad) for n in ids])
        out.append(pd.DataFrame(dict(src=src, ra=pos[:, 0], dec=pos[:, 1], feh=b["[Fe/H]"])))
    return pd.concat(out, ignore_index=True).dropna()
