"""Spectroscopic stars around the LMC from the public SDSS summary files (fetched by scripts/fetch_giants.py):
APOGEE DR17 allStarLite and SDSS-V DR20 Astra 0.8.1 astraAllStarBossNet. Loose selection, applied identically to both:
within 12 deg of the LMC centre and 100 < v_helio < 450 km/s. Quality cuts come later (scripts/build_giants.py).
Writes data/spec_giants.parquet: survey (apogee | boss), source_id (Gaia DR3), ra, dec, v, ev, feh, teff, logg, snr, and the
quality columns of each survey (APOGEE: extratarg, aspcapflag, vscatter; BOSS: flag_bad).

sources.yaml entries read: apogee_dr17_allstarlite, boss_dr20_bossnet (both checked against their sha256 by lmcold.sources.path).
Output: an intermediate file (not in sources.yaml) that feeds scripts/fetch_gaia_giants.py and scripts/build_giants.py, whose
products are release_giants_apogee and release_giants_boss. Units: ra, dec [deg]; v, ev [km/s, heliocentric]; feh [dex];
teff [K]; logg [dex, cgs]; snr per pixel as given by each pipeline. One row per spectrum summary row, so a star may appear in both
surveys and, in APOGEE, more than once (duplicates are removed in build_giants.py).

Run: .venv/bin/python scripts/build_spec_giants.py
"""
import numpy as np
import pandas as pd
from astropy.io import fits

from lmcold.constants import LMC
from lmcold.footprint import angular_distance
from lmcold.sources import DATA, path

# %% [markdown]
# ### Settings
# The columns read from each summary file and their common names, the same in both surveys.

# %%
# File column -> common column name. APOGEE DR17 carries Gaia EDR3 source_ids, which are the same as the DR3 ones.
APOGEE = {"GAIAEDR3_SOURCE_ID": "source_id", "RA": "ra", "DEC": "dec", "VHELIO_AVG": "v", "VERR": "ev", "FE_H": "feh",
          "TEFF": "teff", "LOGG": "logg", "SNR": "snr", "EXTRATARG": "extratarg", "ASPCAPFLAG": "aspcapflag",
          "VSCATTER": "vscatter"}
BOSS = {"gaia_dr3_source_id": "source_id", "ra": "ra", "dec": "dec", "v_rad": "v", "e_v_rad": "ev", "fe_h": "feh", "teff": "teff",
        "logg": "logg", "snr": "snr", "flag_bad": "flag_bad"}


# %% [markdown]
# ### Reading one summary file
# `read` takes the chosen columns of one FITS file and keeps the rows within 12 deg of the LMC centre with
# heliocentric velocity between 100 and 450 km/s.

# %%
def read(name, cols, hdu):
    """Read the selected columns of one SDSS summary file and apply the loose position and velocity window.

    Parameters
    ----------
    name : str
        sources.yaml entry of the FITS file.
    cols : dict
        File column name -> common column name.
    hdu : int
        FITS extension holding the table.

    Returns
    -------
    pandas.DataFrame
        Rows within 12 deg of the LMC centre (constants.LMC) with 100 < v < 450 km/s, columns renamed by `cols`.
    """
    with fits.open(path(name), memmap=True) as f:
        d = f[hdu].data
        # Columns are converted to native byte order (FITS is big-endian), which pandas needs.
        t = pd.DataFrame({new: np.asarray(d[old]).astype(np.asarray(d[old]).dtype.newbyteorder("=")) for old, new in cols.items()})
    # 12 deg: the radius of the giant samples; 100-450 km/s: a loose heliocentric window around the LMC velocity
    # (rows with NaN v fail it).
    keep = (angular_distance(t.ra, t.dec, LMC["ra0"], LMC["dec0"]) < 12) & (t.v > 100) & (t.v < 450)
    return t[keep.to_numpy()]


# %% [markdown]
# ### Load the data
# The APOGEE DR17 allStarLite and SDSS-V DR20 BOSSNet summary files (fetched by `scripts/fetch_giants.py`), each
# cut to the loose window by `read` and labelled by survey.

# %%
out = pd.concat([read("apogee_dr17_allstarlite", APOGEE, 1).assign(survey="apogee"),
                 read("boss_dr20_bossnet", BOSS, 1).assign(survey="boss")], ignore_index=True)

# %% [markdown]
# ### Write the table
# One row per summary-file row: `data/spec_giants.parquet`, read by `scripts/fetch_gaia_giants.py` and
# `scripts/build_giants.py`.

# %%
out.to_parquet(DATA / "spec_giants.parquet", index=False)
print(out.survey.value_counts().to_dict())
