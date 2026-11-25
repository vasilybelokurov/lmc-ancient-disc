"""Parent sample of the APOGEE selection function: 2MASS sources of the APOGEE-2 targeting files of fields LMC1-35 inside a
field's BrtRGB or FntRGB polygon and 0.95 deg of its centre (lmcold.apogee_target), with Gaia DR3 astrometry, observed flags and
the spectroscopic columns of observed stars.

Gaia: gaiadr3.tmass_psc_xsc_best_neighbour (Gaia's own 2MASS cross-match) at Gaia@AIP, the nearest mate where several Gaia
sources share a 2MASS source; `astro_ok` is the SDSS-V carton's
astrometric cuts (lmcold.magcloud.astrometry_ok), the stand-in for the unpublished Washington + DDO51 giant flag and the same
cut as the BOSS parent. observed: an allVisit row of that field (LCO 2.5 m) with APOGEE2_TARGET1 bit 23 (Magellanic Cloud
target) and not bit 22 (literature lists). Spectroscopic columns: the star's allStarLite row (LCO, not a duplicate);
the quality cuts are applied in scripts/apogee_selection.py.
Writes data/apogee_parent.parquet: one row per (field, star).

sources.yaml entries read: apogee2_field, apogee2_object_lmc1 ... apogee2_object_lmc35, apogee_dr17_allvisit,
apogee_dr17_allstarlite; Gaia DR3 is queried live at Gaia@AIP. Entry written: release_apogee_parent.
Output columns: from apogee2Object apogee_id, ra, dec [deg], j, h, k [mag, 2MASS]; nbr (bright neighbour, lmcold.apogee_target),
cls (1 BrtRGB, 2 FntRGB), rfield (distance from the field centre [deg]), field, maglim (FntRGB faint H limit of the field [mag]);
from Gaia gsep (2MASS-Gaia separation [arcsec, Gaia's best-neighbour table]), source_id (nullable Int64; missing if no Gaia
mate), pmra, pmdec [mas/yr], parallax, parallax_error [mas]; astro_ok; observed; and for observed stars the allStarLite
columns of ALLSTAR (feh [dex], snr, teff [K], logg, ev, vscatter, v [km/s], extratarg, aspcapflag). A star lying in two
overlapping fields has two rows.

Run: .venv/bin/python scripts/build_apogee_parent.py
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyvo
from astropy.io import fits

from lmcold import apogee_target as at
from lmcold import giants, magcloud
from lmcold.footprint import angular_distance
from lmcold.sources import DATA, path, sha256

# %% [markdown]
# ### Settings
# The allStarLite columns kept and the query for the Gaia DR3 mates of the 2MASS sources.

# %%
# allStarLite column -> parent column (common names of lmcold.giants).
ALLSTAR = {"APOGEE_ID": "apogee_id", "EXTRATARG": "extratarg", "ASPCAPFLAG": "aspcapflag", "FE_H": "feh", "SNR": "snr",
           "TEFF": "teff", "LOGG": "logg", "VERR": "ev", "VSCATTER": "vscatter", "VHELIO_AVG": "v", "TELESCOPE": "telescope"}
# Gaia DR3 sources with a 2MASS best neighbour inside a cone about a field centre ({r} = field radius + 0.01 deg).
# WHERE: only the cone test; the 2MASS designation (tmass) is matched to the APOGEE_ID later (lmcold.apogee_target.join_gaia).
GAIA = """SELECT t.original_ext_source_id AS tmass, t.angular_distance AS gsep, g.source_id, g.pmra, g.pmdec, g.parallax, g.parallax_error
FROM gaiadr3.gaia_source AS g JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS t ON t.source_id = g.source_id
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra}, {dec}, {r}))"""


# %% [markdown]
# ### Helpers
# `native` turns FITS columns into a table; `field_stars` returns the polygon candidates of one field with
# their Gaia mates.

# %%
def native(d, cols):
    """FITS table columns as a DataFrame: strings stripped, numbers in native byte order.

    Parameters
    ----------
    d : FITS_rec
        Table data.
    cols : dict
        FITS column -> output column.

    Returns
    -------
    pandas.DataFrame
    """
    out = {}
    # Byte strings -> stripped str; big-endian FITS numbers -> native byte order (pandas needs native).
    for old, new in cols.items():
        a = np.asarray(d[old])
        out[new] = np.char.strip(a.astype(str)) if a.dtype.kind in "SU" else a.astype(a.dtype.newbyteorder("="))
    return pd.DataFrame(out)


def field_stars(name, ra, dec):
    """RGB-polygon candidates of one APOGEE-2 LMC field with their Gaia DR3 mate.

    Parameters
    ----------
    name : str
        Field name, 'LMC1' ... 'LMC35'.
    ra, dec : float
        Field centre [deg] (apogee2Field).

    Returns
    -------
    pandas.DataFrame
        Objects of apogee2Object_<name> inside the BrtRGB or FntRGB polygon and within lmcold.apogee_target.RADIUS (0.95 deg)
        of the centre, with the Gaia columns of GAIA (left join: objects without a mate are kept).
    """
    # Targeting file of the field: 2MASS id, position [deg] and J, H, K [mag].
    o = native(fits.getdata(path(f"apogee2_object_{name.lower()}")), {c: c.lower() for c in ("APOGEE_ID", "RA", "DEC", "J", "H", "K")})
    o[["j", "h", "k"]] = o[["j", "h", "k"]].astype(float).round(3)  # 2MASS magnitudes are quoted to 1e-3; float32 breaks edge ties
    o["nbr"] = at.bright_neighbour(o.ra.to_numpy(), o.dec.to_numpy(), o.h.to_numpy())  # on the full object file
    # RGB class in (J - K, H): 0 outside, 1 BrtRGB, 2 FntRGB polygon (H limit of this field); distance from centre [deg].
    o["cls"] = at.rgb_class((o.j - o.k).to_numpy(), o.h.to_numpy(), at.MAGLIM[name])
    o["rfield"] = angular_distance(o.ra, o.dec, ra, dec)
    # Keep polygon members within the field radius (lmcold.apogee_target.RADIUS, 0.95 deg).
    o = o[(o.cls > 0) & (o.rfield <= at.RADIUS)].assign(field=name, maglim=at.MAGLIM[name])
    # Asynchronous TAP job at Gaia@AIP for the cone of the field (wait up to 900 s); the result table as a DataFrame.
    job = pyvo.dal.TAPService("https://gaia.aip.de/tap").submit_job(GAIA.format(ra=ra, dec=dec, r=at.RADIUS + 0.01), maxrec=2_000_000)
    job.run().wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=900)
    job.raise_if_error()
    g = job.fetch_result().to_table().to_pandas()
    # Attach the Gaia mate by 2MASS designation (nearest where several Gaia sources share one 2MASS source).
    return at.join_gaia(o, g)


# %% [markdown]
# ### Load the data
# The field centres (apogee2Field), then for each field its targeting file and its Gaia mates
# (queried live at Gaia@AIP). The allVisit and allStarLite files are read in the steps below.

# %%
# Field centres of LMC1-35 (apogee2Field may list a field more than once; the first row is kept); fields run in parallel.
f = native(fits.getdata(path("apogee2_field")), {"FIELD_NAME": "field", "RA": "ra", "DEC": "dec"}).drop_duplicates("field")
f = f[f.field.isin(at.FIELDS)]
with ThreadPoolExecutor(8) as ex:
    p = pd.concat(ex.map(field_stars, f.field, f.ra, f.dec), ignore_index=True)

# %% [markdown]
# ### Astrometric cuts
# The astrometric cuts of the SDSS-V carton, the stand-in for the unpublished
# Washington + DDO51 giant flag.

# %%
# Boolean (N,): passes the carton's astrometric cuts (lmcold.magcloud.astrometry_ok); False where there is no Gaia mate.
p["astro_ok"] = magcloud.astrometry_ok(*(p[c].to_numpy(float) for c in ("ra", "dec", "pmra", "pmdec", "parallax", "parallax_error")))

# %% [markdown]
# ### Observed flags
# A candidate is observed if allVisit has a visit of it in this field
# targeted as a Magellanic Cloud star.

# %%
with fits.open(path("apogee_dr17_allvisit"), memmap=True) as h:
    v = native(h[1].data, {"APOGEE_ID": "apogee_id", "FIELD": "field", "TELESCOPE": "telescope", "APOGEE2_TARGET1": "t1"})
# Observed: a visit in this field from the LCO 2.5 m telescope with APOGEE2_TARGET1 bit 23 set and bit 22 unset.
v = v[(v.telescope == "lco25m") & v.field.isin(at.FIELDS) & giants.bit(v.t1, 23) & ~giants.bit(v.t1, 22)]
p["observed"] = pd.MultiIndex.from_frame(p[["field", "apogee_id"]]).isin(pd.MultiIndex.from_frame(v[["field", "apogee_id"]]))

# %% [markdown]
# ### Spectroscopic columns
# The allStarLite row of each observed star (LCO, not a duplicate).

# %%
with fits.open(path("apogee_dr17_allstarlite"), memmap=True) as h:
    a = native(h[1].data, ALLSTAR)
# allStarLite rows of the observed stars: LCO, EXTRATARG duplicate bit (4) unset; one row per APOGEE_ID.
a = a[a.apogee_id.isin(p.apogee_id[p.observed]) & (a.telescope == "lco25m") & ~giants.bit(a.extratarg, giants.EXTRATARG_DUPLICATE)]
a = a.drop_duplicates("apogee_id").set_index("apogee_id")
p = p.join(a.drop(columns="telescope"), on="apogee_id")

# %% [markdown]
# ### Write the parent
# One row per (field, star): `data/apogee_parent.parquet`.

# %%
p.to_parquet(DATA / "apogee_parent.parquet", index=False)
print(f"{len(p)} (field, star) rows, {p.apogee_id.nunique()} stars; {p.astro_ok.sum()} astro_ok, {p.observed.sum()} observed,"
      f" {p.snr.notna().sum()} with allStarLite rows;", sha256(DATA / "apogee_parent.parquet"))
