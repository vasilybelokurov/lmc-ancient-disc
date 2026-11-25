"""Gaia XP [M/H] for the parents of the giant selection functions, from Table 1 of Andrae, Rix & Chandra (2023, ApJS 267, 8;
Zenodo 7945154, scripts/fetch_andrae23.py): the BOSS eligible stars (data/boss_parent.parquet) and the APOGEE candidates
(data/apogee_parent.parquet).
In the full catalogue hot stars (Teff > 5500 K, beyond the training set above 7000 K) are assigned spuriously low [M/H]
(Zhang, Ardern-Arentsen & Belokurov 2024, MNRAS 533, 889, Appendix A), so [M/H] is kept only for cool giants:
Teff_XP < TEFF_MAX and log g_XP < LOGG_MAX (mh_xp = NaN otherwise). The vetted Table 2 of Andrae et al. needs precise
parallaxes and holds < 0.2 per cent of the LMC stars, so it is not used.
Writes data/xp_andrae.parquet: source_id, mh_xp, teff_xp, logg_xp.

sources.yaml entry read: andrae23_table1. Other inputs: data/boss_parent.parquet (release_boss_parent, build_boss_parent.py) and
data/apogee_parent.parquet (release_apogee_parent, build_apogee_parent.py). The output (not a sources.yaml entry) is the XP [M/H]
axis of the selection functions (boss_selection.py, apogee_selection.py; figures/paper_selection.png panels b, c) and of the
XP comparison in figures/paper_mass.png (c). Units: mh_xp [dex], teff_xp [K], logg_xp [dex, cgs], all XGBoost estimates.
Parent stars absent from Table 1 are absent from the output.

Run: .venv/bin/python scripts/build_xp_andrae.py   (after build_boss_parent.py and build_apogee_parent.py)
"""
import gzip
import shutil

import numpy as np
import pandas as pd
from astropy.io import fits

from lmcold.sources import DATA, path, sha256

# %% [markdown]
# ### Settings
# The cool-giant limits below which the XP [M/H] is kept.

# %%
TEFF_MAX, LOGG_MAX = 5500.0, 3.5  # K, dex: XP [M/H] kept only below both (cool giants; see docstring)

# %% [markdown]
# ### Decompress Table 1
# The catalogue of Andrae et al. (2023) is decompressed once next to the download.

# %%
# Path of the downloaded gzipped FITS Table 1 (sources.yaml entry andrae23_table1)
gz = path("andrae23_table1")
raw = gz.with_suffix("")  # decompressed once next to the download (memory-mapped below)
if not raw.exists():
    # Stream-decompress in 64 MB chunks (64 << 20 bytes); skipped if the FITS file already exists
    with gzip.open(gz, "rb") as f, open(raw, "wb") as g:
        shutil.copyfileobj(f, g, 64 << 20)

# %% [markdown]
# ### Load the data
# The source_ids of both parents, and the rows of Table 1 for these stars.

# %%
# Union of the parent source_ids (APOGEE candidates without a Gaia match have NaN source_id and are dropped).
ids = np.union1d(pd.read_parquet(DATA / "boss_parent.parquet", columns=["source_id"]).source_id,
                 pd.read_parquet(DATA / "apogee_parent.parquet", columns=["source_id"]).source_id.dropna().astype("int64"))
with fits.open(raw, memmap=True) as f:
    # HDU 1 holds the table; columns are read lazily from the memory map
    d = f[1].data
    sid = np.asarray(d["source_id"]).astype(np.int64)
    k = np.flatnonzero(np.isin(sid, ids))  # row indices of the parent stars in the 175M-row table
    # XGBoost [M/H] [dex], Teff [K] and log g [dex, cgs] of those stars
    t = pd.DataFrame({"source_id": sid[k], "mh_xp": d["mh_xgboost"][k].astype(float), "teff_xp": d["teff_xgboost"][k].astype(float),
                      "logg_xp": d["logg_xgboost"][k].astype(float)})

# %% [markdown]
# ### Cool giants only
# [M/H] is set to NaN outside the cool-giant limits; the table is written to `data/xp_andrae.parquet`.

# %%
# Hot or high-gravity stars: [M/H] unreliable (module docstring), set to NaN; Teff and log g are kept
t.loc[~((t.teff_xp < TEFF_MAX) & (t.logg_xp < LOGG_MAX)), "mh_xp"] = np.nan
# Write the table and print the counts with the sha256 of the output
t.to_parquet(DATA / "xp_andrae.parquet", index=False)
print(f"{len(ids)} parent stars, {len(t)} in Andrae+23 Table 1, {t.mh_xp.notna().sum()} cool giants with [M/H];",
      sha256(DATA / "xp_andrae.parquet"))
