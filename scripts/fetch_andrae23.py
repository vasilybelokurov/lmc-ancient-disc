"""Download Table 1 of Andrae, Rix & Chandra (2023, ApJS 267, 8): XGBoost [M/H], Teff and log g from Gaia DR3 XP spectra for
175 million stars (Zenodo record 7945154, doi:10.5281/zenodo.7945154) into ~/data/gaia/andrae2023. Prints sha256 for
data/sources.yaml.

sources.yaml entry written: andrae23_table1 (tier A). Read by scripts/build_xp_andrae.py, which keeps the stars of the giant
selection-function parents and writes data/xp_andrae.parquet.
The file (~10^8 rows) stays outside the repository; lmcold.download.fetch resumes an interrupted download. The Zenodo record gives
an MD5 (checked by hand, see sources.yaml), not a SHA-1, so no checksum is passed to fetch here.

Needs network access to zenodo.org and room for the file in ~/data; no run time is recorded.
Run: .venv/bin/python scripts/fetch_andrae23.py
"""
from pathlib import Path

from lmcold.download import fetch
from lmcold.sources import sha256

# %% [markdown]
# ### Download the table
# One file from the Zenodo record into ~/data/gaia/andrae2023, then its size [bytes] and sha256 are printed for
# data/sources.yaml.

# %%
# Destination outside the repository; the path matches the `file` field of andrae23_table1.
out = Path("~/data/gaia/andrae2023/table_1_catwise.fits.gz").expanduser()
fetch("https://zenodo.org/records/7945154/files/table_1_catwise.fits.gz", out)
print(out, out.stat().st_size, "sha256", sha256(out), flush=True)
