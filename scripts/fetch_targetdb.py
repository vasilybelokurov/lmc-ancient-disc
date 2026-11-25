"""Download the SDSS-V DR20 targeting database tables (targetdb 2.0.0, parquet; data.sdss.org/sas/dr20/mos/target/2.0.0/) needed to
give every Gaia star its carton memberships and priorities: mos_carton (names, plans), mos_carton_to_target (carton, target,
priority), mos_target (target -> catalogid), mos_catalog_to_gaia_dr3_source (catalogid -> Gaia DR3 source_id),
mos_xpfeh_gaia_dr3 (the Andrae et al. 2023 XP [M/H] used by the SDSS-V XP cartons). SHA-1 from the
release. Into ~/data/sdss/dr20/targetdb/; prints sha256 for data/sources.yaml.

Read by scripts/build_boss_cartons.py (carton memberships and XP [M/H] of the BOSS parent stars). sources.yaml entries:
sdss5_targetdb_<table> for each of the five tables (tier A).

Needs network access to data.sdss.org and room for the tables in ~/data; no size or run time is recorded.
Run: .venv/bin/python scripts/fetch_targetdb.py
"""
from pathlib import Path

import requests

from lmcold.download import fetch
from lmcold.sources import sha256

# %% [markdown]
# ### Download and check
# The release checksum file gives the SHA-1 of each table; each table is downloaded, checked and its sha256 printed.

# %%
URL = "https://data.sdss.org/sas/dr20/mos/target/2.0.0/parquet"
# Release checksum file: lines '<sha1>  <path>/<name>.parquet' -> {name.parquet: sha1}; fetch checks each download.
sha = {l.split()[1].split("/")[-1]: l.split()[0] for l in requests.get(f"{URL}/mos_target_parquet_2.0.0.sha1sum", timeout=60).text.splitlines()}
for name in ("mos_carton", "mos_catalog_to_gaia_dr3_source", "mos_xpfeh_gaia_dr3", "mos_carton_to_target", "mos_target"):
    out = Path(f"~/data/sdss/dr20/targetdb/{name}.parquet").expanduser()
    fetch(f"{URL}/{name}.parquet", out, sha[f"{name}.parquet"])
    print(out, out.stat().st_size, "sha256", sha256(out), flush=True)
