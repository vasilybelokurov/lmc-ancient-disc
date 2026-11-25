"""Gaia DR3 astrometry and photometry for the spectroscopic giants, from the public Gaia archive (anonymous TAP upload of the
source_ids, in batches). Reads data/spec_giants.parquet (column source_id; written by scripts/build_spec_giants.py from the public
SDSS files); writes data/gaia_giants.parquet.

sources.yaml: data/spec_giants.parquet is listed as release_spec_giants; data/gaia_giants.parquet (release_gaia_giants) is what
scripts/build_giants.py joins to the spectra to produce giants_apogee.parquet and giants_boss.parquet (release_giants_apogee,
release_giants_boss; both list this script among their builders).
Output columns: those of COLS (Gaia DR3 gaia_source: ra, dec [deg]; parallax and errors [mas]; pmra, pmdec and errors [mas/yr];
pmra_pmdec_corr; ruwe; phot_g_mean_mag [mag]). Source ids with no Gaia DR3 row are simply absent.

Needs network access to the Gaia archive (one asynchronous job per batch); no run time is recorded.
Run: .venv/bin/python scripts/fetch_gaia_giants.py   (after build_spec_giants.py)
"""
import numpy as np
import pandas as pd
from astropy.table import Table
from astroquery.gaia import Gaia

from lmcold.sources import DATA, sha256

# %% [markdown]
# ### Settings
# The Gaia DR3 columns fetched and the number of source_ids per uploaded table.

# %%
# gaia_source columns fetched for each giant (units as in Gaia DR3: deg, mas, mas/yr, mag).
COLS = "source_id, ra, dec, parallax, parallax_error, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr, ruwe, phot_g_mean_mag"
BATCH = 50000  # source_ids per uploaded table (one asynchronous job per batch)

# %% [markdown]
# ### Load the data
# The Gaia DR3 source_ids of the spectroscopic giants, each once.

# %%
ids = np.unique(pd.read_parquet(DATA / "spec_giants.parquet").source_id.to_numpy())
ids = ids[ids > 0]  # valid Gaia DR3 source_ids are positive; anything else is a spectrum without a Gaia id
# %% [markdown]
# ### Query the Gaia archive
# The ids are uploaded in batches and joined to gaia_source; the rows of all batches are saved together.

# %%
# One DataFrame of Gaia rows per batch.
out = []
for k in range(0, len(ids), BATCH):
    # Upload the batch as TAP_UPLOAD.ids and inner-join it to gaia_source on source_id (no other condition).
    up = Table({"source_id": ids[k:k + BATCH]})
    job = Gaia.launch_job_async(f"SELECT {', '.join('g.' + c.strip() for c in COLS.split(','))} FROM tap_upload.ids AS u "
                                "JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id",
                                upload_resource=up, upload_table_name="ids", verbose=False)
    out.append(job.get_results().to_pandas())
    print(f"batch {k // BATCH}: {len(out[-1])} rows", flush=True)
# All batches in one table, written with its sha256.
g = pd.concat(out, ignore_index=True)
g.to_parquet(DATA / "gaia_giants.parquet", index=False)
print(len(g), "rows of", len(ids), "ids;", sha256(DATA / "gaia_giants.parquet"))
