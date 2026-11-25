"""2MASS Point Source Catalog within 5 deg of the LMC centre, 12 <= H <= 14, from the public IRSA TAP service (table fp_psc).
For the infrared test of the BOSS parent in high extinction (scripts/paper_parent_extinction.py). Writes data/twomass_centre.parquet.

sources.yaml entry written: twomass_centre (259,488 rows). Columns: ra, dec [deg]; j_m, h_m, k_m (2MASS J, H, Ks [mag]);
ph_qual (photometric quality flags, JHK) and cc_flg (contamination and confusion flags).

Needs network access to the IRSA TAP service; no run time is recorded.
Run: .venv/bin/python scripts/fetch_twomass_centre.py
"""
import pyvo

from lmcold.constants import LMC
from lmcold.sources import DATA, sha256

# %% [markdown]
# ### Query and save
# One asynchronous TAP query of the 2MASS Point Source Catalog; the rows are saved as data/twomass_centre.parquet.

# %%
# WHERE: 12 <= H <= 14 mag (both ends inclusive) and within 5 deg of the adopted LMC centre (constants.LMC ra0, dec0).
QUERY = (f"SELECT ra, dec, j_m, h_m, k_m, ph_qual, cc_flg FROM fp_psc WHERE h_m BETWEEN 12 AND 14 AND "
         f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {LMC['ra0']}, {LMC['dec0']}, 5.0)) = 1")
t = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP").run_async(QUERY, maxrec=5_000_000).to_table().to_pandas()
t.to_parquet(DATA / "twomass_centre.parquet", index=False)
print(len(t), "rows;", sha256(DATA / "twomass_centre.parquet"))
